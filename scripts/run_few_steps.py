"""Compare native SmolVLA Euler step counts on paired LIBERO-Spatial rollouts."""

import argparse
import fcntl
import os
from collections import Counter
from pathlib import Path
from statistics import median, quantiles
from tempfile import TemporaryDirectory

import imageio_ffmpeg
from check_run import read_jsonl
from final_evidence import check_actions, check_video, diagnose
from final_protocol import (
    ROOT,
    check_shard,
    expected_interface,
    file_sha256,
    interface_hashes,
    load_protocol,
    read_json,
    reference_results,
)
from run_experiment import freeze, run, summarize, write_json


def paired_rows(folder, task, counts, plan, official, checkpoint, scale, ffmpeg, student=None):
    """Audit the raw evidence before admitting a task into the comparison."""
    rows = []
    starts = {}
    draws = {}
    runtimes = []
    variants = [(f"flow-{k}", k, "adapted", checkpoint) for k in counts]
    if student is not None:
        variants.append(("student-2", 2, "distilled", student[1]))
    for label, k, role, record in variants:
        root = folder / label
        protocol, episodes = check_shard(
            root,
            task,
            role,
            {**plan, "flow_steps": k, "noise_protocol": "episode_seeded_native_gaussian_sha256"},
            official,
            record,
        )
        runtimes.append({key: protocol[key] for key in ("gpu", "cuda", "mujoco_gl", "versions")})
        for ep in episodes:
            state = ep["initial_state_id"]
            episode_dir = root / f"task-{task:02d}_state-{state:02d}"
            start = (ep["seed"], ep["initial_sim_sha256"])
            if starts.setdefault(state, start) != start:
                raise ValueError(f"Different simulator starts for task {task}, state {state}")
            for chunk in read_jsonl(episode_dir / "chunks.jsonl"):
                digest = chunk["noise_sha256"]
                key = (state, chunk["step"])
                if len(bytes.fromhex(digest)) != 32 or draws.setdefault(key, digest) != digest:
                    raise ValueError(f"Different flow noise for task {task}, state/step {key}")
            check_actions(episode_dir, scale)
            check_video(episode_dir / "cameras.mp4", ffmpeg, ep["steps"] + 1)
            rows.append(
                {
                    "task": task,
                    "state": state,
                    "flow_steps": k,
                    "variant": label,
                    "seed": ep["seed"],
                    "initial_sim_sha256": ep["initial_sim_sha256"],
                    "success": ep["success"],
                    "steps": ep["steps"],
                    "inference_ms": ep["inference_ms"],
                    **diagnose(episode_dir, ep, plan["lift_threshold_m"]),
                }
            )
    if any(runtime != runtimes[0] for runtime in runtimes):
        raise ValueError("Runtime changed between paired configurations")
    return rows, runtimes[0]


def compare(rows, counts, baseline, *, group="flow_steps"):
    scores = {}
    reference = {(r["task"], r["state"]): r for r in rows if r[group] == baseline}
    for k in counts:
        selected = [r for r in rows if r[group] == k]
        times = [t for r in selected for t in r["inference_ms"]]
        pairs = Counter()
        flips = []
        for row in selected:
            other = reference[row["task"], row["state"]]
            pairs[other["success"], row["success"]] += 1
            if other["success"] != row["success"]:
                flips.append(
                    {
                        "task": row["task"],
                        "state": row["state"],
                        "reference": other["outcome"],
                        "candidate": row["outcome"],
                    }
                )
        scores[str(k)] = {
            **summarize(selected),
            "chunk_calls": len(times),
            "median_ms": median(times),
            "p95_ms": quantiles(times, n=100, method="inclusive")[94],
            "both": pairs[True, True],
            "reference_only": pairs[True, False],
            "candidate_only": pairs[False, True],
            "neither": pairs[False, False],
            "flipped_starts": flips,
        }
    return scores


def collect(
    output, counts, plan, official, checkpoint, model, scale, historical=None, *, student=None
):
    output.mkdir(exist_ok=True)
    rows = []
    baseline = plan["flow_steps"]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    extra = []
    if student is not None:
        extra += ["--student-checkpoint", student[0]]
    if plan.get("flow_noise_seed_offset", 0):
        extra += ["--flow-noise-seed-offset", plan["flow_noise_seed_offset"]]
    for task in plan["task_ids"]:
        shard = output / f"task-{task:02d}"
        if shard.is_dir():
            batch, runtime = paired_rows(
                shard, task, counts, plan, official, checkpoint, scale, ffmpeg, student
            )
        else:
            with TemporaryDirectory(dir=output, prefix=f".task-{task:02d}-") as temp:
                fresh = Path(temp) / "rollouts"
                run(
                    "evaluate.py",
                    "--output",
                    fresh,
                    "--checkpoint",
                    model,
                    "--task-ids",
                    task,
                    "--state-ids",
                    *plan["initial_state_ids"],
                    "--record-scene",
                    "--flow-steps",
                    *counts,
                    *extra,
                )
                batch, runtime = paired_rows(
                    fresh, task, counts, plan, official, checkpoint, scale, ffmpeg, student
                )
                fresh.rename(shard)
        freeze(output / "runtime.json", runtime)
        if historical is not None:
            for row in batch:
                ref = historical["adapted", task, row["state"]]
                if (int(ref["seed"]), ref["initial_sim_sha256"]) != (
                    row["seed"],
                    row["initial_sim_sha256"],
                ):
                    raise ValueError("Final starts differ from the published evaluation")
        rows.extend(batch)
        write_json(
            output / "progress.json", {"verified_rollouts": len(rows), "last_verified_task": task}
        )
        print(f"{output.name}: verified {len(rows)} rollouts through task {task:02d}", flush=True)
    if student is None:
        scores = compare(rows, counts, baseline)
    else:
        labels = [f"flow-{k}" for k in counts] + ["student-2"]
        scores = compare(rows, labels, f"flow-{baseline}", group="variant")
    write_json(output / "summary.json", {"scores": scores, "episodes": rows})
    for k, score in scores.items():
        print(
            f"{output.name}  K={k}  {score['successes']}/{score['episodes']}  "
            f"median={score['median_ms']:.1f} ms  P95={score['p95_ms']:.1f} ms",
            flush=True,
        )
    return scores


def choose(scores, baseline, recipe):
    base = scores[str(baseline)]
    eligible = []
    for k in recipe["low_step_candidates"]:
        score = scores[str(k)]
        if (
            score["successes"] >= base["successes"] - recipe["max_development_loss_episodes"]
            and score["median_ms"] <= base["median_ms"] * recipe["max_median_latency_ratio"]
        ):
            eligible.append(k)
    return min(eligible, key=lambda k: (scores[str(k)]["median_ms"], k), default=None)


def workflow(output, model, record_path):
    plan, official, checkpoint = load_protocol(record_path)
    published_plan, _, published = load_protocol()
    historical, _ = reference_results(
        ROOT / "results/evaluation", published_plan, official, published
    )
    if file_sha256(model / "model.safetensors") != checkpoint["checkpoint_sha256"]:
        raise ValueError("Model weights differ from the checkpoint record")
    if interface_hashes(model) != expected_interface(checkpoint):
        raise ValueError("Input or output processors differ from the checkpoint record")
    recipe = read_json(ROOT / "configs/few_step_spatial.json")
    dev = {
        **plan,
        "initial_state_ids": read_json(ROOT / "configs/spatial_train.json")[
            "development_state_ids"
        ],
    }
    reserve = {**plan, "initial_state_ids": recipe["confirmation_state_ids"]}
    splits = [set(p["initial_state_ids"]) for p in (dev, plan, reserve)]
    if sum(map(len, splits)) != len(set.union(*splits)):
        raise ValueError("Development, final and confirmation starts overlap")
    counts = recipe["development_flow_steps"]
    baseline = plan["flow_steps"]
    freeze(
        output / "run-plan.json",
        {
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "interface_sha256": expected_interface(checkpoint),
            "recipe": recipe,
            "development": dev,
            "final": plan,
            "confirmation": reserve,
        },
    )
    scale = read_json(ROOT / "configs/final_action_scale.json")["adapted"]
    scores = collect(output / "development", counts, dev, official, checkpoint, model, scale)
    winner = choose(scores, baseline, recipe)
    freeze(
        output / "selection.json",
        {
            "selected_flow_steps": winner,
            "next": "paired_final" if winner is not None else "review_two_step_failures",
        },
    )
    if winner is None:
        print("Review development flips before training a two-step student.", flush=True)
        return
    print(f"Locked K={winner} on development. Starting the paired final comparison.", flush=True)
    scores = collect(
        output / "final", [baseline, winner], plan, official, checkpoint, model, scale, historical
    )
    candidate = scores[str(winner)]
    delta = candidate["successes"] - scores[str(baseline)]["successes"]
    n = candidate["episodes"]
    gap_pp = 100 * delta / n
    margin = recipe["final_loss_margin_episodes_per_100"]
    band = recipe["borderline_episodes_per_100"]
    repeat = bool(candidate["flipped_starts"]) and (
        abs(gap_pp) <= band or abs(gap_pp + margin) <= band
    )
    freeze(
        output / "confirmation-decision.json",
        {"final_delta_pp": gap_pp, "run_confirmation": repeat},
    )
    if repeat:
        print(
            "Final success gap falls in the prespecified band. Checking fresh starts.", flush=True
        )
        collect(
            output / "confirmation", [baseline, winner], reserve, official, checkpoint, model, scale
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-record", type=Path, help="Record for a locally reproduced checkpoint"
    )
    args = parser.parse_args()
    os.environ.update(
        HF_HUB_OFFLINE="1",
        PYTHONHASHSEED="0",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
        MUJOCO_GL="egl",
        PYTHONDONTWRITEBYTECODE="1",
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".few-step.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        workflow(output, args.checkpoint.resolve(), args.checkpoint_record)


if __name__ == "__main__":
    main()
