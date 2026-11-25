"""Train the long Spatial recipe and select one checkpoint by paired development rollouts."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time_ns

import imageio_ffmpeg
from final_evidence import FAILURES, check_actions, check_video, diagnose
from final_protocol import (
    ROOT,
    check_shard,
    file_sha256,
    load_protocol,
    read_json,
    reference_results,
)

SCRIPTS = ROOT / "scripts"


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def freeze(path, value):
    if path.is_file() and read_json(path) != value:
        raise ValueError(f"{path}: stored experiment settings or results have changed")
    write_json(path, value)


def run(script, *args, capture=False):
    cmd = [sys.executable, "-u", str(SCRIPTS / script), *map(str, args)]
    print("Running", script, *args, flush=True)
    return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE if capture else None)


def train_to(folder, step, micro):
    latest = max(
        (
            int(p.name)
            for p in (folder / "checkpoints").glob("*")
            if p.is_dir() and p.name.isdecimal()
        ),
        default=0,
    )
    if latest >= step:
        if not (folder / "checkpoints" / f"{step:06d}" / "history.json").is_file():
            raise ValueError(f"Missing expected checkpoint at step {step}")
        return
    if folder.exists() and not latest:
        folder.rename(folder.with_name(f"{folder.name}.incomplete-{time_ns()}"))
    run(
        "train.py",
        "--long-budget",
        "--output",
        folder,
        "--micro-batch-size",
        micro,
        "--stop-at",
        step,
        *(["--resume"] if latest else []),
    )


def checkpoint_record(training, step, baseline, output):
    folder = training / "checkpoints" / f"{step:06d}"
    model = folder / "pretrained_model"
    dest = output / f"checkpoint-{step:06d}.json"
    if dest.is_file():
        record = read_json(dest)
        if file_sha256(model / "model.safetensors") != record["checkpoint_sha256"]:
            raise ValueError(f"Previously verified checkpoint changed at {step}")
    else:
        record = json.loads(run("check_checkpoint.py", model, capture=True).stdout)
    if record["step"] != step or any(
        record[k] != baseline[k]
        for k in (
            "base_sha256",
            "split_sha256",
            "initialization",
            "dataset",
            "seed",
            "train_statistics_verified",
        )
    ):
        raise ValueError("Long and short recipes do not share the intended base and data")
    history = read_json(folder / "history.json")
    if (
        history["recipe"]["precision"] != "fp32_trainable_bf16_autocast"
        or history["promoted_parameters"] <= 0
    ):
        raise ValueError("Missing FP32 training metadata")
    if not (folder / "fp32_master.safetensors").is_file():
        raise ValueError("Missing full-precision resume weights")
    freeze(dest, record)
    return model, record


def rollouts(folder, tasks, plan, official, model, record, runtime, scale, ffmpeg=None):
    folder.parent.mkdir(parents=True, exist_ok=True)

    def verify(path):
        protocol, episodes = check_shard(path, tasks, "adapted", plan, official, record)
        if {k: protocol[k] for k in runtime} != runtime:
            raise ValueError("Evaluation runtime differs from the recorded comparison")
        rows = []
        for ep in episodes:
            task, state = ep["task_id"], ep["initial_state_id"]
            episode_dir = path / f"task-{task:02d}_state-{state:02d}"
            check_actions(episode_dir, scale)
            if ffmpeg:
                check_video(episode_dir / "cameras.mp4", ffmpeg, ep["steps"] + 1)
            rows.append(
                {
                    "task": task,
                    "state": state,
                    "seed": ep["seed"],
                    "initial_sim_sha256": ep["initial_sim_sha256"],
                    "success": ep["success"],
                    "steps": ep["steps"],
                    "policy_calls": ep["chunk_calls"],
                    **diagnose(episode_dir, ep, plan["lift_threshold_m"]),
                }
            )
        return rows

    if folder.exists():
        return verify(folder)
    with TemporaryDirectory(dir=folder.parent, prefix=".in-progress-") as temp:
        new = Path(temp) / "rollouts"
        run(
            "evaluate.py",
            "--output",
            new,
            "--checkpoint",
            model,
            "--record-scene",
            "--task-ids",
            *tasks,
            "--state-ids",
            *plan["initial_state_ids"],
        )
        episodes = verify(new)
        new.rename(folder)
    return episodes


def summarize(rows):
    by_task = {}
    for row in rows:
        t = by_task.setdefault(str(row["task"]), {"successes": 0, "episodes": 0})
        t["episodes"] += 1
        t["successes"] += int(row["success"])
    outcomes = Counter(r["outcome"] for r in rows)
    return {
        "episodes": len(rows),
        "successes": outcomes["success"],
        "failures": {name: outcomes[name] for name in FAILURES},
        "per_task": by_task,
    }


def workflow(output, micro):
    plan, official, baseline = load_protocol()
    recipe = read_json(ROOT / "configs/spatial_long.json")
    reference, runtime = reference_results(ROOT / "results/evaluation", plan, official, baseline)
    dev = {**plan, "initial_state_ids": recipe["development_state_ids"]}
    if set(dev["initial_state_ids"]) & set(plan["initial_state_ids"]):
        raise ValueError("Development and final initial states overlap")
    candidates = recipe["selection_steps"]
    if candidates != sorted(set(candidates)) or candidates[-1] != recipe["steps"]:
        raise ValueError("Invalid candidate checkpoint steps")
    freeze(
        output / "run-plan.json",
        {
            "recipe": {**read_json(ROOT / "configs/spatial_train.json"), **recipe},
            "micro_batch_size": micro,
            "development": dev,
            "final": plan,
            "reference_adapted_sha256": baseline["checkpoint_sha256"],
            "reference_released_sha256": official["weight_sha256"]["policy"],
            "split_sha256": baseline["split_sha256"],
        },
    )
    scale = read_json(ROOT / "configs/final_action_scale.json")["adapted"]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    training = output / "training"
    # These first ten steps are part of the real run and exercise precise save/resume
    # and native-dtype deployment before committing the rest of the GPU budget.
    train_to(training, 10, micro)
    model, record = checkpoint_record(training, 10, baseline, output)
    rollouts(
        output / "preflight",
        [0],
        {**dev, "initial_state_ids": [0]},
        official,
        model,
        record,
        runtime,
        scale,
        ffmpeg,
    )
    scores = {}
    starts = None
    for step in candidates:
        train_to(training, step, micro)
        model, record = checkpoint_record(training, step, baseline, output)
        rows = rollouts(
            output / "development" / f"{step:06d}",
            dev["task_ids"],
            dev,
            official,
            model,
            record,
            runtime,
            scale,
        )
        pairs = {(r["task"], r["state"]): (r["seed"], r["initial_sim_sha256"]) for r in rows}
        if starts is not None and pairs != starts:
            raise ValueError("Candidate checkpoints did not start from identical simulator states")
        starts = pairs
        scores[step] = summarize(rows)
        write_json(output / f"development-{step:06d}.json", {**scores[step], "rows": rows})
        print(f"Development at {step}: {scores[step]['successes']}/{len(rows)}", flush=True)
    winner = max(candidates, key=lambda s: (scores[s]["successes"], -s))
    model, record = checkpoint_record(training, winner, baseline, output)
    freeze(
        output / "selection.json",
        {
            "step": winner,
            "checkpoint_sha256": record["checkpoint_sha256"],
            "tie_break": recipe["selection_tie_break"],
            "development": {str(k): v for k, v in scores.items()},
        },
    )
    print(f"Locked development selection at {winner}. Starting final evaluation.", flush=True)
    final_rows = []
    for task in plan["task_ids"]:
        final_rows.extend(
            rollouts(
                output / "final" / f"task-{task:02d}",
                [task],
                plan,
                official,
                model,
                record,
                runtime,
                scale,
                ffmpeg,
            )
        )
    comparison = {}
    for role in ("adapted", "released", "base"):
        paired = Counter()
        for row in final_rows:
            old = reference[role, row["task"], row["state"]]
            if (int(old["seed"]), old["initial_sim_sha256"]) != (
                row["seed"],
                row["initial_sim_sha256"],
            ):
                raise ValueError("Final comparison starts do not match the reference")
            old_success = old["success"] == "True"
            outcome = {
                (True, True): "both",
                (True, False): "long_only",
                (False, True): "reference_only",
                (False, False): "neither",
            }[row["success"], old_success]
            paired[outcome] += 1
        comparison[role] = {
            k: paired[k] for k in ("both", "long_only", "reference_only", "neither")
        }
    final = {
        "selected_step": winner,
        "checkpoint_sha256": record["checkpoint_sha256"],
        **summarize(final_rows),
        "paired_with": comparison,
        "rows": final_rows,
    }
    write_json(output / "final-summary.json", final)
    print(
        f"Final success {final['successes']}/{final['episodes']}. All paired starts verified.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--micro-batch-size", choices=[16, 32], required=True, type=int)
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
    with (output / ".long.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        workflow(output, args.micro_batch_size)


if __name__ == "__main__":
    main()
