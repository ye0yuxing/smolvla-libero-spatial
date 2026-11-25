"""Audit paired rollouts and build a portable report, videos and raw evidence archives."""

import argparse
import csv
import gzip
import json
import re
import shutil
import tarfile
from collections import Counter
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

from final_evidence import (
    FAILURE_LABELS,
    FAILURES,
    OUTCOMES,
    check_actions,
    check_video,
    diagnose,
    paired_outcome,
)
from final_protocol import (
    ROLES,
    ROOT,
    check_shard,
    file_sha256,
    load_protocol,
    read_json,
)

EPISODE_FILES = ("episode.json", "steps.jsonl", "chunks.jsonl", "scene.jsonl", "cameras.mp4")


def audit(run, plan, official, checkpoint, ffmpeg):
    expected = {
        "plan": plan,
        "released_sha256": official["weight_sha256"]["policy"],
        "adapted_sha256": checkpoint["checkpoint_sha256"],
    }
    if read_json(run / "comparison.json") != expected:
        raise ValueError("Frozen comparison does not match the protocol")
    instructions = read_json(ROOT / "configs/spatial_train.json")["instructions"]
    scales = read_json(ROOT / "configs/final_action_scale.json")
    rows, pairs, examples, raw_files = [], [], {}, {role: [] for role in ROLES}
    times = {role: [] for role in ROLES}
    max_error = {role: 0.0 for role in ROLES}
    machine = None
    for task in plan["task_ids"]:
        by_state = {}
        for role in ROLES:
            shard = run / role / f"task-{task:02d}"
            protocol, episodes = check_shard(shard, task, role, plan, official, checkpoint)
            if len({ep["initial_sim_sha256"] for ep in episodes}) != len(episodes):
                raise ValueError(f"{shard}: multiple state IDs produced the same settled state")
            runtime = {key: protocol[key] for key in ("gpu", "cuda", "mujoco_gl", "versions")}
            if machine is not None and runtime != machine:
                raise ValueError(f"{shard}: runtime differs across shards")
            machine = runtime
            raw_files[role].extend(shard / name for name in ("protocol.json", "summary.json"))
            for ep in episodes:
                state = ep["initial_state_id"]
                folder = shard / f"task-{task:02d}_state-{state:02d}"
                if ep["instruction"] != instructions[task] or not re.fullmatch(
                    "[0-9a-f]{64}", ep["initial_sim_sha256"]
                ):
                    raise ValueError(
                        f"{folder}: instruction or initial simulator checksum is invalid"
                    )
                check_video(folder / "cameras.mp4", ffmpeg, ep["steps"] + 1)
                max_error[role] = max(max_error[role], check_actions(folder, scales[role]))
                times[role].extend(ep["inference_ms"])
                raw_files[role].extend(folder / name for name in EPISODE_FILES)
                row = {
                    "policy": role,
                    "task_id": task,
                    "initial_state_id": state,
                    "seed": ep["seed"],
                    "initial_sim_sha256": ep["initial_sim_sha256"],
                    "success": ep["success"],
                    "steps": ep["steps"],
                    "chunk_calls": ep["chunk_calls"],
                    **diagnose(folder, ep, plan["lift_threshold_m"]),
                }
                rows.append(row)
                by_state.setdefault(state, {})[role] = row
        for state in sorted(by_state):
            released, adapted = (by_state[state][r] for r in ROLES)
            if any(released[key] != adapted[key] for key in ("seed", "initial_sim_sha256")):
                raise ValueError(
                    f"Task {task}, state {state}: policies started in different simulator states"
                )
            outcome = paired_outcome(released["success"], adapted["success"])
            pair = {
                "task_id": task,
                "initial_state_id": state,
                "seed": released["seed"],
                "released_success": released["success"],
                "adapted_success": adapted["success"],
                "paired_outcome": outcome,
            }
            pairs.append(pair)
            examples.setdefault(outcome, by_state[state])
    if not pairs:
        raise ValueError("Final comparison is empty")
    metrics = {}
    for role in ROLES:
        sub = [row for row in rows if row["policy"] == role]
        ok = [row["steps"] for row in sub if row["success"]]
        counts = Counter(row["outcome"] for row in sub)
        metrics[role] = {
            "successes": len(ok),
            "episodes": len(sub),
            "actions": sum(row["steps"] for row in sub),
            "chunk_calls": len(times[role]),
            "median_success_steps": median(ok) if ok else None,
            "median_chunk_ms": round(median(times[role]), 2),
            "decoded_frames": sum(row["steps"] + 1 for row in sub),
            "max_abs_action_error": max_error[role],
            "failure_counts": {key: counts[key] for key in FAILURES},
            "failures_with_other_bowl_contact": sum(
                row["other_bowl_contact"] for row in sub if not row["success"]
            ),
        }
    tasks = []
    for task in plan["task_ids"]:
        group = [p for p in pairs if p["task_id"] == task]
        tasks.append(
            {
                "task_id": task,
                "instruction": instructions[task],
                "pairs": len(group),
                **{role: sum(p[f"{role}_success"] for p in group) for role in ROLES},
            }
        )
    outcomes = Counter(p["paired_outcome"] for p in pairs)
    summary = {
        "pair_count": len(pairs),
        "policies": metrics,
        "paired_counts": {key: outcomes[key] for key in OUTCOMES},
        "adapted_minus_released_pp": round(
            100 * (metrics["adapted"]["successes"] - metrics["released"]["successes"]) / len(pairs),
            2,
        ),
        "per_task": tasks,
        "runtime": machine,
    }
    return rows, pairs, examples, raw_files, summary


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_archive(path, run, files):
    # Only whitelisted research artifacts enter the portable archives.
    with (
        path.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=5, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as archive,
    ):
        for source in sorted(files):
            info = tarfile.TarInfo(source.relative_to(run).as_posix())
            info.mode, info.size = 0o644, source.stat().st_size
            with source.open("rb") as stream:
                archive.addfile(info, stream)


def write_readme(out, summary, examples, run, threshold):
    policies = summary["policies"]
    total = summary["pair_count"]
    released, adapted = (policies[role]["successes"] for role in ROLES)
    delta = summary["adapted_minus_released_pp"]
    lines = [
        "# Paired closed-loop evaluation on LIBERO-Spatial",
        "",
        (
            f"The project-trained policy completed **{adapted}/{total}** placements. The released "
            f"LIBERO checkpoint completed **{released}/{total}** on the same starts. The difference "
            f"in success rate was **{delta:+g} percentage points** (adapted minus released). Each "
            "pair shared its instruction, seed and post-settling simulator-state digest."
        ),
        "",
        (
            "These outcomes cover additional starting configurations for ten instructions represented "
            "in adaptation. Training histories and budgets differ between checkpoints. The comparison "
            "measures execution under the fixed protocol. [Protocol and weight digests](comparison.json) "
            "· [Adapted checkpoint provenance](adapted-checkpoint.json)"
        ),
        "",
        "## Success by target location",
        "",
        "| Task | Target bowl location | Released | Adapted |",
        "| --- | --- | ---: | ---: |",
    ]
    for task in summary["per_task"]:
        cue = (
            task["instruction"]
            .removeprefix("pick up the black bowl ")
            .removesuffix(" and place it on the plate")
        )
        lines.append(
            f"| {task['task_id']} | {cue} | {task['released']}/{task['pairs']} "
            f"| {task['adapted']}/{task['pairs']} |"
        )
    counts = summary["paired_counts"]
    lines += [
        "",
        (
            f"Both policies succeeded on {counts['both']} starts. Only the released policy succeeded "
            f"on {counts['released_only']}, only the adapted policy on {counts['adapted_only']}, and "
            f"both failed on {counts['neither']}. [Per-task results](per-task.csv) · "
            "[One row per matched start](pairs.csv) · [One row per rollout](episodes.csv)"
        ),
        "",
        "## Execution and failures",
        "",
        "| Measurement | Released | Adapted |",
        "| --- | ---: | ---: |",
    ]
    for label, field in (
        ("Control actions", "actions"),
        ("Chunk predictions", "chunk_calls"),
        ("Median actions to success", "median_success_steps"),
        ("Median synchronized chunk prediction (ms)", "median_chunk_ms"),
    ):
        values = [policies[role][field] for role in ROLES]
        cells = ["—" if v is None else f"{v:,g}" for v in values]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    for key in FAILURES:
        lines.append(
            f"| {FAILURE_LABELS[key]} | {policies['released']['failure_counts'][key]} "
            f"| {policies['adapted']['failure_counts'][key]} |"
        )
    lines += [
        "",
        (
            f"Lift means the target bowl rose at least {threshold * 100:g} cm above its settled "
            "height while touching both fingerpad groups in a sampled frame. LIBERO determines "
            "placement success. Among unsuccessful rollouts, another black bowl contacted both "
            f"fingerpad groups in {policies['released']['failures_with_other_bowl_contact']} released "
            f"and {policies['adapted']['failures_with_other_bowl_contact']} adapted episodes. "
            "Contact, height and plate distance are observations from the simulator. The model "
            "received only the camera images, robot state and instruction."
        ),
        "",
        "## Matched examples",
        "",
        (
            "The first task and state in numeric order for each observed paired outcome appears "
            "below. Each video shows the external view on the left and the wrist camera on the right. "
            "Frame zero follows settling. An empty contact or lift entry means that event was "
            "unobserved at the 20 Hz sampling rate."
        ),
        "",
        (
            "| Pair outcome | Task / state | Policy and video | Outcome | Actions | First target "
            "contact / lift frame | Final bowl–plate horizontal separation |"
        ),
        "| --- | --- | --- | --- | ---: | --- | ---: |",
    ]
    clips = out / "examples"
    clips.mkdir()
    for outcome in OUTCOMES:
        if outcome not in examples:
            continue
        for role in ROLES:
            row = examples[outcome][role]
            task, state = row["task_id"], row["initial_state_id"]
            ident = f"task-{task:02d}_state-{state:02d}"
            clip = f"{ident}_{role}.mp4"
            shutil.copyfile(run / role / f"task-{task:02d}" / ident / "cameras.mp4", clips / clip)
            contact, lift = row["first_target_contact_frame"], row["first_lift_frame"]
            frames = (
                f"{contact if contact is not None else '—'} / {lift if lift is not None else '—'}"
            )
            desc = "Successful placement" if row["success"] else FAILURE_LABELS[row["outcome"]]
            lines.append(
                f"| {outcome.replace('_', ' ')} | {task} / {state} | [{role}](examples/{clip}) "
                f"| {desc} | {row['steps']} | {frames} | {100 * row['final_target_plate_xy_m']:.1f} cm |"
            )
    lines += [
        "",
        "## Recheck the evidence",
        "",
        (
            "Every episode includes both cameras, commands, predicted chunks, simulator measurements "
            "and termination flags. The audit decoded every video, checked transition continuity, "
            "reconstructed every applied command with the corresponding checkpoint’s action scale and "
            "compared the settled state across each pair. The archives contain the full per-task "
            "manifests and raw recordings. [Summary and runtime versions](summary.json) · "
            "[File digests](sha256.json) · [Released rollouts](released-rollouts.tar.gz) · "
            "[Adapted rollouts](adapted-rollouts.tar.gz)"
        ),
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build(run, output, ffmpeg):
    record = run / "adapted-checkpoint.json"
    plan, official, checkpoint = load_protocol(record if record.is_file() else None)
    rows, pairs, examples, files, summary = audit(run, plan, official, checkpoint, ffmpeg)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        out = Path(tmp) / "review"
        out.mkdir()
        write_csv(out / "episodes.csv", rows)
        write_csv(out / "pairs.csv", pairs)
        write_csv(out / "per-task.csv", summary["per_task"])
        write_json(out / "summary.json", summary)
        write_json(out / "adapted-checkpoint.json", checkpoint)
        shutil.copyfile(run / "comparison.json", out / "comparison.json")
        for role in ROLES:
            write_archive(out / f"{role}-rollouts.tar.gz", run, files[role])
        write_readme(out, summary, examples, run, plan["lift_threshold_m"])
        hashes = {
            p.relative_to(out).as_posix(): file_sha256(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        }
        write_json(out / "sha256.json", hashes)
        if output.exists():
            if read_json(output / "sha256.json") != hashes or any(
                file_sha256(output / k) != v for k, v in hashes.items()
            ):
                raise ValueError("Existing report differs from this run. Choose a new destination.")
        else:
            out.rename(output)
    print(f"Verified {len(rows)} rollouts and {len(pairs)} matched initial states.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", type=Path, help="Run directory or unpacked archives and comparison.json"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--ffmpeg", help="Video decoder path (defaults to ffmpeg on PATH or imageio-ffmpeg)"
    )
    args = parser.parse_args()
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    build(args.run.resolve(), args.output.resolve(), ffmpeg)


if __name__ == "__main__":
    main()
