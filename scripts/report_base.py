"""Audit the step-zero rollouts and measure the paired change after fine-tuning."""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from final_evidence import check_actions, check_video, diagnose
from final_protocol import (
    ROOT,
    expected_interface,
    file_sha256,
    load_protocol,
    read_json,
    reference_record,
    reference_results,
)
from report_final import EPISODE_FILES, write_archive, write_csv, write_json
from run_base import check_base_task


def build(run, reference_dir, output, ffmpeg):
    plan, official, checkpoint = load_protocol(reference_record(reference_dir))
    reference, runtime = reference_results(reference_dir, plan, official, checkpoint)
    scale = read_json(ROOT / "configs/final_action_scale.json")["adapted"]
    rows, tasks, files = [], [], []
    max_error = 0.0
    for task in plan["task_ids"]:
        shard = run / f"task-{task:02d}"
        episodes = check_base_task(shard, task, plan, official, checkpoint, reference, runtime)
        files.extend(shard / name for name in ("protocol.json", "summary.json"))
        group = []
        for ep in episodes:
            state = ep["initial_state_id"]
            folder = shard / f"task-{task:02d}_state-{state:02d}"
            check_video(folder / "cameras.mp4", ffmpeg, ep["steps"] + 1)
            max_error = max(max_error, check_actions(folder, scale))
            files.extend(folder / name for name in EPISODE_FILES)
            group.append(
                {
                    "task_id": task,
                    "initial_state_id": state,
                    "seed": ep["seed"],
                    "initial_sim_sha256": ep["initial_sim_sha256"],
                    "base": ep["success"],
                    "adapted": reference["adapted", task, state]["success"] == "True",
                    "released": reference["released", task, state]["success"] == "True",
                    "base_steps": ep["steps"],
                    "base_chunk_calls": ep["chunk_calls"],
                    **diagnose(folder, ep, plan["lift_threshold_m"]),
                }
            )
        rows.extend(group)
        tasks.append(
            {
                "task_id": task,
                "instruction": episodes[0]["instruction"],
                "starts": len(group),
                **{
                    role: sum(row[role] for row in group)
                    for role in ("base", "adapted", "released")
                },
            }
        )
    total = len(rows)
    counts = {role: sum(row[role] for row in rows) for role in ("base", "adapted", "released")}
    both = sum(row["base"] and row["adapted"] for row in rows)
    delta = 100 * (counts["adapted"] - counts["base"]) / total
    summary = {
        "starts": total,
        "successes": counts,
        "success_rate": {role: n / total for role, n in counts.items()},
        "adapted_minus_base_pp": delta,
        "paired_base_and_adapted": {
            "both": both,
            "base_only": counts["base"] - both,
            "adapted_only": counts["adapted"] - both,
            "neither": total - counts["base"] - counts["adapted"] + both,
        },
        "base_weight_sha256": checkpoint["base_sha256"],
        "adapted_weight_sha256": checkpoint["checkpoint_sha256"],
        "reference_episode_table_sha256": file_sha256(reference_dir / "episodes.csv"),
        "interface_sha256": expected_interface(checkpoint),
        "decoded_base_frames": sum(row["base_steps"] + 1 for row in rows),
        "max_abs_action_error": max_error,
        "per_task": tasks,
        "runtime": runtime,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        review = Path(tmp) / "review"
        review.mkdir()
        write_json(review / "summary.json", summary)
        write_json(review / "protocol.json", plan)
        write_csv(review / "paired-outcomes.csv", rows)
        write_csv(review / "per-task.csv", tasks)
        write_archive(review / "base-rollouts.tar.gz", run, files)
        paired = summary["paired_base_and_adapted"]
        lines = [
            "# SmolVLA before and after Spatial fine-tuning",
            "",
            (
                f"The original robot-pretrained weights completed **{counts['base']}/{total}** "
                f"placements. The project checkpoint after {checkpoint['step']:,} updates completed "
                f"**{counts['adapted']}/{total}** on those same starts, a change of "
                f"**{delta:+g} percentage points**. The released LIBERO policy completed "
                f"{counts['released']}/{total}."
            ),
            "",
            (
                f"Both the initialization and adapted policy succeeded on {paired['both']} starts. "
                f"The initialization alone succeeded on {paired['base_only']}, the adapted policy "
                f"alone on {paired['adapted_only']}, and both failed on {paired['neither']}."
            ),
            "",
            (
                "The base uses the adapted policy’s LIBERO camera mapping, robot state, action "
                "representation and normalization fitted to the training demonstrations. Its "
                "learned tensors come from the pinned robot-pretrained model with zero Spatial "
                "updates. Each comparison matches the instruction, seed and settled simulator "
                "state. These are additional starts for the same ten instructions used in adaptation."
            ),
            "",
            "| Task | Target bowl location | Base | Adapted | Released |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for t in tasks:
            location = (
                t["instruction"]
                .removeprefix("pick up the black bowl ")
                .removesuffix(" and place it on the plate")
            )
            lines.append(
                f"| {t['task_id']} | {location} | {t['base']}/{t['starts']} "
                f"| {t['adapted']}/{t['starts']} | {t['released']}/{t['starts']} |"
            )
        lines.extend(
            [
                "",
                (
                    "Every base video was decoded and every simulator command was checked "
                    "against its predicted chunk and the shared action scale. [Paired outcomes]"
                    "(paired-outcomes.csv) · [Per-task counts](per-task.csv) · [Run summary]"
                    "(summary.json) · [Protocol](protocol.json) · [Raw rollouts](base-rollouts.tar.gz)"
                ),
                "",
            ]
        )
        (review / "README.md").write_text("\n".join(lines))
        hashes = {p.name: file_sha256(p) for p in sorted(review.iterdir())}
        write_json(review / "sha256.json", hashes)
        if output.exists():
            if read_json(output / "sha256.json") != hashes or any(
                file_sha256(output / k) != v for k, v in hashes.items()
            ):
                raise ValueError("An existing review differs. Choose a fresh destination.")
        else:
            review.rename(output)
    print(
        f"Base {counts['base']}/{total}, adapted {counts['adapted']}/{total}, "
        f"adapted minus base {delta:+g} percentage points.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--reference", type=Path, default=ROOT / "results/evaluation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", help="Path to an FFmpeg decoder")
    args = parser.parse_args()
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    build(args.run.resolve(), args.reference.resolve(), args.output.resolve(), ffmpeg)


if __name__ == "__main__":
    main()
