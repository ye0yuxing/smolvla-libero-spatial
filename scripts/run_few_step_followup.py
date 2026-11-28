"""Run the prespecified noise repeat or two-step distillation after the native comparison."""

import argparse
import fcntl
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from check_run import read_jsonl
from final_protocol import (
    ROOT,
    expected_interface,
    file_sha256,
    interface_hashes,
    load_protocol,
    read_json,
    reference_results,
)
from run_experiment import freeze, run, write_json
from run_few_steps import choose, collect, compare


def status(output, phase, **details):
    write_json(
        output / "status.json",
        {"phase": phase, "time": datetime.now().astimezone().isoformat(), **details},
    )
    print(phase, details, flush=True)


def primary_complete(primary):
    if (
        not (primary / "selection.json").is_file()
        or not (primary / "development/summary.json").is_file()
    ):
        return False
    if read_json(primary / "selection.json")["selected_flow_steps"] is None:
        return True
    decision = primary / "confirmation-decision.json"
    if not (primary / "final/summary.json").is_file() or not decision.is_file():
        return False
    return (
        not read_json(decision)["run_confirmation"]
        or (primary / "confirmation/summary.json").is_file()
    )


def audit_second_seed(primary, repeat, counts, plan):
    results = {}
    for k in counts:
        outcomes = Counter()
        for task in plan["task_ids"]:
            for state in plan["initial_state_ids"]:
                relative = Path(f"task-{task:02d}/flow-{k}/task-{task:02d}_state-{state:02d}")
                old, new = primary / relative, repeat / relative
                a, b = (read_json(p / "episode.json") for p in (old, new))
                if any(a[key] != b[key] for key in ("initial_sim_sha256", "seed", "instruction")):
                    raise ValueError("The noise repeat changed the task or initial simulator state")
                old_draws, new_draws = (read_jsonl(p / "chunks.jsonl") for p in (old, new))
                if any(
                    x["noise_sha256"] == y["noise_sha256"] for x, y in zip(old_draws, new_draws)
                ):
                    raise ValueError("A repeated policy call used its original random draw")
                outcomes[a["success"], b["success"]] += 1
        results[str(k)] = {
            "both": outcomes[True, True],
            "original_only": outcomes[True, False],
            "repeat_only": outcomes[False, True],
            "neither": outcomes[False, False],
        }
    return results


def student_comparisons(folder):
    rows = read_json(folder / "summary.json")["episodes"]
    vs_two = compare(rows, ["flow-2", "student-2"], "flow-2", group="variant")
    write_json(folder / "student-vs-native-two.json", vs_two["student-2"])


def write_summary(output, result):
    lines = ["# Few-step closed-loop follow-up", ""]
    if result["branch"] == "independent_policy_noise":
        sections = [("Same starts with a second policy seed", result["scores"])]
    else:
        sections = [("Student development", result["development"])]
        if "final" in result:
            sections.append(("Student final", result["final"]))
    for title, scores in sections:
        lines.extend(
            [
                f"## {title}",
                "",
                "| Policy | Success | Median ms | P95 ms | Gains vs 10 steps | Losses vs 10 steps |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for label, score in scores.items():
            lines.append(
                f"| {label} | {score['successes']}/{score['episodes']} | "
                f"{score['median_ms']:.1f} | {score['p95_ms']:.1f} | "
                f"{score['candidate_only']} | {score['reference_only']} |"
            )
        lines.append("")
    if result["branch"] == "two_step_distillation":
        for split in ("development", "final"):
            if split in result:
                pair = read_json(output / f"student-{split}" / "student-vs-native-two.json")
                lines.append(
                    f"On {split} starts the student gained {pair['candidate_only']} "
                    f"and lost {pair['reference_only']} placements against native two-step inference."
                )
    else:
        lines += [
            "## Effect of changing only the noise seed",
            "",
            "| Euler steps | Both seeds succeed | Original only | Repeat only | Neither |",
            "|---|---:|---:|---:|---:|",
        ]
        for label, pairs in result["paired_between_seeds"].items():
            lines.append(
                f"| {label} | {pairs['both']} | {pairs['original_only']} | "
                f"{pairs['repeat_only']} | {pairs['neither']} |"
            )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def workflow(primary, model, output, record_path=None):
    plan, official, checkpoint = load_protocol(record_path)
    native_recipe = read_json(ROOT / "configs/few_step_spatial.json")
    recipe = read_json(ROOT / "configs/few_step_followup.json")
    if file_sha256(model / "model.safetensors") != checkpoint[
        "checkpoint_sha256"
    ] or interface_hashes(model) != expected_interface(checkpoint):
        raise ValueError("Teacher weights or processors differ from the supplied checkpoint record")
    record_args = ["--checkpoint-record", record_path] if record_path is not None else []
    original_plan = read_json(primary / "run-plan.json")
    if (
        original_plan["recipe"] != native_recipe
        or original_plan["final"] != plan
        or original_plan["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]
    ):
        raise ValueError(
            "The primary experiment has a different selection rule, split or checkpoint"
        )
    freeze(
        output / "plan.json",
        {"recipe": recipe, "native_recipe": native_recipe, "primary": original_plan},
    )
    status(output, "waiting_for_native_evaluation")
    with (primary / ".few-step.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
    if not primary_complete(primary):
        status(output, "resuming_incomplete_native_evaluation")
        run("run_few_steps.py", "--output", primary, "--checkpoint", model, *record_args)
    if not primary_complete(primary):
        raise ValueError("The native comparison did not finish")
    winner = read_json(primary / "selection.json")["selected_flow_steps"]
    dev_scores = read_json(primary / "development/summary.json")["scores"]
    if choose(dev_scores, plan["flow_steps"], native_recipe) != winner:
        raise ValueError("Stored native selection differs from the development rule")
    _, _, published = load_protocol()
    historical, _ = reference_results(ROOT / "results/evaluation", plan, official, published)
    scale = read_json(ROOT / "configs/final_action_scale.json")["adapted"]
    if winner is not None:
        status(output, "independent_policy_noise", selected_steps=winner)
        states = recipe["repeat_state_ids"]
        if (
            not set(states) <= set(plan["initial_state_ids"])
            or not recipe["policy_noise_seed_offset"]
        ):
            raise ValueError("Noise repeats need original final starts and a different policy seed")
        repeat = {
            **plan,
            "initial_state_ids": states,
            "flow_noise_seed_offset": recipe["policy_noise_seed_offset"],
        }
        counts = [plan["flow_steps"], winner]
        folder = output / "noise-repeat"
        scores = collect(folder, counts, repeat, official, checkpoint, model, scale, historical)
        pairs = audit_second_seed(primary / "final", folder, counts, repeat)
        result = {
            "branch": "independent_policy_noise",
            "selected_flow_steps": winner,
            "scores": scores,
            "paired_between_seeds": pairs,
        }
    else:
        status(output, "training_two_step_student")
        training = output / "distillation"
        run("distill_two_steps.py", "--teacher", model, "--output", training, *record_args)
        record = read_json(training / "student-record.json")
        student = (training / "pretrained_model", record)
        if (
            file_sha256(student[0] / "model.safetensors") != record["checkpoint_sha256"]
            or record["provenance"]["parent_sha256"] != checkpoint["checkpoint_sha256"]
        ):
            raise ValueError("Distilled weights or parent identity differ from the export")
        counts = [recipe["teacher_flow_steps"], recipe["student_flow_steps"]]
        status(output, "student_development_rollouts")
        dev_folder = output / "student-development"
        scores = collect(
            dev_folder,
            counts,
            original_plan["development"],
            official,
            checkpoint,
            model,
            scale,
            student=student,
        )
        student_comparisons(dev_folder)
        ten, two, trained = (scores[k] for k in ("flow-10", "flow-2", "student-2"))
        advances = (
            trained["successes"] - two["successes"] >= recipe["min_student_gain_episodes"]
            and trained["successes"]
            >= ten["successes"] - native_recipe["max_development_loss_episodes"]
            and trained["median_ms"] <= ten["median_ms"] * native_recipe["max_median_latency_ratio"]
        )
        freeze(
            output / "student-selection.json",
            {"advance": advances, "student_sha256": record["checkpoint_sha256"]},
        )
        result = {
            "branch": "two_step_distillation",
            "student_qualified": advances,
            "development": scores,
            "offline_validation": record["offline_validation"],
        }
        if advances:
            status(output, "student_final_rollouts")
            final_folder = output / "student-final"
            result["final"] = collect(
                final_folder,
                counts,
                plan,
                official,
                checkpoint,
                model,
                scale,
                historical,
                student=student,
            )
            student_comparisons(final_folder)
    write_json(output / "result.json", result)
    write_summary(output, result)
    status(output, "complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary", type=Path, required=True, help="Native comparison output directory"
    )
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-record", type=Path, help="Verification record for the teacher"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        PYTHONHASHSEED="0",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
        MUJOCO_GL="egl",
        PYTHONDONTWRITEBYTECODE="1",
    )
    with (output / ".followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            workflow(args.primary.resolve(), args.teacher.resolve(), output, args.checkpoint_record)
        except Exception as error:
            status(output, "failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
