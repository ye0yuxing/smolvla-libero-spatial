"""Export and independently verify the recorded native sweep and two-step student rollouts."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median, quantiles
from tempfile import TemporaryDirectory

from check_run import read_jsonl
from final_evidence import FAILURES
from final_protocol import (
    ROOT,
    file_sha256,
    load_protocol,
    read_json,
    reference_results,
)
from report_final import write_json

NATIVE = "native-development"
DEVELOPMENT = "student-development"
FINAL = "student-final"
SCORE_FIELDS = ("episodes", "successes", "chunk_calls", "median_ms", "p95_ms")
PAIR_FIELDS = ("both", "reference_only", "candidate_only", "neither")
EPISODE_FIELDS = (
    "task",
    "state",
    "seed",
    "initial_sim_sha256",
    "success",
    "outcome",
    "steps",
    "inference_ms",
)


def metrics(rows):
    times = [ms for row in rows for ms in row["inference_ms"]]
    return {
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "chunk_calls": len(times),
        "median_ms": round(median(times), 4),
        "p95_ms": round(quantiles(times, n=100, method="inclusive")[94], 4),
    }


def export(native, followup, dest):
    """Keep the audited episode evidence and the scores reported by the GPU runner."""
    if dest.exists():
        raise FileExistsError(f"Choose a new output directory: {dest}")
    student = read_json(followup / "distillation/student-record.json")
    teacher = read_json(native / "run-plan.json")["checkpoint_sha256"]
    selected = read_json(followup / "student-selection.json")
    if (
        read_json(native / "selection.json")["selected_flow_steps"] is not None
        or selected != {"advance": True, "student_sha256": student["checkpoint_sha256"]}
        or student["provenance"]["parent_sha256"] != teacher
    ):
        raise ValueError("These runs do not contain a qualified student from the native comparison")
    summaries = {}
    rows = []
    runtime = read_json(native / "development/runtime.json")
    for stage, folder in (
        (NATIVE, native / "development"),
        (DEVELOPMENT, followup / DEVELOPMENT),
        (FINAL, followup / FINAL),
    ):
        if read_json(folder / "runtime.json") != runtime:
            raise ValueError(f"Different runtime in {stage}")
        raw = read_json(folder / "summary.json")
        summaries[stage] = {}
        for label, score in raw["scores"].items():
            variant = f"flow-{label}" if stage == NATIVE else label
            summaries[stage][variant] = {key: round(score[key], 4) for key in SCORE_FIELDS}
        for row in raw["episodes"]:
            variant = row.get("variant", f"flow-{row['flow_steps']}")
            rows.append({"stage": stage, "variant": variant, **{k: row[k] for k in EPISODE_FIELDS}})
    final = read_json(followup / FINAL / "summary.json")
    against_two = read_json(followup / FINAL / "student-vs-native-two.json")
    pairs = {
        label: {key: score[key] for key in PAIR_FIELDS}
        for label, score in (("flow-10", final["scores"]["student-2"]), ("flow-2", against_two))
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as temp:
        out = Path(temp) / "tables"
        out.mkdir()
        with (out / "episodes.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        write_json(out / "student.json", student)
        write_json(
            out / "summary.json",
            {
                "teacher_sha256": teacher,
                "student_sha256": student["checkpoint_sha256"],
                "runtime": runtime,
                "scores": summaries,
                "final_student_pairs": pairs,
                "file_sha256": {
                    name: file_sha256(out / name) for name in ("episodes.jsonl", "student.json")
                },
            },
        )
        verify(out)
        out.rename(dest)


def verify(directory):
    summary = read_json(directory / "summary.json")
    for name in ("episodes.jsonl", "student.json"):
        if file_sha256(directory / name) != summary["file_sha256"][name]:
            raise ValueError(f"Checksum mismatch for {name}")
    student = read_json(directory / "student.json")
    plan, official, teacher = load_protocol()
    reference, _ = reference_results(ROOT / "results/evaluation", plan, official, teacher)
    if (
        student["checkpoint_sha256"] != summary["student_sha256"]
        or student["provenance"]["parent_sha256"] != summary["teacher_sha256"]
        or student["provenance"]["split_sha256"] != teacher["split_sha256"]
        or student["provenance"]["dataset"] != teacher["dataset"]
        or (
            student["provenance"]["teacher_flow_steps"],
            student["provenance"]["student_flow_steps"],
        )
        != (10, 2)
    ):
        raise ValueError("Unexpected student or teacher provenance")
    dev = read_json(ROOT / "configs/spatial_train.json")["development_state_ids"]
    native = read_json(ROOT / "configs/few_step_spatial.json")["development_flow_steps"]
    final_states = plan["initial_state_ids"]
    if set(dev) & set(final_states):
        raise ValueError("Development and final starts overlap")
    variants = ("flow-10", "flow-2", "student-2")
    panels = {
        NATIVE: (tuple(f"flow-{k}" for k in native), dev),
        DEVELOPMENT: (variants, dev),
        FINAL: (variants, final_states),
    }
    rows = read_jsonl(directory / "episodes.jsonl")
    indexed = {(r["stage"], r["variant"], r["task"], r["state"]): r for r in rows}
    expected = {
        (stage, v, t, s)
        for stage, (vs, states) in panels.items()
        for v in vs
        for t in plan["task_ids"]
        for s in states
    }
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError("Missing, repeated or unexpected rollout")
    starts = {}
    for (stage, _, task, state), row in indexed.items():
        digest = row["initial_sim_sha256"]
        if (
            not re.fullmatch("[0-9a-f]{64}", digest)
            or row["seed"] != plan["base_seed"] + 1000 * task + state
            or starts.setdefault((task, state), digest) != digest
            or (
                stage == FINAL and reference["adapted", task, state]["initial_sim_sha256"] != digest
            )
        ):
            raise ValueError("Unpaired simulator start")
        steps, times = row["steps"], row["inference_ms"]
        if (
            type(row["success"]) is not bool
            or row["outcome"] not in ("success", *FAILURES)
            or row["success"] != (row["outcome"] == "success")
            or not 0 < steps <= plan["max_steps"]
            or len(times) != math.ceil(steps / plan["n_action_steps"])
            or any(not math.isfinite(ms) or ms <= 0 for ms in times)
        ):
            raise ValueError("Invalid terminal outcome, control horizon or chunk timings")
    scores = {}
    for stage, (vs, states) in panels.items():
        scores[stage] = {
            v: metrics([indexed[stage, v, t, s] for t in plan["task_ids"] for s in states])
            for v in vs
        }
    if scores != summary["scores"]:
        raise ValueError("Episode data and recorded policy scores disagree")
    for label in ("flow-10", "flow-2"):
        outcomes = Counter(
            (indexed[FINAL, label, t, s]["success"], indexed[FINAL, "student-2", t, s]["success"])
            for t in plan["task_ids"]
            for s in final_states
        )
        pairs = {
            "both": outcomes[True, True],
            "reference_only": outcomes[True, False],
            "candidate_only": outcomes[False, True],
            "neither": outcomes[False, False],
        }
        if pairs != summary["final_student_pairs"][label]:
            raise ValueError(f"Paired outcomes disagree with the recorded comparison to {label}")
    print(
        f"Verified {len(rows)} few-step rollouts across the native sweep, student development and final."
    )
    for label, score in scores[FINAL].items():
        print(
            f"{label}: {score['successes']}/{score['episodes']}, median {score['median_ms']:.2f} ms/chunk"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-runs", nargs=2, type=Path, metavar=("NATIVE", "FOLLOWUP"))
    parser.add_argument("--output", type=Path, help="New directory for the compact evidence")
    parser.add_argument("--verify", type=Path, help="Check a previously exported directory")
    args = parser.parse_args()
    if args.from_runs and args.output and not args.verify:
        export(*args.from_runs, args.output)
    elif not args.from_runs and not args.output:
        verify(args.verify or ROOT / "results/few-steps")
    else:
        parser.error("Use --from-runs with --output, or --verify by itself")
