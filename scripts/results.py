"""Build or verify the small, three-policy results table using Python's standard library."""

import argparse
import csv
import json
import math
import re
import tarfile
from collections import Counter
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

from final_evidence import FAILURES
from final_protocol import (
    ROOT,
    file_sha256,
    load_protocol,
    read_json,
    reference_record,
    reference_results,
)
from report_final import write_csv, write_json

POLICIES = ("base", "adapted", "released")


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def episode_metadata(archive):
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith("/episode.json"):
                yield json.load(tar.extractfile(member))


def aggregate(rows, plan):
    states = plan["initial_state_ids"]
    tasks = plan["task_ids"]
    index = {(r["policy"], int(r["task_id"]), int(r["initial_state_id"])): r for r in rows}
    required = {(p, t, s) for p in POLICIES for t in tasks for s in states}
    if set(index) != required or len(rows) != len(required):
        raise ValueError("Incomplete or repeated policy/task/start combination")
    names = read_json(ROOT / "configs/spatial_train.json")["instructions"]
    per_task = []
    for t in tasks:
        counts = {p: 0 for p in POLICIES}
        start_hashes = set()
        for s in states:
            triplet = [index[p, t, s] for p in POLICIES]
            seed = plan["base_seed"] + 1000 * t + s
            digest = triplet[0]["initial_sim_sha256"]
            if not re.fullmatch("[0-9a-f]{64}", digest) or digest in start_hashes:
                raise ValueError(f"Invalid or repeated initial scene on task {t}, start {s}")
            start_hashes.add(digest)
            for row in triplet:
                if int(row["seed"]) != seed or row["initial_sim_sha256"] != digest:
                    raise ValueError(f"Mismatched starting conditions on task {t}, start {s}")
                if row["success"] not in ("True", "False") or row["outcome"] not in (
                    "success",
                    *FAILURES,
                ):
                    raise ValueError("Invalid outcome label")
                ok = row["success"] == "True"
                if ok != (row["outcome"] == "success"):
                    raise ValueError("Success flag disagrees with outcome")
                counts[row["policy"]] += int(ok)
        per_task.append({"task_id": t, "instruction": names[t], "starts": len(states), **counts})
    policies = {}
    for p in POLICIES:
        subset = [r for r in rows if r["policy"] == p]
        outcomes = Counter(r["outcome"] for r in subset)
        times = []
        for r in subset:
            steps = int(r["steps"])
            elapsed = json.loads(r["inference_ms"])
            if not (
                0 < steps <= plan["max_steps"]
                and len(elapsed)
                == int(r["chunk_calls"])
                == math.ceil(steps / plan["n_action_steps"])
                and all(math.isfinite(ms) and ms >= 0 for ms in elapsed)
            ):
                raise ValueError("Invalid control horizon or inference timing")
            times.extend(elapsed)
        policies[p] = {
            "episodes": len(subset),
            "successes": outcomes["success"],
            "actions": sum(int(r["steps"]) for r in subset),
            "chunk_calls": len(times),
            "median_chunk_ms": round(median(times), 2),
            "failure_counts": {name: outcomes[name] for name in FAILURES},
        }
    starts = len(states) * len(tasks)
    delta = 100 * (policies["adapted"]["successes"] - policies["base"]["successes"]) / starts
    return {
        "starts": starts,
        "rollouts": len(rows),
        "adapted_minus_base_pp": delta,
        "policies": policies,
    }, per_task


def collect(reference_dir, base_dir, dest):
    plan, official, checkpoint = load_protocol(reference_record(reference_dir))
    ref, runtime = reference_results(reference_dir, plan, official, checkpoint)
    base_summary = read_json(base_dir / "summary.json")
    checks = read_json(base_dir / "sha256.json")
    if (
        file_sha256(base_dir / "summary.json") != checks["summary.json"]
        or file_sha256(base_dir / "paired-outcomes.csv") != checks["paired-outcomes.csv"]
        or base_summary["reference_episode_table_sha256"]
        != file_sha256(reference_dir / "episodes.csv")
        or base_summary["base_weight_sha256"] != checkpoint["base_sha256"]
        or base_summary["adapted_weight_sha256"] != checkpoint["checkpoint_sha256"]
        or base_summary["runtime"] != runtime
    ):
        raise ValueError("Base and adapted reports do not describe the same experiment")
    for row in read_csv(base_dir / "paired-outcomes.csv"):
        t, s = int(row["task_id"]), int(row["initial_state_id"])
        if ("base", t, s) in ref or any(
            row[p] != ref[p, t, s]["success"] for p in ("released", "adapted")
        ):
            raise ValueError(f"Duplicate or mismatched pair at task {t}, start {s}")
        mapped = dict(
            row,
            policy="base",
            success=row["base"],
            steps=row["base_steps"],
            chunk_calls=row["base_chunk_calls"],
        )
        ref["base", t, s] = {key: mapped[key] for key in ref["adapted", t, s]}
    rows = []
    for role in POLICIES:
        directory = base_dir if role == "base" else reference_dir
        path = directory / f"{role}-rollouts.tar.gz"
        if file_sha256(path) != read_json(directory / "sha256.json")[path.name]:
            raise ValueError(f"Incorrect archive digest for {role}")
        for ep in episode_metadata(path):
            row = dict(ref[role, ep["task_id"], ep["initial_state_id"]])
            if any(
                str(ep[k]) != row[k]
                for k in ("success", "steps", "chunk_calls", "seed", "initial_sim_sha256")
            ):
                raise ValueError("Raw episode disagrees with its outcome row")
            row["inference_ms"] = json.dumps(ep["inference_ms"], separators=(",", ":"))
            rows.append(row)
    rows.sort(
        key=lambda r: (int(r["task_id"]), int(r["initial_state_id"]), POLICIES.index(r["policy"]))
    )
    metrics, tasks = aggregate(rows, plan)
    if {p: metrics["policies"][p]["successes"] for p in POLICIES} != base_summary["successes"]:
        raise ValueError("Raw counts disagree with reported success totals")
    if dest.exists():
        raise FileExistsError(f"Choose a new results directory: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        out = Path(tmp) / "evaluation"
        out.mkdir()
        write_csv(out / "episodes.csv", rows)
        write_csv(out / "per-task.csv", tasks)
        write_json(
            out / "summary.json",
            {**metrics, "runtime": runtime, "base_weight_sha256": checkpoint["base_sha256"]},
        )
        write_json(out / "comparison.json", read_json(reference_dir / "comparison.json"))
        if checkpoint != read_json(ROOT / "results/training/checkpoint.json"):
            write_json(out / "adapted-checkpoint.json", checkpoint)
        write_json(out / "sha256.json", {p.name: file_sha256(p) for p in sorted(out.iterdir())})
        out.rename(dest)


def verify_recorded_selection(plan, checkpoint):
    training = ROOT / "results/training"
    recipe = read_json(ROOT / "configs/spatial_train.json")
    history = read_json(training / "training_history.json")
    selection = read_json(training / "selection.json")
    if (
        history["recipe"] != recipe
        or history["step"] != recipe["steps"]
        or history["split_sha256"] != checkpoint["split_sha256"]
        or file_sha256(training / "spatial_split.json") != checkpoint["split_sha256"]
    ):
        raise ValueError("Recorded training history, recipe and episode split disagree")
    if checkpoint["interface_sha256"] != read_json(ROOT / "configs/spatial_interface_sha256.json"):
        raise ValueError(
            "The selected checkpoint differs from the interface used by the base control"
        )
    scores = {int(s): v for s, v in selection["development"].items()}
    if set(scores) != set(recipe["selection_steps"]):
        raise ValueError("Development results are missing candidate checkpoints")
    starts = len(recipe["development_state_ids"])
    for score in scores.values():
        tasks = score["per_task"]
        if (
            set(map(int, tasks)) != set(plan["task_ids"])
            or any(
                t["episodes"] != starts or not 0 <= t["successes"] <= starts for t in tasks.values()
            )
            or score["episodes"] != len(tasks) * starts
            or score["successes"] != sum(t["successes"] for t in tasks.values())
            or sum(score["failures"].values()) + score["successes"] != score["episodes"]
        ):
            raise ValueError("Inconsistent development coverage or success counts")
    winner = max(scores, key=lambda s: (scores[s]["successes"], -s))
    if (
        selection["tie_break"] != recipe["selection_tie_break"]
        or selection["tie_break"] != "earliest_step"
        or winner != selection["step"]
        or winner != checkpoint["step"]
        or selection["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]
    ):
        raise ValueError(
            "Selected checkpoint does not match the development scores and final weights"
        )
    print(f"Verified checkpoint selection at {winner:,} updates.")


def verify(directory):
    plan, official, checkpoint = load_protocol(reference_record(directory))
    if directory.resolve() == (ROOT / "results/evaluation").resolve():
        verify_recorded_selection(plan, checkpoint)
    reference_results(directory, plan, official, checkpoint)
    hashes = read_json(directory / "sha256.json")
    expected = {"comparison.json", "episodes.csv", "per-task.csv", "summary.json"}
    if (directory / "adapted-checkpoint.json").is_file():
        expected.add("adapted-checkpoint.json")
    if set(hashes) != expected or any(
        file_sha256(directory / name) != hashes[name] for name in expected
    ):
        raise ValueError("Recorded table or manifest digest mismatch")
    metrics, tasks = aggregate(read_csv(directory / "episodes.csv"), plan)
    saved = read_json(directory / "summary.json")
    if (
        any(saved[k] != v for k, v in metrics.items())
        or saved["base_weight_sha256"] != checkpoint["base_sha256"]
    ):
        raise ValueError("Recorded aggregate disagrees with the episode table or weights")
    if [{k: str(v) for k, v in t.items()} for t in tasks] != read_csv(directory / "per-task.csv"):
        raise ValueError("Per-task counts disagree with the episode table")
    print(f"Verified {metrics['rollouts']} rollouts on {metrics['starts']} matched starts.")
    for p in POLICIES:
        print(f"{p}: {metrics['policies'][p]['successes']}/{metrics['starts']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-reports", type=Path, nargs=2, metavar=("ADAPTED_AND_RELEASED", "BASE")
    )
    parser.add_argument("--output", type=Path, help="New directory for the generated tables")
    parser.add_argument(
        "--verify", type=Path, help="Check existing tables (defaults to the recorded results)"
    )
    args = parser.parse_args()
    if args.from_reports:
        if args.output is None or args.verify is not None:
            parser.error("--from-reports requires --output and cannot be combined with --verify")
        collect(*args.from_reports, args.output)
        verify(args.output)
    elif args.output is None:
        verify(args.verify or ROOT / "results/evaluation")
    else:
        parser.error("--output requires --from-reports")


if __name__ == "__main__":
    main()
