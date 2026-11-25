"""Shared definitions for collecting and auditing the final paired comparison."""

import csv
import hashlib
import json
from pathlib import Path

from check_run import check_run

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("released", "adapted")
RECORDED_CHECKPOINT = ROOT / "results/training/checkpoint.json"


def read_json(path):
    return json.loads(path.read_text())


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def interface_hashes(model_dir):
    files = {"config.json", "policy_preprocessor.json", "policy_postprocessor.json"}
    for pipeline in ("policy_preprocessor.json", "policy_postprocessor.json"):
        files.update(
            step["state_file"]
            for step in read_json(model_dir / pipeline)["steps"]
            if step.get("state_file")
        )
    return {name: file_sha256(model_dir / name) for name in sorted(files)}


def expected_interface(checkpoint):
    if "interface_sha256" in checkpoint:
        return checkpoint["interface_sha256"]
    if checkpoint["checkpoint_sha256"] == read_json(RECORDED_CHECKPOINT)["checkpoint_sha256"]:
        return read_json(ROOT / "configs/spatial_interface_sha256.json")
    raise ValueError(
        "Generate the checkpoint record with scripts/check_checkpoint.py to identify its interface"
    )


def load_protocol(checkpoint_record=None):
    plan = read_json(ROOT / "configs/final_spatial.json")
    official = read_json(ROOT / "configs/official_spatial.json")
    training = read_json(ROOT / "configs/spatial_train.json")
    checkpoint = read_json(RECORDED_CHECKPOINT)
    if checkpoint_record is not None:
        other = read_json(checkpoint_record)
        for key in (
            "initialization",
            "base_sha256",
            "dataset",
            "split_sha256",
            "seed",
            "train_statistics_verified",
        ):
            if other[key] != checkpoint[key]:
                raise ValueError(f"Adapted checkpoint uses a different training recipe: {key}")
        checkpoint = other
    if checkpoint["step"] not in training["selection_steps"]:
        raise ValueError("Checkpoint step is outside the declared development candidates")
    development_states = set(official["initial_state_ids"]) | set(training["development_state_ids"])
    if set(plan["initial_state_ids"]) & development_states:
        raise ValueError("Final and development states overlap")
    return plan, official, checkpoint


def reference_record(directory):
    record = directory / "adapted-checkpoint.json"
    return record if record.is_file() else RECORDED_CHECKPOINT


def reference_results(directory, plan, official, checkpoint):
    hashes = read_json(directory / "sha256.json")
    for name in ("comparison.json", "episodes.csv", "summary.json"):
        if file_sha256(directory / name) != hashes[name]:
            raise ValueError(f"Corrupt reference file: {name}")
    comparison = read_json(directory / "comparison.json")
    if comparison != {
        "plan": plan,
        "released_sha256": official["weight_sha256"]["policy"],
        "adapted_sha256": checkpoint["checkpoint_sha256"],
    }:
        raise ValueError("The reference was collected with a different plan or weights")
    with (directory / "episodes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    summary = read_json(directory / "summary.json")
    roles = set(summary["policies"])
    if roles not in (set(ROLES), {*ROLES, "base"}):
        raise ValueError("Unexpected policies in the reference")
    indexed = {(r["policy"], int(r["task_id"]), int(r["initial_state_id"])): r for r in rows}
    expected = {
        (role, t, s) for role in roles for t in plan["task_ids"] for s in plan["initial_state_ids"]
    }
    if set(indexed) != expected or len(rows) != len(expected):
        raise ValueError("Incomplete or duplicate reference episodes")
    for role in roles:
        flags = [r["success"] for r in rows if r["policy"] == role]
        if (
            set(flags) - {"True", "False"}
            or flags.count("True") != summary["policies"][role]["successes"]
        ):
            raise ValueError(f"Invalid {role} outcomes in the reference")
    return indexed, summary["runtime"]


def check_shard(folder, task, role, plan, official, checkpoint):
    protocol, summary = check_run(folder)
    expected = {k: v for k, v in plan.items() if k != "lift_threshold_m"}
    expected.update({k: official[k] for k in ("backbone", "assets", "demonstrations")})
    expected["task_ids"] = [task] if isinstance(task, int) else list(task)
    digest = official["weight_sha256"]["policy"]
    if role == "adapted":
        digest = checkpoint["checkpoint_sha256"]
        expected["policy"] = {
            "source": "project_checkpoint",
            **{
                k: checkpoint[k]
                for k in (
                    "initialization",
                    "base_sha256",
                    "dataset",
                    "split_sha256",
                    "step",
                    "seed",
                )
            },
        }
    else:
        expected["policy"] = official["policy"]
    expected["weight_sha256"] = {**official["weight_sha256"], "policy": digest}
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"{folder}: unexpected {key} in the evaluation manifest")
    return protocol, summary["episodes"]
