"""Evaluate both frozen checkpoints on every final task/state pair, then build a report."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from final_protocol import (
    ROLES,
    ROOT,
    check_shard,
    file_sha256,
    load_protocol,
    read_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New or resumable run directory")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Adapted pretrained_model directory"
    )
    parser.add_argument(
        "--checkpoint-record",
        type=Path,
        help="check_checkpoint.py JSON for an independent rerun of the same training recipe",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    checkpoint_dir = args.checkpoint.resolve()
    plan, official, checkpoint = load_protocol(args.checkpoint_record)
    released_dir = ROOT / ".cache" / "inputs" / "policy" / official["policy"]["revision"]
    for folder, digest in (
        (released_dir, official["weight_sha256"]["policy"]),
        (checkpoint_dir, checkpoint["checkpoint_sha256"]),
    ):
        if file_sha256(folder / "model.safetensors") != digest:
            raise ValueError(f"Incorrect weights in {folder.name}")
    from safetensors.numpy import load_file

    scales = read_json(ROOT / "configs/final_action_scale.json")
    for role, folder in zip(ROLES, (released_dir, checkpoint_dir), strict=True):
        steps = read_json(folder / "policy_postprocessor.json")["steps"]
        step = next(item for item in steps if item["registry_name"] == "unnormalizer_processor")
        stored = load_file(folder / step["state_file"])
        if any(stored[f"action.{key}"].tolist() != scales[role][key] for key in ("mean", "std")):
            raise ValueError(f"{role}: action statistics differ from this recipe")
    output.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MUJOCO_GL": "egl",
    }
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = {
            "plan": plan,
            "released_sha256": official["weight_sha256"]["policy"],
            "adapted_sha256": checkpoint["checkpoint_sha256"],
        }
        frozen = output / "comparison.json"
        if frozen.exists() and json.loads(frozen.read_text()) != spec:
            raise ValueError("Existing output belongs to a different comparison")
        if not frozen.exists():
            frozen.write_text(json.dumps(spec, indent=2) + "\n")
        if args.checkpoint_record is not None:
            (output / "adapted-checkpoint.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
        for index, task in enumerate(plan["task_ids"]):
            for role in ROLES if index % 2 == 0 else reversed(ROLES):
                parent = output / role
                parent.mkdir(exist_ok=True)
                shard = parent / f"task-{task:02d}"
                if shard.exists():
                    check_shard(shard, task, role, plan, official, checkpoint)
                    print(f"Verified completed {role} task {task:02d}", flush=True)
                    continue
                with TemporaryDirectory(dir=parent, prefix=f".task-{task:02d}-") as tmp:
                    work = Path(tmp) / "run"
                    cmd = [
                        sys.executable,
                        "-u",
                        str(ROOT / "scripts/evaluate.py"),
                        "--output",
                        str(work),
                        "--record-scene",
                        "--task-ids",
                        str(task),
                        "--state-ids",
                        *map(str, plan["initial_state_ids"]),
                    ]
                    if role == "adapted":
                        cmd.extend(["--checkpoint", str(checkpoint_dir)])
                    print(f"Starting {role} task {task:02d}", flush=True)
                    subprocess.run(cmd, env=env, check=True)
                    check_shard(work, task, role, plan, official, checkpoint)
                    work.rename(shard)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/report_final.py"),
                str(output),
                "--output",
                str(output / "review"),
            ],
            check=True,
        )
        print("Verified paired results and review package are ready.", flush=True)


if __name__ == "__main__":
    main()
