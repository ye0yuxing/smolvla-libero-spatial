"""Measure SmolVLA before Spatial fine-tuning on the recorded final starts."""

import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from check_run import check_run
from final_protocol import (
    ROLES,
    ROOT,
    expected_interface,
    load_protocol,
    read_json,
    reference_record,
    reference_results,
)


def check_base_task(folder, task, plan, official, checkpoint, reference, runtime):
    protocol, summary = check_run(folder)
    expected = {k: v for k, v in plan.items() if k != "lift_threshold_m"}
    expected.update({k: official[k] for k in ("backbone", "assets", "demonstrations")})
    expected.update(runtime)
    expected["task_ids"] = [task]
    expected["policy"] = {
        "source": "pretrained_before_spatial_finetuning",
        **checkpoint["initialization"],
        "spatial_updates": 0,
        "base_sha256": checkpoint["base_sha256"],
        "interface_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "normalization_split_sha256": checkpoint["split_sha256"],
        "interface_sha256": expected_interface(checkpoint),
    }
    expected["weight_sha256"] = {**official["weight_sha256"], "policy": checkpoint["base_sha256"]}
    for name, value in expected.items():
        if protocol.get(name) != value:
            raise ValueError(f"{folder}: unexpected {name} in the base protocol")
    instruction = read_json(ROOT / "configs/spatial_train.json")["instructions"][task]
    for ep in summary["episodes"]:
        if ep["instruction"] != instruction:
            raise ValueError(f"{folder}: task instruction mismatch")
        for role in ROLES:
            old = reference[role, task, ep["initial_state_id"]]
            if (int(old["seed"]), old["initial_sim_sha256"]) != (
                ep["seed"],
                ep["initial_sim_sha256"],
            ):
                raise ValueError(
                    f"{folder}: initial scene differs from {role} on state {ep['initial_state_id']}"
                )
    return summary["episodes"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Adapted checkpoint providing the LIBERO interface",
    )
    parser.add_argument("--reference", type=Path, default=ROOT / "results/evaluation")
    parser.add_argument("--output", type=Path, required=True, help="New or resumable run directory")
    args = parser.parse_args()
    record = reference_record(args.reference).resolve()
    plan, official, checkpoint = load_protocol(record)
    reference, runtime = reference_results(args.reference, plan, official, checkpoint)
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MUJOCO_GL": "egl",
    }
    with (run / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for task in plan["task_ids"]:
            shard = run / f"task-{task:02d}"
            if shard.exists():
                check_base_task(shard, task, plan, official, checkpoint, reference, runtime)
                print(f"Verified completed base task {task:02d}", flush=True)
                continue
            with TemporaryDirectory(dir=run, prefix=f".task-{task:02d}-") as tmp:
                work = Path(tmp) / "run"
                cmd = [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts/evaluate.py"),
                    "--output",
                    str(work),
                    "--checkpoint",
                    str(args.checkpoint.resolve()),
                    "--step-zero",
                    "--checkpoint-record",
                    str(record),
                    "--record-scene",
                    "--task-ids",
                    str(task),
                    "--state-ids",
                    *map(str, plan["initial_state_ids"]),
                ]
                print(f"Starting step-zero task {task:02d}", flush=True)
                subprocess.run(cmd, env=env, check=True)
                check_base_task(work, task, plan, official, checkpoint, reference, runtime)
                work.rename(shard)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/report_base.py"),
                str(run),
                "--reference",
                str(args.reference.resolve()),
                "--output",
                str(run / "review"),
            ],
            check=True,
        )
        print("Verified step-zero comparison ready.", flush=True)


if __name__ == "__main__":
    main()
