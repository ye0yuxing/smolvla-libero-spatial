"""Verify control, video and scene evidence for one LIBERO-Spatial rollout."""

import math
import re
import subprocess

from check_run import read_jsonl

OUTCOMES = ("both", "released_only", "adapted_only", "neither")
FAILURES = ("no_target_contact", "contact_without_lift", "lift_without_placement")
FAILURE_LABELS = {
    "no_target_contact": "No sampled two-pad contact with target",
    "contact_without_lift": "Target contact, lift threshold unreached during contact",
    "lift_without_placement": "Target lifted during contact, placement unsuccessful",
}


def check_video(path, ffmpeg, expected):
    run = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-threads",
            "1",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-progress",
            "pipe:1",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    counts = re.findall(r"^frame=\s*(\d+)\s*$", run.stdout, re.MULTILINE)
    if not counts or int(counts[-1]) != expected:
        raise ValueError(f"{path}: decoded video frame count differs from trace")


def check_actions(folder, scale):
    chunks = {c["step"]: c["normalized_actions"] for c in read_jsonl(folder / "chunks.jsonl")}
    error = 0.0
    for row in read_jsonl(folder / "steps.jsonl"):
        normalized = chunks[row["chunk_start"]][row["chunk_offset"]]
        for action, n, mean, std in zip(
            row["action"], normalized, scale["mean"], scale["std"], strict=True
        ):
            error = max(error, abs(action - (n * std + mean)))
    if error > 3e-6:
        raise ValueError(f"{folder}: action postprocessing error {error}")
    return error


def diagnose(folder, ep, threshold):
    samples = read_jsonl(folder / "scene.jsonl")
    if len(samples) != ep["steps"] + 1:
        raise ValueError(f"{folder}: scene trace and control steps disagree")
    for index, row in enumerate(samples):
        if row["frame"] != index or not all(
            type(row[key]) is bool for key in ("target_between_pads", "other_bowl_between_pads")
        ):
            raise ValueError(f"{folder}: invalid scene sample {index}")
        for key in ("target_xyz", "plate_xyz"):
            if len(row[key]) != 3 or not all(map(math.isfinite, row[key])):
                raise ValueError(f"{folder}: invalid position at scene sample {index}")
    initial_z = samples[0]["target_xyz"][2]
    contacts = [row for row in samples if row["target_between_pads"]]
    lifts = [row for row in contacts if row["target_xyz"][2] - initial_z >= threshold]
    final = samples[-1]
    if ep["success"]:
        outcome = "success"
    elif not contacts:
        outcome = FAILURES[0]
    elif not lifts:
        outcome = FAILURES[1]
    else:
        outcome = FAILURES[2]
    return {
        "outcome": outcome,
        "first_target_contact_frame": contacts[0]["frame"] if contacts else None,
        "first_lift_frame": lifts[0]["frame"] if lifts else None,
        "max_target_rise_in_contact_m": round(
            max(row["target_xyz"][2] - initial_z for row in contacts), 5
        )
        if contacts
        else None,
        "other_bowl_contact": any(row["other_bowl_between_pads"] for row in samples),
        "final_target_plate_xy_m": round(
            math.dist(final["target_xyz"][:2], final["plate_xyz"][:2]), 5
        ),
    }


def paired_outcome(released, adapted):
    return {
        (True, True): OUTCOMES[0],
        (True, False): OUTCOMES[1],
        (False, True): OUTCOMES[2],
        (False, False): OUTCOMES[3],
    }[released, adapted]
