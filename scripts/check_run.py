"""Check the saved transitions and print per-task results without loading a model."""

import argparse
import json
import math
from pathlib import Path
from statistics import median


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_episode(folder, ep, protocol):
    steps = read_jsonl(folder / "steps.jsonl")
    chunks = read_jsonl(folder / "chunks.jsonl")
    horizon = protocol["n_action_steps"]
    if not steps or len(steps) != ep["steps"] or len(steps) > protocol["max_steps"]:
        raise ValueError(f"{folder.name}: invalid step count")
    expected_seed = protocol["base_seed"] + 1000 * ep["task_id"] + ep["initial_state_id"]
    if ep["seed"] != expected_seed:
        raise ValueError(f"{folder.name}: seed disagrees with the protocol")
    if ep["reached_step_limit"] != (len(steps) == protocol["max_steps"]):
        raise ValueError(f"{folder.name}: incorrect step-limit flag")
    expected = list(range(0, len(steps), horizon))
    if [c["step"] for c in chunks] != expected or ep["chunk_calls"] != len(chunks):
        raise ValueError(f"{folder.name}: missing or misplaced policy decision")
    timings = ep["inference_ms"]
    if len(timings) != len(chunks) or any(not math.isfinite(t) or t < 0 for t in timings):
        raise ValueError(f"{folder.name}: invalid inference timings")
    for chunk in chunks:
        actions = chunk["normalized_actions"]
        if len(actions) != protocol["chunk_size"] or any(
            len(a) != 7 or not all(map(math.isfinite, a)) for a in actions
        ):
            raise ValueError(f"{folder.name}: invalid predicted chunk")
    for t, row in enumerate(steps):
        offset = t % horizon
        if (row["step"], row["chunk_start"], row["chunk_offset"]) != (t, t - offset, offset):
            raise ValueError(f"{folder.name}: transition {t} has an invalid action index")
        for key, width in (("state", 8), ("next_state", 8), ("action", 7)):
            if len(row[key]) != width or not all(map(math.isfinite, row[key])):
                raise ValueError(f"{folder.name}: invalid {key} in transition {t}")
        if t and steps[t - 1]["next_state"] != row["state"]:
            raise ValueError(f"{folder.name}: discontinuity at transition {t}")
        if t + 1 < len(steps) and (row["success"] or row["terminated"] or row["truncated"]):
            raise ValueError(f"{folder.name}: recorded a step after completion")
    last = steps[-1]
    if any(last[k] != ep[k] for k in ("success", "terminated", "truncated")):
        raise ValueError(f"{folder.name}: final step disagrees with episode result")
    if not (
        last["success"]
        or last["terminated"]
        or last["truncated"]
        or len(steps) == protocol["max_steps"]
    ):
        raise ValueError(f"{folder.name}: episode ends early")
    if not (folder / "cameras.mp4").stat().st_size:
        raise ValueError(f"{folder.name}: empty video")


def check_run(root):
    protocol = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    expected = {(t, s) for t in protocol["task_ids"] for s in protocol["initial_state_ids"]}
    observed = set()
    per_task = {t: 0 for t in protocol["task_ids"]}
    for ep in summary["episodes"]:
        task, state = ep["task_id"], ep["initial_state_id"]
        if (task, state) not in expected or (task, state) in observed:
            raise ValueError(f"Unexpected or repeated task/state pair {(task, state)}")
        observed.add((task, state))
        folder = root / f"task-{task:02d}_state-{state:02d}"
        if json.loads((folder / "episode.json").read_text()) != ep:
            raise ValueError(f"{folder.name}: episode metadata disagrees with summary")
        check_episode(folder, ep, protocol)
        per_task[task] += int(ep["success"])
    successes = sum(per_task.values())
    if observed != expected or (summary["successes"], summary["n_episodes"]) != (
        successes,
        len(expected),
    ):
        raise ValueError("Coverage or success totals disagree with the protocol")
    return protocol, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    protocol, summary = check_run(parser.parse_args().run)
    per_task = {task: 0 for task in protocol["task_ids"]}
    for ep in summary["episodes"]:
        per_task[ep["task_id"]] += int(ep["success"])
    for task, n in per_task.items():
        print(f"Task {task:02d}  {n}/{len(protocol['initial_state_ids'])}")
    successes, total = summary["successes"], summary["n_episodes"]
    print(f"All tasks  {successes}/{total} ({successes / total:.1%})")
    num_actions = sum(ep["steps"] for ep in summary["episodes"])
    times = [t for ep in summary["episodes"] for t in ep["inference_ms"]]
    print(f"{num_actions:,} actions, {len(times)} chunk predictions, {median(times):.2f} ms median")


if __name__ == "__main__":
    main()
