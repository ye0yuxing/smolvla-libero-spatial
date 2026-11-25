"""Select complete Spatial demonstrations and fit train-only normalizers."""

import hashlib
import json
import random
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from prepare import CONFIG, ROOT, fetch_snapshot, input_dir

RECIPE = ROOT / "configs" / "spatial_train.json"
SPLIT = ROOT / ".cache" / "training" / "spatial_split.json"
CAMERAS = ("observation.images.image", "observation.images.image2")


def spatial_split(root, settings, shared):
    info = json.loads((root / "meta/info.json").read_text())
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pandas()
    names = settings["instructions"]
    task_ids = [int(tasks.loc[name, "task_index"]) for name in names]
    if len(set(task_ids)) != 10:
        raise ValueError("Expected ten different Spatial instructions")
    target = set(task_ids)
    episode_tasks = {}
    # The v3 episode table only contains ranges. Task labels live in the frame shards.
    for path in sorted((root / "data").glob("*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "task_index"])
        for ep, task in set(zip(*(c.to_numpy() for c in table.columns), strict=True)):
            if episode_tasks.setdefault(int(ep), int(task)) != task:
                raise ValueError(f"Episode {ep} contains more than one task")
    if len(episode_tasks) != info["total_episodes"]:
        raise ValueError("Incomplete demonstration shards")
    metadata = {}
    for path in sorted((root / "meta/episodes").glob("*/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            metadata[row["episode_index"]] = row
    selected = {e for e, t in episode_tasks.items() if t in target}
    groups = []
    for suite_id, (name, task_id) in enumerate(zip(names, task_ids, strict=True)):
        episodes = sorted(e for e in selected if episode_tasks[e] == task_id)
        if len(episodes) < 2:
            raise ValueError(f"Too few episodes for {name}")
        count = max(1, round(len(episodes) * settings["validation_fraction"]))
        holdout = set(random.Random(settings["split_seed"] + suite_id).sample(episodes, count))
        group = {"suite_id": suite_id, "task_index": task_id, "instruction": name}
        for part, ids in (
            ("train", sorted(set(episodes) - holdout)),
            ("validation", sorted(holdout)),
        ):
            group[part] = ids
            group[f"{part}_frames"] = sum(metadata[e]["length"] for e in ids)
        groups.append(group)
    train_ids = {e for g in groups for e in g["train"]}
    files = set()
    arrays = {"observation.state": [], "action": []}
    for ep in selected:
        row = metadata[ep]
        for camera in CAMERAS:
            prefix = f"videos/{camera}"
            files.add(
                info["video_path"].format(
                    video_key=camera,
                    chunk_index=row[f"{prefix}/chunk_index"],
                    file_index=row[f"{prefix}/file_index"],
                )
            )
    for path in sorted((root / "data").glob("*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", *arrays])
        mask = np.isin(table["episode_index"].to_numpy(), list(train_ids))
        if mask.any():
            kept = table.filter(mask)
            for key, parts in arrays.items():
                parts.append(np.asarray(kept[key].to_pylist(), dtype=np.float64))
    stats = {}
    for key, width in (("observation.state", 8), ("action", 7)):
        values = np.concatenate(arrays[key])
        if (
            values.shape != (sum(g["train_frames"] for g in groups), width)
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Invalid training frames for {key}")
        stats[key] = {"mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist()}
        if min(stats[key]["std"]) <= 0:
            raise ValueError(f"Constant training feature in {key}")
    return {
        "dataset": shared["demonstrations"],
        "fps": info["fps"],
        "split_seed": settings["split_seed"],
        "tasks": groups,
        "stats": stats,
    }, sorted(files)


def main():
    shared = json.loads(CONFIG.read_text())
    settings = json.loads(RECIPE.read_text())
    root = input_dir("demonstrations", shared)
    ds = dict(**shared["demonstrations"], repo_type="dataset", local_dir=root, max_workers=6)
    snapshot_download(**ds, allow_patterns=["meta/*", "data/*"])
    split, videos = spatial_split(root, settings, shared)
    print(
        f"Selected {sum(len(t['train']) for t in split['tasks'])} training episodes, "
        f"{sum(len(t['validation']) for t in split['tasks'])} validation episodes and {len(videos)} video shards.",
        flush=True,
    )
    snapshot_download(**ds, allow_patterns=videos)
    base = input_dir("base", settings)
    snapshot_download(
        **settings["base"], local_dir=base, allow_patterns=["config.json", "model.safetensors"]
    )
    with (base / "model.safetensors").open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != settings["base_sha256"]:
            raise ValueError("Incorrect base policy weights")
    fetch_snapshot("backbone", shared)
    SPLIT.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=SPLIT.parent, delete=False) as out:
        json.dump(split, out, indent=2, allow_nan=False)
        out.write("\n")
    Path(out.name).replace(SPLIT)
    print("The Spatial demonstrations, base weights and training statistics are ready.")


if __name__ == "__main__":
    main()
