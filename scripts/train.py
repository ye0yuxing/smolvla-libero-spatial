"""Fine-tune the SmolVLA action expert on a single GPU."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import statistics
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.optim.optimizers import load_optimizer_state, save_optimizer_state
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import RenameObservationsProcessorStep
from lerobot.utils.random_utils import load_rng_state, save_rng_state
from prepare import CONFIG, input_dir
from prepare_training import CAMERAS, RECIPE, SPLIT
from torch.utils.data import DataLoader


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def make_model(settings, shared, split, weights):
    cfg = SmolVLAConfig.from_pretrained(input_dir("base", settings))
    cfg.vlm_model_name = str(input_dir("backbone", shared))
    cfg.pretrained_path = settings["base"]["repo_id"]
    cfg.push_to_hub = False
    cfg.train_expert_only = cfg.freeze_vision_encoder = cfg.train_state_proj = True
    cfg.device = "cuda"
    rename = dict(
        zip(CAMERAS, ("observation.images.camera1", "observation.images.camera2"), strict=True)
    )
    cfg.input_features = {"observation.state": PolicyFeature(FeatureType.STATE, (8,))}
    cfg.input_features.update(
        {dest: PolicyFeature(FeatureType.VISUAL, (3, 256, 256)) for dest in rename.values()}
    )
    cfg.output_features = {"action": PolicyFeature(FeatureType.ACTION, (7,))}
    policy = SmolVLAPolicy.from_pretrained(weights, config=cfg, strict=True)
    pre, post = make_smolvla_pre_post_processors(cfg, dataset_stats=split["stats"])
    rename_step = next(s for s in pre.steps if isinstance(s, RenameObservationsProcessorStep))
    rename_step.rename_map = rename
    return policy, pre, post


def open_data(split, shared, part, horizon):
    return LeRobotDataset(
        **split["dataset"],
        root=input_dir("demonstrations", shared),
        episodes=sorted(ep for t in split["tasks"] for ep in t[part]),
        delta_timestamps={"action": [i / split["fps"] for i in range(horizon)]},
        video_backend="pyav",
        return_uint8=True,
    )


def training_indices(seed, start, end, size, batch):
    # The next batch depends only on the optimizer step, even with worker prefetching.
    for step in range(start + 1, end + 1):
        generator = torch.Generator().manual_seed(seed + step)
        yield torch.randint(size, (batch,), generator=generator).tolist()


def validation_batches(data, split, cfg):
    tasks = np.asarray(data.hf_dataset.with_format("numpy")["task_index"])
    rng = random.Random(cfg["split_seed"])
    indices = []
    for group in split["tasks"]:
        ids = np.flatnonzero(tasks == group["task_index"]).tolist()
        indices.extend(rng.sample(ids, cfg["validation_frames_per_task"]))
    return list(
        DataLoader(
            data,
            batch_size=cfg["batch_size"],
            sampler=indices,
            num_workers=cfg["workers"],
            generator=torch.Generator().manual_seed(0),
        )
    )


@torch.inference_mode()
def validation_loss(policy, pre, batches, seed):
    policy.eval()
    total, valid = 0.0, 0
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        for batch in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy(pre(batch))
            count = int((~batch["action_is_pad"]).sum())
            total += loss.item() * count
            valid += count
    policy.train()
    return total / valid


def learning_rate(step, cfg):
    warmup = cfg["warmup_steps"]
    if step <= warmup:
        return cfg["learning_rate"] * step / warmup
    phase = (step - warmup) / (cfg["steps"] - warmup)
    floor = cfg["final_lr_fraction"]
    return cfg["learning_rate"] * (floor + (1 - floor) * (1 + math.cos(math.pi * phase)) / 2)


def save_checkpoint(output, history, policy, pre, post, optimizer, shared):
    parent = output / "checkpoints"
    parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=parent) as tmp:
        stage = Path(tmp) / "checkpoint"
        model = stage / "pretrained_model"
        model.mkdir(parents=True)
        policy.save_pretrained(model)
        pre.save_pretrained(model)
        post.save_pretrained(model)
        # Store public Hub IDs in portable exports. Offline loading uses the pinned local cache.
        model_config = json.loads((model / "config.json").read_text())
        model_config["vlm_model_name"] = shared["backbone"]["repo_id"]
        write_json(model / "config.json", model_config)
        processor = json.loads((model / "policy_preprocessor.json").read_text())
        next(s for s in processor["steps"] if s["registry_name"] == "tokenizer_processor")[
            "config"
        ]["tokenizer_name"] = shared["backbone"]["repo_id"]
        write_json(model / "policy_preprocessor.json", processor)
        write_json(
            model / "training_provenance.json",
            {
                "initialization": history["recipe"]["base"],
                "base_sha256": history["recipe"]["base_sha256"],
                "dataset": shared["demonstrations"],
                "split_sha256": history["split_sha256"],
                "step": history["step"],
                "seed": history["recipe"]["seed"],
            },
        )
        save_optimizer_state(optimizer, stage)
        save_rng_state(stage)
        write_json(stage / "history.json", history)
        stage.rename(parent / f"{history['step']:06d}")


def train(output, resume, stop):
    cfg, shared, split = (json.loads(p.read_text()) for p in (RECIPE, CONFIG, SPLIT))
    if split["dataset"] != shared["demonstrations"]:
        raise ValueError("Dataset revision mismatch. Run scripts/prepare_training.py.")
    end = cfg["steps"] if stop is None else stop
    if not 0 < end <= cfg["steps"] or not torch.cuda.is_bf16_supported():
        raise ValueError("Use a valid stop step and a BF16-capable CUDA GPU")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)
    split_hash = hashlib.sha256(SPLIT.read_bytes()).hexdigest()
    history = {
        "recipe": cfg,
        "split_sha256": split_hash,
        "step": 0,
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "versions": {k: version(k) for k in ("lerobot", "torch", "transformers", "av")},
        "training_seconds": 0.0,
        "data_wait_seconds": 0.0,
        "peak_allocated_gib": 0.0,
        "peak_reserved_gib": 0.0,
        "training": [],
        "validation": [],
    }
    checkpoint = None
    if resume:
        checkpoints = sorted(p for p in (output / "checkpoints").glob("*") if p.name.isdecimal())
        if not checkpoints:
            raise FileNotFoundError("No completed checkpoint to resume")
        checkpoint = checkpoints[-1]
        history = json.loads((checkpoint / "history.json").read_text())
        if history["recipe"] != cfg or history["split_sha256"] != split_hash:
            raise ValueError("The recipe or split has changed since checkpoint creation")
    start = history["step"]
    if start >= end:
        raise ValueError(f"Already at step {start}, requested {end}")
    weights = checkpoint / "pretrained_model" if resume else input_dir("base", cfg)
    policy, pre, post = make_model(cfg, shared, split, weights)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    history["trainable_parameters"] = sum(p.numel() for p in parameters)
    history["total_parameters"] = sum(p.numel() for p in policy.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=cfg["learning_rate"], betas=(0.9, 0.95), weight_decay=1e-10
    )
    train_data = open_data(split, shared, "train", policy.config.chunk_size)
    val_data = open_data(split, shared, "validation", policy.config.chunk_size)
    for part, data in (("train", train_data), ("validation", val_data)):
        if len(data) != sum(t[part + "_frames"] for t in split["tasks"]):
            raise ValueError(f"Dataset frame count disagrees with the {part} split")
    batches = validation_batches(val_data, split, cfg)
    indices = training_indices(cfg["seed"], start, end, len(train_data), cfg["batch_size"])
    loader = iter(
        DataLoader(
            train_data,
            batch_sampler=indices,
            num_workers=cfg["workers"],
            generator=torch.Generator().manual_seed(cfg["seed"]),
        )
    )
    if resume:
        load_optimizer_state(optimizer, checkpoint)
        load_rng_state(checkpoint)
    else:
        score = validation_loss(policy, pre, batches, cfg["seed"])
        history["validation"].append({"step": 0, "loss": score})
        print(f"Initial validation loss {score:.6f}", flush=True)
    policy.train()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for step in range(start + 1, end + 1):
        begin = perf_counter()
        sample = next(loader)
        history["data_wait_seconds"] += perf_counter() - begin
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(pre(sample))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        history["training_seconds"] += perf_counter() - begin
        losses.append(loss.item())
        history["step"] = step
        if step % cfg["log_every"] == 0 or step == end:
            point = {
                "step": step,
                "updates": len(losses),
                "mean_loss": statistics.mean(losses),
                "lr": lr,
            }
            history["training"].append(point)
            print(json.dumps(point), flush=True)
            losses.clear()
        if step % cfg["validate_every"] == 0 or step == end:
            score = validation_loss(policy, pre, batches, cfg["seed"])
            history["validation"].append({"step": step, "loss": score})
            print(f"Step {step} validation loss {score:.6f}", flush=True)
        if step % cfg["checkpoint_every"] == 0 or step == end:
            for key, value in (
                ("peak_allocated_gib", torch.cuda.max_memory_allocated()),
                ("peak_reserved_gib", torch.cuda.max_memory_reserved()),
            ):
                history[key] = max(history[key], value / 2**30)
            save_checkpoint(output, history, policy, pre, post, optimizer, shared)
            print(
                f"Saved step {step}, peak allocated {history['peak_allocated_gib']:.2f} GiB",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true", help="Resume the latest complete checkpoint"
    )
    parser.add_argument("--stop-at", type=int, help="Save and stop at this absolute step")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with (args.output / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(args.output, args.resume, args.stop_at)


if __name__ == "__main__":
    main()
