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
from training_precision import check_adam_precision, promote_trainable, save_policy


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


def save_checkpoint(output, history, policy, pre, post, optimizer, shared, promoted):
    parent = output / "checkpoints"
    parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=parent) as tmp:
        stage = Path(tmp) / "checkpoint"
        model = stage / "pretrained_model"
        model.mkdir(parents=True)
        save_policy(policy, model, promoted)
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


def backward_loss(policy, pre, sample, micro_batch_size):
    padding = sample["action_is_pad"]
    if len(padding) == micro_batch_size:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(pre(sample))
        loss.backward()
        return loss.detach().item()
    # LeRobot averages over valid action timesteps, which can differ between microbatches.
    valid = int((~padding).sum())
    total = 0.0
    for offset in range(0, len(padding), micro_batch_size):
        section = slice(offset, offset + micro_batch_size)
        batch = {key: value[section] for key, value in sample.items()}
        fraction = int((~padding[section]).sum()) / valid
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(pre(batch))
        (loss * fraction).backward()
        total += loss.detach().item() * fraction
    return total


def train(output, resume, stop, long_budget, micro_batch_size, profile):
    cfg, shared, split = (json.loads(p.read_text()) for p in (RECIPE, CONFIG, SPLIT))
    if long_budget:
        cfg.update(json.loads((RECIPE.parent / "spatial_long.json").read_text()))
    micro_batch_size = cfg["batch_size"] if micro_batch_size is None else micro_batch_size
    if micro_batch_size < 1 or cfg["batch_size"] % micro_batch_size:
        raise ValueError("The microbatch size must be a positive divisor of the effective batch")
    if split["dataset"] != shared["demonstrations"]:
        raise ValueError("Dataset revision mismatch. Run scripts/prepare_training.py.")
    end = cfg["steps"] if stop is None else stop
    if profile:
        end = 50
    if not 0 < end <= cfg["steps"] or not torch.cuda.is_bf16_supported():
        raise ValueError("Use a valid stop step and a BF16-capable CUDA GPU")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)
    split_hash = hashlib.sha256(SPLIT.read_bytes()).hexdigest()
    if long_budget:
        reference = json.loads((CONFIG.parents[1] / "results/training/checkpoint.json").read_text())
        base_file = input_dir("base", cfg) / "model.safetensors"
        with base_file.open("rb") as f:
            base_hash = hashlib.file_digest(f, "sha256").hexdigest()
        if (split_hash, base_hash) != (reference["split_sha256"], cfg["base_sha256"]):
            raise ValueError("The long-budget run requires the original split and base weights")
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
        if history.get("micro_batch_size", cfg["batch_size"]) != micro_batch_size:
            raise ValueError("The microbatch size changed on resume")
    start = history["step"]
    if start >= end:
        raise ValueError(f"Already at step {start}, requested {end}")
    weights = checkpoint / "pretrained_model" if resume else input_dir("base", cfg)
    policy, pre, post = make_model(cfg, shared, split, weights)
    promoted = promote_trainable(policy, checkpoint) if long_budget else {}
    parameters = [p for p in policy.parameters() if p.requires_grad]
    if long_budget:
        if any(p.dtype != torch.float32 for p in parameters):
            raise ValueError("Every trainable weight must be FP32")
        history["micro_batch_size"] = micro_batch_size
        history["promoted_parameters"] = sum(p.numel() for p in promoted.values())
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
    batches = (
        []
        if profile
        else validation_batches(val_data, split, {**cfg, "batch_size": micro_batch_size})
    )
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
        if long_budget:
            check_adam_precision(optimizer)
        load_rng_state(checkpoint)
    elif not profile:
        score = validation_loss(policy, pre, batches, cfg["seed"])
        history["validation"].append({"step": 0, "loss": score})
        print(f"Initial validation loss {score:.6f}", flush=True)
    policy.train()
    torch.cuda.reset_peak_memory_stats()
    losses = []
    interval_seconds = 0.0
    measured_seconds = 0.0
    for step in range(start + 1, end + 1):
        begin = perf_counter()
        sample = next(loader)
        history["data_wait_seconds"] += perf_counter() - begin
        lr = learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = backward_loss(policy, pre, sample, micro_batch_size)
        torch.nn.utils.clip_grad_norm_(parameters, 10.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = perf_counter() - begin
        interval_seconds += elapsed
        if profile and step > 10:
            measured_seconds += elapsed
        history["training_seconds"] += elapsed
        if long_budget and step == start + 1:
            check_adam_precision(optimizer)
        losses.append(loss)
        history["step"] = step
        if step % (10 if profile else cfg["log_every"]) == 0 or step == end:
            point = {
                "step": step,
                "updates": len(losses),
                "mean_loss": statistics.mean(losses),
                "lr": lr,
            }
            if long_budget:
                point["windows_per_second"] = len(losses) * cfg["batch_size"] / interval_seconds
                point["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                status = output / "progress.tmp"
                write_json(status, point)
                status.replace(output / "progress.json")
            history["training"].append(point)
            print(json.dumps(point), flush=True)
            losses.clear()
            interval_seconds = 0.0
        if profile:
            continue
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
            save_checkpoint(output, history, policy, pre, post, optimizer, shared, promoted)
            print(
                f"Saved step {step}, peak allocated {history['peak_allocated_gib']:.2f} GiB",
                flush=True,
            )
    if profile:
        write_json(
            output / "profile.json",
            {
                "effective_batch": cfg["batch_size"],
                "micro_batch_size": micro_batch_size,
                "warmup_updates": 10,
                "measured_updates": end - 10,
                "windows_per_second": (end - 10) * cfg["batch_size"] / measured_seconds,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "trainable_parameters": history["trainable_parameters"],
                "fp32_promoted_parameters": history["promoted_parameters"],
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true", help="Resume the latest complete checkpoint"
    )
    parser.add_argument("--stop-at", type=int, help="Save and stop at this absolute step")
    parser.add_argument("--long-budget", action="store_true", help="Use configs/spatial_long.json")
    parser.add_argument(
        "--micro-batch-size", type=int, help="Accumulate to the recipe's batch size"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Time 40 long-budget updates after 10 warm-up updates",
    )
    args = parser.parse_args()
    if args.profile and (not args.long_budget or args.resume or args.stop_at is not None):
        parser.error("--profile needs a fresh --long-budget run")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with (args.output / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(
            args.output,
            args.resume,
            args.stop_at,
            args.long_budget,
            args.micro_batch_size,
            args.profile,
        )


if __name__ == "__main__":
    main()
