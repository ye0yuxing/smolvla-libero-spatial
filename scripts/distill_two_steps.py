"""Distill two Euler updates from the midpoint and endpoint of a frozen ten-step teacher."""

import argparse
import fcntl
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from evaluate import load_policy
from final_protocol import (
    ROOT,
    expected_interface,
    file_sha256,
    interface_hashes,
    load_protocol,
    processor_artifact_hashes,
    read_json,
)
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from prepare_training import RECIPE, SPLIT
from run_experiment import freeze, write_json
from safetensors import safe_open
from torch.utils.data import DataLoader
from train import learning_rate, open_data, training_indices, validation_batches
from training_precision import check_adam_precision, promote_trainable, save_policy

EXPERT = (
    "model.vlm_with_expert.lm_expert.",
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.action_time_mlp_in.",
    "model.action_time_mlp_out.",
)


@torch.no_grad()
def observation_cache(policy, batch):
    model = policy.model
    images, masks = policy.prepare_images(batch)
    embeds, valid, attention = model.embed_prefix(
        images,
        masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        state=policy.prepare_state(batch),
    )
    _, cache = model.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(valid, attention),
        position_ids=valid.cumsum(dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[embeds, None],
        use_cache=True,
    )
    return valid, cache


def integrate_nodes(model, condition, noise, steps):
    if steps < 2 or steps % 2:
        raise ValueError("Midpoint supervision requires an even number of Euler steps")
    valid, cache = condition
    x = noise
    dt = -1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), 1.0 + i * dt, dtype=torch.float32, device=x.device)
        velocity = model.denoise_step(valid, cache, x, t)
        x = x + dt * velocity
        if i + 1 == steps // 2:
            midpoint = x
    return midpoint, x


def gaussian(policy, size, seed, device):
    shape = (size, policy.config.chunk_size, policy.config.max_action_dim)
    return torch.randn(
        shape,
        generator=torch.Generator(device=device).manual_seed(seed),
        dtype=torch.float32,
        device=device,
    )


def squared_error(actual, target, valid, width):
    error = (actual[..., :width].float() - target[..., :width].float()).square()
    return (error * valid.unsqueeze(-1)).sum() / (valid.sum() * width).clamp_min(1)


def native_parity(policy, batch, noise, endpoint):
    policy.reset()
    native = policy.predict_action_chunk(dict(batch), noise=noise)
    torch.testing.assert_close(endpoint[..., : native.shape[-1]], native, rtol=1e-5, atol=1e-5)


def complete_processor_artifacts(teacher, student):
    """Copy missing teacher sidecars and verify the complete student processor payload."""
    expected = processor_artifact_hashes(teacher)
    for name in expected:
        target = student / name
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=target.parent) as temp:
            pending = Path(temp) / "artifact"
            shutil.copyfile(teacher / name, pending)
            pending.replace(target)
    if processor_artifact_hashes(student) != expected:
        raise ValueError("Student processor artifact contents differ from the teacher")
    return expected


def check_export(teacher, student, provenance):
    if read_json(student / "training_provenance.json") != provenance:
        raise ValueError("Unexpected student training history")
    before, after = interface_hashes(teacher), interface_hashes(student)
    before.pop("config.json")
    after.pop("config.json")
    expected_config = {**read_json(teacher / "config.json"), "num_steps": 2}
    if before != after or read_json(student / "config.json") != expected_config:
        raise ValueError(
            "Student inputs, action execution or normalization differ from the teacher"
        )
    changed = 0
    with (
        safe_open(teacher / "model.safetensors", framework="pt", device="cpu") as a,
        safe_open(student / "model.safetensors", framework="pt", device="cpu") as b,
    ):
        names = sorted(a.keys())
        if names != sorted(b.keys()):
            raise ValueError("Exported model has different parameters")
        for name in names:
            x, y = a.get_tensor(name), b.get_tensor(name)
            if x.dtype != y.dtype or x.shape != y.shape:
                raise ValueError(f"Changed inference shape or dtype for {name}")
            if not torch.equal(x, y):
                if not name.startswith(EXPERT) or not torch.isfinite(y).all():
                    raise ValueError(f"Invalid change to {name}")
                changed += 1
    if not changed:
        raise ValueError("All deployed action-expert weights match initialization")
    return changed


def fit(teacher, pre, teacher_path, shared, split, recipe, provenance, output):
    student, _, _ = load_policy(shared, teacher_path)
    student.eval()
    for name, param in student.named_parameters():
        param.requires_grad_(name.startswith(EXPERT))
    promoted = promote_trainable(student, None)
    tunable = {n: p for n, p in student.named_parameters() if p.requires_grad}
    params = list(tunable.values())
    opt = torch.optim.AdamW(params, lr=recipe["learning_rate"], betas=(0.9, 0.95), weight_decay=0.0)
    start = 0
    resume = output / "resume.pt"
    if resume.is_file():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["provenance"] != provenance or set(state["weights"]) != set(tunable):
            raise ValueError("Resume state or trainable parameter names changed")
        with torch.no_grad():
            for name, param in tunable.items():
                param.copy_(state["weights"][name])
        opt.load_state_dict(state["optimizer"])
        start = state["step"]
        del state
    end, size = recipe["updates"], recipe["batch_size"]
    data = open_data(split, shared, "train", teacher.config.chunk_size)
    if len(data) != sum(group["train_frames"] for group in split["tasks"]):
        raise ValueError("Unexpected number of training observations")
    batches = DataLoader(
        data,
        batch_sampler=training_indices(recipe["seed"], start, end, len(data), size),
        num_workers=recipe["workers"],
        generator=torch.Generator().manual_seed(recipe["seed"]),
    )
    width = teacher.config.action_feature.shape[0]
    began = perf_counter()
    for step, sample in enumerate(batches, start=start + 1):
        batch = pre(sample)
        device = batch["observation.state"].device
        valid = (~sample["action_is_pad"]).to(device)
        noise = gaussian(teacher, size, recipe["seed"] + step, device)
        with torch.no_grad():
            target_mid, target_end = integrate_nodes(
                teacher.model, observation_cache(teacher, batch), noise, 10
            )
            if step == start + 1:
                native_parity(teacher, batch, noise, target_end)
        condition = observation_cache(student, batch)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mid, endpoint = integrate_nodes(student.model, condition, noise, 2)
            mid_loss = squared_error(mid, target_mid, valid, width)
            end_loss = squared_error(endpoint, target_end, valid, width)
            loss = (mid_loss + end_loss) * 0.5
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0, error_if_nonfinite=True)
        lr = learning_rate(step, {**recipe, "steps": end})
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        if step == start + 1:
            check_adam_precision(opt)
        if step == start + 1 or step % 50 == 0 or step == end:
            point = {
                "step": step,
                "midpoint_mse": mid_loss.item(),
                "endpoint_mse": end_loss.item(),
                "seconds_per_update": (perf_counter() - began) / (step - start),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            write_json(output / "progress.json", point)
            print(
                f"Distillation {step}/{end}  endpoint MSE {point['endpoint_mse']:.5f}  "
                f"{point['seconds_per_update']:.2f} s/update",
                flush=True,
            )
        if step % recipe["checkpoint_every"] == 0 or step == end:
            temporary = output / "resume.pt.tmp"
            torch.save(
                {
                    "step": step,
                    "provenance": provenance,
                    "optimizer": opt.state_dict(),
                    "weights": {n: p.detach().cpu() for n, p in tunable.items()},
                },
                temporary,
            )
            temporary.replace(resume)
    with TemporaryDirectory(dir=output) as temp:
        model = Path(temp) / "pretrained_model"
        model.mkdir()
        save_policy(student, model, promoted)
        for name in interface_hashes(teacher_path):
            shutil.copyfile(teacher_path / name, model / name)
        complete_processor_artifacts(teacher_path, model)
        write_json(
            model / "config.json", {**read_json(teacher_path / "config.json"), "num_steps": 2}
        )
        write_json(model / "training_provenance.json", provenance)
        model.rename(output / "pretrained_model")


@torch.no_grad()
def offline_check(teacher, student, pre, data, split, recipe):
    original = read_json(RECIPE)
    batches = validation_batches(data, split, {**original, "batch_size": recipe["batch_size"]})
    sums = {"native_2": 0.0, "distilled_2": 0.0}
    frames, points = 0, 0
    for i, sample in enumerate(batches):
        batch = pre(sample)
        device = batch["observation.state"].device
        valid = (~sample["action_is_pad"]).to(device)
        width = teacher.config.action_feature.shape[0]
        noise = gaussian(teacher, len(valid), recipe["seed"] + 100000 + i, device)
        cond = observation_cache(teacher, batch)
        _, reference = integrate_nodes(teacher.model, cond, noise, 10)
        _, native = integrate_nodes(teacher.model, cond, noise, 2)
        _, distilled = integrate_nodes(student.model, observation_cache(student, batch), noise, 2)
        if i == 0:
            native_parity(student, batch, noise, distilled)
        weight = int(valid.sum()) * width
        for name, result in (("native_2", native), ("distilled_2", distilled)):
            sums[name] += squared_error(result, reference, valid, width).item() * weight
        frames += len(valid)
        points += weight
    return {
        "heldout_observations": frames,
        "valid_action_coordinates": points,
        "endpoint_mse_vs_teacher": {k: v / points for k, v in sums.items()},
    }


def train(teacher_path, output, checkpoint_record=None):
    recipe = read_json(ROOT / "configs/few_step_followup.json")
    if (recipe["teacher_flow_steps"], recipe["student_flow_steps"]) != (10, 2):
        raise ValueError("This recipe distills a ten-step teacher into a two-step student")
    _, shared, parent = load_protocol(checkpoint_record)
    if (
        file_sha256(teacher_path / "model.safetensors") != parent["checkpoint_sha256"]
        or interface_hashes(teacher_path) != expected_interface(parent)
        or file_sha256(SPLIT) != parent["split_sha256"]
    ):
        raise ValueError("Teacher, interface or data split differs from the recorded policy")
    split = read_json(SPLIT)
    provenance = {
        "source": "two_step_distillation",
        "parent_sha256": parent["checkpoint_sha256"],
        "dataset": parent["dataset"],
        "split_sha256": parent["split_sha256"],
        "updates": recipe["updates"],
        "seed": recipe["seed"],
        "teacher_flow_steps": 10,
        "student_flow_steps": 2,
    }
    output.mkdir(parents=True, exist_ok=True)
    freeze(output / "recipe.json", {"settings": recipe, "provenance": provenance})
    torch.manual_seed(recipe["seed"])
    torch.set_num_threads(1)
    teacher, pre, _ = load_policy(shared, teacher_path)
    teacher.eval().requires_grad_(False)
    if (
        teacher.config.num_steps != 10
        or not teacher.config.use_cache
        or teacher.config.adapt_to_pi_aloha
        or teacher.model._rtc_enabled()
    ):
        raise ValueError("Expected the recorded native ten-step Spatial policy")
    model = output / "pretrained_model"
    if not model.is_dir():
        fit(teacher, pre, teacher_path, shared, split, recipe, provenance, output)
    changed = check_export(teacher_path, model, provenance)
    # Resumed exports must carry the same tokenizer and processor files as fresh exports.
    artifacts = complete_processor_artifacts(teacher_path, model)
    student, _, _ = load_policy(shared, model)
    student.eval().requires_grad_(False)
    data = open_data(split, shared, "validation", teacher.config.chunk_size)
    report = offline_check(teacher, student, pre, data, split, recipe)
    record = {
        "provenance": provenance,
        "checkpoint_sha256": file_sha256(model / "model.safetensors"),
        "changed_expert_tensors": changed,
        "interface_sha256": interface_hashes(model),
        "processor_artifact_sha256": artifacts,
        "offline_validation": report,
    }
    write_json(output / "student-record.json", record)
    print("Verified native two-step export and held-out endpoint errors", report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-record", type=Path, help="Verification record for the teacher"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".distill.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(args.teacher.resolve(), args.output.resolve(), args.checkpoint_record)
