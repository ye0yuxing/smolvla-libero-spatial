"""Run SmolVLA on fixed LIBERO-Spatial initial states and record the executed actions."""

import argparse
import hashlib
import json
import os
import random
from contextlib import closing, nullcontext
from importlib.metadata import distribution, version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import imageio_ffmpeg
import numpy as np
import torch
from final_protocol import (
    ROOT,
    expected_interface,
    file_sha256,
    interface_hashes,
    load_protocol,
    read_json,
)
from lerobot.envs.utils import prepare_canonical_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from prepare import CONFIG, input_dir
from spatial_scene import SpatialScene


def open_suite(config, config_dir):
    # hf-libero prompts interactively without a config. Use a per-process config.
    root = distribution("hf-libero").locate_file("libero/libero").resolve()
    assets = input_dir("assets", config).resolve()
    if not (assets / "scenes").is_dir():
        raise FileNotFoundError("Simulator assets missing. Run python scripts/prepare.py first.")
    paths = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(input_dir("demonstrations", config)),
        "assets": str(assets),
    }
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    (Path(config_dir) / "config.yaml").write_text(json.dumps(paths))
    from libero import libero

    # hf-libero 0.1.5 reads geometry through this cache independently of its YAML.
    libero._assets_path_cache = str(assets)
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[config["suite"]](task_order_index=0)


def load_policy(config, checkpoint=None, *, weights=None):
    model_dir = checkpoint if checkpoint is not None else input_dir("policy", config)
    backbone = str(input_dir("backbone", config))
    cfg = SmolVLAConfig.from_pretrained(model_dir)
    cfg.vlm_model_name = backbone
    policy = SmolVLAPolicy.from_pretrained(
        model_dir if weights is None else weights, config=cfg, strict=True
    )
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=model_dir,
        preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": backbone}},
    )
    return policy, pre, post


def step_zero_weights(checkpoint, checkpoint_record=None):
    recipe = read_json(ROOT / "configs/spatial_train.json")
    _, _, reference = load_protocol(checkpoint_record)
    provenance = read_json(checkpoint / "training_provenance.json")
    keys = ("initialization", "base_sha256", "dataset", "split_sha256", "step", "seed")
    if provenance != {k: reference[k] for k in keys} or (
        file_sha256(checkpoint / "model.safetensors") != reference["checkpoint_sha256"]
    ):
        raise ValueError("The supplied model does not match the verified adapted checkpoint")
    folder = input_dir("base", recipe)
    if recipe["base"] != reference["initialization"] or (
        file_sha256(folder / "model.safetensors") != reference["base_sha256"]
    ):
        raise ValueError("The pretrained weights do not match the training initialization")
    expected = expected_interface(reference)
    if interface_hashes(checkpoint) != expected:
        raise ValueError("The LIBERO interface differs from the verified adapted checkpoint")
    return folder, {
        "source": "pretrained_before_spatial_finetuning",
        **recipe["base"],
        "spatial_updates": 0,
        "base_sha256": reference["base_sha256"],
        "interface_checkpoint_sha256": reference["checkpoint_sha256"],
        "normalization_split_sha256": reference["split_sha256"],
        "interface_sha256": expected,
    }


def model_input(observation, instruction):
    batch = prepare_canonical_observation({k: v[None] for k, v in observation.items()})
    batch["task"] = [instruction]
    return batch


def video_frame(observation):
    return np.concatenate(
        (observation["observation.images.image"], observation["observation.images.image2"]), axis=1
    )


def record_line(stream, record):
    stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")


@torch.inference_mode()
def run_episode(
    env, processors, config, state_id, folder, record_scene=False, *, record_noise=False
):
    from lerobot.sim.integrations.libero import canonical_observation

    policy, pre, post = processors
    seed = config["base_seed"] + 1000 * env.task_id + state_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy.reset()
    pre.reset()
    post.reset()
    raw, _ = env.reset(seed=seed)
    if config.get("flow_noise_seed_offset", 0):
        torch.manual_seed(seed + config["flow_noise_seed_offset"])
    obs = canonical_observation(raw)
    scene = SpatialScene(env) if record_scene else None
    hz = config["control_hz"]
    folder.mkdir()
    frame = video_frame(obs)
    video = imageio_ffmpeg.write_frames(
        str(folder / "cameras.mp4"),
        (frame.shape[1], frame.shape[0]),
        fps=hz,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart", "-threads", "1"],
    )
    inference_ms = []
    simulation_seconds = 0.0
    with (
        closing(video),
        (folder / "steps.jsonl").open("w") as steps,
        (folder / "chunks.jsonl").open("w") as chunks,
        (folder / "scene.jsonl").open("w") if scene else nullcontext() as scene_log,
    ):
        video.send(None)
        video.send(frame)
        if scene:
            record_line(scene_log, scene.sample(0))
        for step in range(config["max_steps"]):
            offset = step % policy.config.n_action_steps
            if offset == 0:
                batch = pre(model_input(obs, env.task_description))
                torch.cuda.synchronize()
                start = perf_counter()
                if record_noise:
                    shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
                    noise = policy.model.sample_noise(shape, batch["observation.state"].device)
                    actions = policy.predict_action_chunk(batch, noise=noise)
                else:
                    actions = policy.predict_action_chunk(batch)
                torch.cuda.synchronize()
                inference_ms.append((perf_counter() - start) * 1000)
                expected = (1, policy.config.chunk_size, 7)
                if tuple(actions.shape) != expected or not torch.isfinite(actions).all():
                    raise ValueError(f"Invalid action chunk at step {step}, shape {actions.shape}")
                chunk = {"step": step, "normalized_actions": actions[0].float().tolist()}
                if record_noise:
                    noise_bytes = noise.cpu().numpy().tobytes()
                    chunk["noise_sha256"] = hashlib.sha256(noise_bytes).hexdigest()
                record_line(chunks, chunk)
            action = post(actions[:, offset])[0].detach().float().cpu().numpy()
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError(f"Invalid environment action at step {step}")
            start = perf_counter()
            raw, reward, terminated, truncated, info = env.step(action)
            simulation_seconds += perf_counter() - start
            next_obs = canonical_observation(raw)
            success = bool(info["is_success"])
            record_line(
                steps,
                {
                    "step": step,
                    "chunk_start": step - offset,
                    "chunk_offset": offset,
                    "state": obs["observation.state"].tolist(),
                    "action": action.tolist(),
                    "next_state": next_obs["observation.state"].tolist(),
                    "reward": float(reward),
                    "success": success,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                },
            )
            obs = next_obs
            video.send(video_frame(obs))
            if scene:
                record_line(scene_log, scene.sample(step + 1))
            if success or terminated or truncated:
                break
    return {
        **({"initial_sim_sha256": scene.initial_sim_sha256} if scene else {}),
        "task_id": env.task_id,
        "instruction": env.task_description,
        "initial_state_id": state_id,
        "seed": seed,
        "steps": step + 1,
        "success": success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reached_step_limit": step + 1 == config["max_steps"],
        "chunk_calls": len(inference_ms),
        "inference_ms": [round(t, 2) for t in inference_ms],
        "simulation_seconds": round(simulation_seconds, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New directory for this run")
    parser.add_argument(
        "--checkpoint", type=Path, help="Project-trained pretrained_model directory"
    )
    parser.add_argument(
        "--step-zero",
        action="store_true",
        help="Load the original base weights with the supplied checkpoint's LIBERO interface",
    )
    parser.add_argument(
        "--checkpoint-record",
        type=Path,
        help="Use this verified training record for a step-zero comparison",
    )
    parser.add_argument("--task-ids", nargs="+", type=int, help="Default: all ten tasks")
    parser.add_argument(
        "--flow-steps",
        nargs="+",
        type=int,
        metavar="K",
        help="Compare Euler step counts at paired starts and save a flow-K directory for each",
    )
    parser.add_argument("--flow-noise-seed-offset", default=0, type=int,
                        help="Change only policy sampling noise after resetting the environment")
    parser.add_argument("--student-checkpoint", type=Path,
                        help="Include a distilled two-step checkpoint in the same rollout order")
    parser.add_argument(
        "--record-scene",
        action="store_true",
        help="Record Spatial goal objects and fingerpad contact for analysis",
    )
    parser.add_argument(
        "--state-ids", nargs="+", type=int, help="Default: the two development states"
    )
    args = parser.parse_args()
    if args.flow_steps and (
        min(args.flow_steps) < 1 or len(set(args.flow_steps)) != len(args.flow_steps)
    ):
        parser.error("--flow-steps must contain distinct positive integers")
    if (args.flow_noise_seed_offset or args.student_checkpoint) and not args.flow_steps:
        parser.error("Noise repeats and student comparisons require --flow-steps")
    if args.student_checkpoint and (not args.checkpoint or args.step_zero):
        parser.error("Student comparisons require the teacher --checkpoint")
    if args.checkpoint_record is not None and not args.step_zero:
        parser.error("--checkpoint-record applies to --step-zero")
    if args.step_zero and args.checkpoint is None:
        parser.error(
            "--step-zero needs the adapted --checkpoint for its configuration and processors"
        )
    config = json.loads(CONFIG.read_text())
    if args.flow_noise_seed_offset:
        config["flow_noise_seed_offset"] = args.flow_noise_seed_offset
    task_ids = args.task_ids or config["task_ids"]
    state_ids = args.state_ids or config["initial_state_ids"]
    if len(set(task_ids)) != len(task_ids) or len(set(state_ids)) != len(state_ids):
        parser.error("Task and state IDs must each be unique.")
    if args.output.exists():
        parser.error("Output directory exists. Choose a fresh directory for each run.")
    if not torch.cuda.is_available():
        parser.error("This baseline requires a CUDA GPU.")
    with TemporaryDirectory(prefix="libero-config-") as tmp:
        suite = open_suite(config, tmp)
        from lerobot.sim.integrations.libero import LiberoEnv, get_task_init_states

        for task_id in task_ids:
            if not 0 <= task_id < len(suite.tasks):
                parser.error(f"Invalid task ID {task_id}")
            count = len(get_task_init_states(suite, task_id))
            if not all(0 <= index < count for index in state_ids):
                parser.error(f"Initial states for task {task_id} must lie in [0, {count})")
        weights, base_record = (
            step_zero_weights(args.checkpoint, args.checkpoint_record)
            if args.step_zero
            else (None, None)
        )
        processors = load_policy(config, args.checkpoint, weights=weights)
        policy = processors[0]
        args.output.mkdir(parents=True)
        manifest = {
            **config,
            "task_ids": task_ids,
            "initial_state_ids": state_ids,
            "control_mode": "relative",
            "chunk_size": policy.config.chunk_size,
            "n_action_steps": policy.config.n_action_steps,
            "flow_steps": policy.config.num_steps,
            "gpu": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
            "mujoco_gl": os.environ["MUJOCO_GL"],
            "versions": {
                p: version(p)
                for p in (
                    "lerobot",
                    "hf-libero",
                    "robosuite",
                    "mujoco",
                    "torch",
                    "torchvision",
                    "transformers",
                    "numpy",
                )
            },
        }
        if args.step_zero:
            manifest["policy"] = base_record
            manifest["weight_sha256"] = {
                **config["weight_sha256"],
                "policy": base_record["base_sha256"],
            }
        elif args.checkpoint is not None:
            provenance = json.loads((args.checkpoint / "training_provenance.json").read_text())
            with (args.checkpoint / "model.safetensors").open("rb") as weights:
                digest = hashlib.file_digest(weights, "sha256").hexdigest()
            manifest["policy"] = {"source": "project_checkpoint", **provenance}
            manifest["weight_sha256"] = {**config["weight_sha256"], "policy": digest}
        if args.record_scene:
            manifest["scene_trace_version"] = 1
        counts = args.flow_steps or [policy.config.num_steps]
        variants = {f"flow-{k}": (k, processors, manifest) for k in counts}
        if args.student_checkpoint:
            student = load_policy(config, args.student_checkpoint)
            provenance = read_json(args.student_checkpoint / "training_provenance.json")
            if (
                provenance.get("source") != "two_step_distillation"
                or provenance.get("parent_sha256") != manifest["weight_sha256"]["policy"]
                or student[0].config.num_steps != 2
                or student[0].config.chunk_size != policy.config.chunk_size
                or student[0].config.n_action_steps != policy.config.n_action_steps
            ):
                raise ValueError("Student provenance or deployment configuration differs from the plan")
            digest = file_sha256(args.student_checkpoint / "model.safetensors")
            variants["student-2"] = (2, student, {
                **manifest, "policy": provenance,
                "weight_sha256": {**manifest["weight_sha256"], "policy": digest},
            })
        labels = list(variants)
        roots = {}
        episodes = {label: [] for label in labels}
        for label, (k, _, base_manifest) in variants.items():
            root = args.output / label if args.flow_steps else args.output
            root.mkdir(exist_ok=True)
            roots[label] = root
            protocol = {**base_manifest, "flow_steps": k}
            if args.flow_steps:
                protocol["noise_protocol"] = "episode_seeded_native_gaussian_sha256"
            (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        for task_id in task_ids:
            for index, state_id in enumerate(state_ids):
                # Balance which solver goes first, including the first call after each task reload.
                shift = (task_id + index) % len(labels)
                for label in labels[shift:] + labels[:shift]:
                    k, active, _ = variants[label]
                    active[0].config.num_steps = k
                    folder = roots[label] / f"task-{task_id:02d}_state-{state_id:02d}"
                    with closing(
                        LiberoEnv(
                            suite,
                            task_id,
                            config["suite"],
                            obs_type="pixels_agent_pos",
                            episode_index=state_id,
                            control_mode="relative",
                            hard_reset=True,
                            num_steps_wait=config["settle_steps"],
                            control_freq=config["control_hz"],
                        )
                    ) as env:
                        episode = run_episode(
                            env, active, config, state_id, folder, args.record_scene,
                            record_noise=bool(args.flow_steps),
                        )
                    (folder / "episode.json").write_text(json.dumps(episode, indent=2) + "\n")
                    episodes[label].append(episode)
                    flow_label = f"  flow={k}" if args.flow_steps else ""
                    if args.student_checkpoint:
                        flow_label += f"  policy={label}"
                    print(
                        f"{folder.name}{flow_label}  success={episode['success']}  steps={episode['steps']}",
                        flush=True,
                    )
        for label, rows in episodes.items():
            successes = sum(ep["success"] for ep in rows)
            summary = {"successes": successes, "n_episodes": len(rows), "episodes": rows}
            (roots[label] / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            flow_label = f" for {label}" if args.flow_steps else ""
            print(f"Completed {len(rows)} episodes{flow_label} with {successes} successes.")


if __name__ == "__main__":
    main()
