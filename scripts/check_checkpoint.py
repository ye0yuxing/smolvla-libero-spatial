"""Verify an adapted checkpoint against the pinned base and the train-only statistics."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from final_protocol import file_sha256, interface_hashes
from prepare import input_dir
from prepare_training import RECIPE, SPLIT
from safetensors import safe_open
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="A pretrained_model directory")
    checkpoint = parser.parse_args().checkpoint
    cfg = json.loads(RECIPE.read_text())
    provenance = json.loads((checkpoint / "training_provenance.json").read_text())
    base = input_dir("base", cfg) / "model.safetensors"
    model = checkpoint / "model.safetensors"
    if file_sha256(base) != cfg["base_sha256"] or provenance["initialization"] != cfg["base"]:
        raise ValueError("The initialization does not match the pinned base")
    if file_sha256(SPLIT) != provenance["split_sha256"]:
        raise ValueError("Training split digest mismatch")
    split = json.loads(SPLIT.read_text())
    for pipeline, registry, keys in (
        ("policy_preprocessor", "normalizer_processor", ["observation.state", "action"]),
        ("policy_postprocessor", "unnormalizer_processor", ["action"]),
    ):
        steps = json.loads((checkpoint / f"{pipeline}.json").read_text())["steps"]
        step = next(s for s in steps if s["registry_name"] == registry)
        stored = load_file(checkpoint / step["state_file"])
        for key in keys:
            for stat in ("mean", "std"):
                expected = torch.tensor(split["stats"][key][stat], dtype=torch.float32)
                if not torch.equal(stored[f"{key}.{stat}"], expected):
                    raise ValueError(f"{pipeline} mismatch for {key}.{stat}")
    torch.set_num_threads(1)
    tensors, changed = Counter(), Counter()
    with (
        safe_open(base, framework="pt", device="cpu") as original,
        safe_open(model, framework="pt", device="cpu") as adapted,
    ):
        keys = sorted(original.keys())
        if keys != sorted(adapted.keys()):
            raise ValueError("Checkpoint keys disagree with the base")
        for key in keys:
            a, b = original.get_tensor(key), adapted.get_tensor(key)
            if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(b).all():
                raise ValueError(f"Invalid tensor {key}")
            if key.startswith("model.vlm_with_expert.vlm."):
                group = "vision_language_backbone"
            elif key.startswith("model.vlm_with_expert.lm_expert."):
                group = "action_expert"
            elif key.startswith("model.state_proj."):
                group = "state_projection"
            else:
                group = "action_and_time_projections"
            tensors[group] += 1
            if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
                changed[group] += 1
    if changed["vision_language_backbone"] or any(
        changed[k] == 0 for k in tensors if k != "vision_language_backbone"
    ):
        raise ValueError(f"Unexpected update pattern: {dict(changed)}")
    report = {
        **provenance,
        "checkpoint_sha256": file_sha256(model),
        "interface_sha256": interface_hashes(checkpoint),
        "checkpoint_bytes": model.stat().st_size,
        "train_statistics_verified": True,
        "tensor_comparison": {k: {"total": n, "changed": changed[k]} for k, n in tensors.items()},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
