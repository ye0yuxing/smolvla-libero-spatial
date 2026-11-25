"""Fetch the pinned public inputs for the LIBERO-Spatial evaluation."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "official_spatial.json"
INPUTS = ROOT / ".cache" / "inputs"


def input_dir(key, config):
    return INPUTS / key / config[key]["revision"]


def fetch_snapshot(key, config):
    spec = config[key]
    print(f"Fetching {spec['repo_id']} at {spec['revision']}", flush=True)
    snapshot_download(
        **spec,
        repo_type="dataset" if key == "assets" else "model",
        local_dir=input_dir(key, config),
        ignore_patterns=["onnx/*", ".gitattributes"],
    )
    if key in config["weight_sha256"]:
        with (input_dir(key, config) / "model.safetensors").open("rb") as weights:
            digest = hashlib.file_digest(weights, "sha256").hexdigest()
        if digest != config["weight_sha256"][key]:
            raise ValueError(f"Weight checksum mismatch for {spec['repo_id']}")


def prepare():
    config = json.loads(CONFIG.read_text())
    for key in ("policy", "backbone", "assets"):
        fetch_snapshot(key, config)
    metadata = hf_hub_download(
        **config["demonstrations"],
        repo_type="dataset",
        filename="meta/info.json",
        local_dir=input_dir("demonstrations", config),
    )
    features = json.loads(Path(metadata).read_text())["features"]
    policy_config = json.loads((input_dir("policy", config) / "config.json").read_text())
    for key, expected in (
        ("observation.state", policy_config["input_features"]["observation.state"]["shape"]),
        ("action", policy_config["output_features"]["action"]["shape"]),
    ):
        if features[key]["shape"] != expected:
            raise ValueError(
                f"Dataset {key} shape {features[key]['shape']} does not match {expected}"
            )
    for key in ("observation.images.image", "observation.images.image2"):
        if features[key]["shape"] != [256, 256, 3]:
            raise ValueError(f"Unexpected dataset camera feature {key}: {features[key]}")
    print("Pinned inputs are ready. Dataset camera, state and action shapes match.")


if __name__ == "__main__":
    prepare()
