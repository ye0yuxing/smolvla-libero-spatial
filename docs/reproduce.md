# Train and evaluate SmolVLA

Run these commands from the repository root. The recorded GPU run used Linux x86-64, Python 3.12, PyTorch 2.8 with CUDA 12.8, MuJoCo 3.8.1, an RTX 4090 and EGL rendering. [Pinned dependencies](../requirements.txt) · [Recorded runtime](../results/evaluation/summary.json)

## Verify the published results

```bash
python3 scripts/results.py
```

This standard-library check validates the recorded training recipe, development selection, final weight digest, file checksums, per-task success counts, inference timing counts and matching initial states for all three policies.

## Prepare the GPU environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
python scripts/prepare.py
python scripts/prepare_training.py
```

The preparation scripts download the pinned model, backbone, tokenizer, simulator assets and demonstration shards. They match LIBERO instructions to complete episodes, split each task by episode, fit the normalizers on training frames and write `.cache/training/spatial_split.json`. [Model, dataset and asset revisions](../configs/official_spatial.json) · [Episode split and fitted statistics](../results/training/spatial_split.json)

## Train, select and evaluate the project policy

```bash
python scripts/run_experiment.py --output outputs/spatial
```

The runner starts from `lerobot/smolvla_base`, freezes the vision-language backbone and runs 25,000 AdamW updates at batch 32. Each checkpoint contains the FP32 weights and optimizer state for resumption. Its `pretrained_model/` directory contains the portable, native-dtype LeRobot policy and processors used for rollouts. Repeating the command verifies completed stages and continues from the latest training checkpoint. To measure training throughput and peak GPU memory on your machine, run `python scripts/train.py --profile --micro-batch-size 32 --output outputs/profile-32`. A physical batch of 16 is available with `--micro-batch-size 16`, accumulating two passes per update. [Training recipe](../configs/spatial_train.json) · [Training and selection implementation](../scripts/run_experiment.py)

At 5,000, 10,000, 15,000 and 25,000 updates, the runner measures closed-loop success on states 0, 1 and 3 of all ten tasks. It chooses the highest success count, breaking ties in favor of the earlier checkpoint, and writes `outputs/spatial/selection.json` before evaluating 100 separate final starts. The resulting `outputs/spatial/final-summary.json` records per-task counts, robot failure categories and paired outcomes against the archived base and released controls. The published run selected step 25,000. [Selection record](../results/training/selection.json) · [Final protocol](../configs/final_spatial.json)

## Load the selected model

Read the winner from your run so that the commands also work when a repeat selects an earlier checkpoint.

```bash
step=$(python -c 'import json, sys; print(format(json.load(sys.stdin)["step"], "06d"))' < outputs/spatial/selection.json)
checkpoint="outputs/spatial/training/checkpoints/$step/pretrained_model"
python scripts/check_checkpoint.py "$checkpoint" > outputs/spatial/selected-checkpoint.json
python scripts/evaluate.py --checkpoint "$checkpoint" --task-ids 2 --state-ids 2 \
  --record-scene --output outputs/one-episode
python scripts/check_run.py outputs/one-episode
```

Each rollout includes both cameras, predicted action chunks, executed controls, simulator measurements and the terminal outcome. [Observation and action conventions](interface.md)

## Rerun both controls and rebuild the tables

To collect all three policies on your machine, use the same selected model and its verification record. The first command runs the released and project policies on the fixed final starts. The second loads the original base tensors with the verified Spatial interface. The third builds the compact tables from their raw evidence.

```bash
python scripts/run_final.py --checkpoint "$checkpoint" \
  --checkpoint-record outputs/spatial/selected-checkpoint.json --output outputs/paired
python scripts/run_base.py --checkpoint "$checkpoint" --reference outputs/paired/review \
  --output outputs/base
python scripts/results.py --from-reports outputs/paired/review outputs/base/review \
  --output outputs/tables
python scripts/results.py --verify outputs/tables
```

The per-run `review/` directories contain videos, action and scene traces, checksums and outcome reports. The evidence audits use FFmpeg from the system or the executable supplied by `imageio-ffmpeg`. [Recorded success and failure measurements](../results/evaluation/README.md)
