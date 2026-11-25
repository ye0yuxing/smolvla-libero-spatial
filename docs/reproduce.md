# Train and evaluate SmolVLA

Run the commands from the repository root. The recorded environment used Linux x86-64, Python 3.12, PyTorch 2.8 with CUDA 12.8, MuJoCo 3.8.1 and an RTX 4090. GPU evaluation needs a working EGL renderer. The video audit uses FFmpeg from the system or the executable supplied by `imageio-ffmpeg`. [Pinned dependencies](../requirements.txt) · [Recorded runtime](../results/evaluation/summary.json)

## Install and prepare the inputs

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
python scripts/prepare.py
python scripts/prepare_training.py
```

`prepare.py` downloads the released LIBERO policy, vision-language backbone, tokenizer, scene assets and dataset feature metadata. `prepare_training.py` downloads the base policy, matches the ten instructions to whole demonstration episodes, and reads the state, action and camera shards for training. It writes the episode split and train-only normalization statistics to `.cache/training/spatial_split.json`. Subsequent training and rollouts use the local snapshots. [Model, data and asset revisions](../configs/official_spatial.json) · [Training recipe](../configs/spatial_train.json)

## Fine-tune and verify a checkpoint

```bash
python scripts/train.py --output outputs/spatial-adaptation
checkpoint=outputs/spatial-adaptation/checkpoints/002000/pretrained_model
python scripts/check_checkpoint.py "$checkpoint" > outputs/spatial-adaptation/verified.json
```

Training exports a model and its processors every 500 updates. To resume an interrupted run from its latest completed checkpoint, repeat the training command with `--resume`. The verifier checks the pinned initialization, split, normalization, parameter changes and interface files. An independent training run can produce a different weight digest. The commands below use the digest recorded for **your** checkpoint.

## Run the matched comparisons

```bash
python scripts/run_final.py \
  --checkpoint "$checkpoint" --checkpoint-record outputs/spatial-adaptation/verified.json \
  --output outputs/paired
python scripts/run_base.py \
  --checkpoint "$checkpoint" --reference outputs/paired/review \
  --output outputs/base
python scripts/results.py \
  --from-reports outputs/paired/review outputs/base/review \
  --output outputs/tables
```

The first runner collects 200 episodes, alternating which policy runs first by task. The second collects 100 more with the original base tensors and the adapted policy’s interface. It matches each start against the first run and reuses its two scores. Rerunning either command verifies and skips completed task directories. Choose a new directory for each distinct experiment.

The per-run `review/` directories hold all raw videos and traces, their checksums and the detailed reports. The final command combines them into one episode table, per-task counts and a summary. Recheck those tables at any time with `python scripts/results.py --verify outputs/tables`. [Recorded success and failure measurements](../results/evaluation/README.md)

## Inspect one rollout

To execute one matched start with your adapted policy, run the following in the same shell after training.

```bash
python scripts/evaluate.py --checkpoint "$checkpoint" --task-ids 2 --state-ids 2 \
  --record-scene --output outputs/one-episode
python scripts/check_run.py outputs/one-episode
```

Each episode has a two-camera video, the full predicted chunks, the commands applied at each control step, simulator scene measurements and the terminal outcome. [Trace fields and timing](interface.md)

## Train with a larger budget

The long recipe uses the same robot-pretrained initialization, 389 Spatial demonstrations and frozen vision-language backbone. It runs 25,000 AdamW updates with a batch of 32, a 1,000-update warm-up and cosine decay from `1e-4` to `2.5e-6`. Trainable parameters and optimizer moments use FP32, with BF16 autocast for the forward pass. Each training checkpoint contains the exact FP32 weights for resumption and a native-dtype LeRobot export for deployment. [Long recipe](../configs/spatial_long.json)

Profile both physical batch sizes on the training data. Each command starts from the base weights, warms up for 10 updates and measures the next 40. Both process 32 windows per optimizer update. Microbatches of 16 contribute to the loss in proportion to their valid action timesteps.

```bash
python scripts/train.py --long-budget --profile --micro-batch-size 16 --output outputs/profile-16
python scripts/train.py --long-budget --profile --micro-batch-size 32 --output outputs/profile-32
python scripts/run_long.py --micro-batch-size 32 --output outputs/spatial-long
```

Use the measured windows per second and peak GPU memory to choose 16 or 32 for the last command. Repeat that command with the same arguments to resume. The first 10 updates belong to the training run. The runner saves their weights, executes one deployment check and restores the training state before continuing.

At 5k, 10k, 15k and 25k updates, each candidate runs the ten tasks from development states 0, 1 and 3. All policies execute up to 50 actions per observation. The runner picks the highest success count over the 30 starts, choosing the earlier checkpoint on ties. It writes the decision to `selection.json` before the selected policy sees the 100 final starts. The resulting `final-summary.json` includes paired outcomes against the recorded policies, per-task success and failure counts, and simulator-state digests. [Training and selection runner](../scripts/run_long.py)
