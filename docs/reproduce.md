# Train and evaluate SmolVLA

Run these commands from the repository root. The recorded GPU run used Linux x86-64, Python 3.12, PyTorch 2.8 with CUDA 12.8, MuJoCo 3.8.1, an RTX 4090 and EGL rendering. [Pinned dependencies](../requirements.txt) · [Recorded runtime](../results/evaluation/summary.json)

## Verify the published results

```bash
python3 scripts/results.py
```

This standard-library check validates the recorded training recipe, checkpoint selection and the three adaptation controls. It also reconstructs the native solver sweep and the student comparison from individual rollout records, including matched starts, median and P95 action-chunk latency, and paired success changes.

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

## Measure few-step action generation

Use the selected checkpoint above to vary the number of Euler steps per model call. Every call returns 50 actions and the environment executes up to 50 before requesting another prediction.

```bash
python scripts/run_few_steps.py --checkpoint "$checkpoint" \
  --checkpoint-record outputs/spatial/selected-checkpoint.json --output outputs/few-steps
```

The development comparison runs 10, 5, 2 and 1 Euler steps on the same 30 starts. Run order rotates across starts. Each rollout uses the episode seed and the native Gaussian sampler. The audit checks the settled simulator state and the noise digest for every policy call shared by a pair. CUDA-synchronized latency covers action generation, including observation encoding inside the policy. Preprocessing, action unnormalization, simulation and video encoding run outside the timer.

A 1- or 2-step candidate advances when its development success count is within one of the 10-step reference or higher and its median model latency is at least 25% lower. The runner locks the fastest qualifying candidate, then reruns it and the 10-step reference on the 100 published starts. When neither candidate qualifies, the development report lists paired failures for reviewing whether a two-step student is warranted.

If the final comparison has discordant outcomes and the candidate's success-rate difference is within two percentage points of zero or of the −3-point margin, the runner evaluates 100 additional starts. The confirmation results are reported separately. These cutoffs guide the experiment and the tables retain the observed counts. [Experiment settings](../configs/few_step_spatial.json)

Each split's `summary.json` contains per-task results, median and P95 model latency, and the task, initial state and failure stage for every flipped outcome. Each task directory holds the dual-camera videos, actions and contact traces. Repeating the command verifies completed task directories and resumes from the first unfinished one.

### Follow up on the selected result

```bash
python scripts/run_few_step_followup.py --primary outputs/few-steps \
  --teacher "$checkpoint" --checkpoint-record outputs/spatial/selected-checkpoint.json \
  --output outputs/few-step-followup
```

Both commands take the same teacher and verification record. The published run selected the 25,000-update teacher. Completed task directories and training snapshots support resumption. At completion, `report.md` summarizes success, latency and paired gains and losses. `result.json` contains the underlying scores.

If a native candidate qualified, compare it with 10 steps on 50 of the original final starts using a second policy-noise seed. The simulator seed and settled state stay fixed. The audit compares noise hashes within each new pair and across the two repeats. The report includes paired outcomes for each solver under the two seeds.

If both short solvers miss the development gate, the runner trains one two-step student for 2,000 updates on the training demonstrations. With the same observations and Gaussian draw, its first Euler update targets teacher step 5 and its second targets teacher step 10. Gradients through both student updates reach the action expert and action/time projections. The vision-language backbone and state projection stay frozen. The loss covers the seven deployed action coordinates at valid demonstration timesteps. The exported policy runs through the ordinary LeRobot sampler. Its configuration, processor files and training record travel with its weights.

Evaluate the student alongside native 2 and 10 steps on the 30 development starts. The student advances to 100 paired final starts if it gains at least one success over native 2, trails 10 by at most one success or does better, and reduces median latency by at least 25%. Each report includes both comparisons. The held-out demonstration errors describe agreement with the teacher. The paired rollouts measure task success. [Follow-up settings](../configs/few_step_followup.json)

### Load the student or rebuild its evidence

The published experiment followed the distillation branch. To inspect a matched rollout from your completed run, use the exported student and the same teacher. This example records the three policies on one final start.

```bash
student="outputs/few-step-followup/distillation/pretrained_model"
python scripts/evaluate.py --checkpoint "$checkpoint" --student-checkpoint "$student" \
  --flow-steps 10 2 --task-ids 3 --state-ids 2 --record-scene --output outputs/paired-example
```

To train a student directly from your own selected teacher, the standalone entry point is `python scripts/distill_two_steps.py --teacher "$checkpoint" --checkpoint-record outputs/spatial/selected-checkpoint.json --output outputs/student`. It writes a native LeRobot policy to `pretrained_model/` and records teacher/student digests and held-out endpoint errors in `student-record.json`.

To rebuild the compact tables from a completed three-policy experiment, run the following commands. The exporter retains a row per rollout and a timing per action-chunk call, then recomputes the scores before saving the tables. [Published evidence](../results/few-steps/README.md)

```bash
python scripts/few_step_results.py --from-runs outputs/few-steps outputs/few-step-followup \
  --output outputs/few-step-tables
python scripts/few_step_results.py --verify outputs/few-step-tables
```
