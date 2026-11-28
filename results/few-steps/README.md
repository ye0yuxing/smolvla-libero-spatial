# Two-step action generation on LIBERO-Spatial

On 100 matched starts, the distilled two-step student and the ten-step teacher each completed 70 placements. Median time to generate a 50-action chunk fell by **47.3%**. Using the teacher weights with the same two-step solver completed 68 placements at essentially the same latency as the student. [All rollout records](episodes.jsonl) · [Recorded scores and runtime](summary.json)

| Final evaluation | Success | Chunk calls | Median ms/chunk | P95 ms/chunk |
| --- | ---: | ---: | ---: | ---: |
| Ten-step teacher | 70/100 | 362 | 381.66 | 427.77 |
| Native two-step | 68/100 | 368 | 200.78 | 209.08 |
| Distilled two-step | 70/100 | 360 | 201.02 | 207.96 |

The 10 trained instructions each run from 10 final states disjoint from development. Policies share the instruction, seed, settled simulator state and Gaussian noise at every shared decision point. Solver order rotates across starts. Timings pool individual, CUDA-synchronized batch-one predictions on a single RTX 4090, including noise sampling, observation encoding and action generation. Input preprocessing, action unnormalization, video and simulation run outside the timer. All policies execute up to 50 commands between observations. [Evaluation code](../../scripts/evaluate.py) · [Fixed starts and control settings](../../configs/final_spatial.json)

## How the student was chosen

The fine-tuned teacher won selection at 25,000 updates. Shortening its native solver to one or two Euler steps missed the development requirement of at most one fewer success than the ten-step teacher and at least 25% lower median latency. These four policies ran on the same 30 starts. [Teacher training](../training/README.md) · [Native selection rule](../../configs/few_step_spatial.json)

| Native solver | 10 steps | 5 steps | 2 steps | 1 step |
| --- | ---: | ---: | ---: | ---: |
| Successful placements | 25/30 | 23/30 | 19/30 | 22/30 |
| Median ms/chunk | 394.0 | 300.1 | 194.9 | 158.5 |

The student initializes from the teacher. Given identical observations and Gaussian noise, it targets the teacher’s fifth and tenth Euler iterates. Its second update starts from its own predicted midpoint. The loss averages midpoint and endpoint squared errors over seven normalized action coordinates and valid demonstration timesteps. Training updates the action expert and action/time projections for 2,000 AdamW steps. The vision-language backbone and state projection stay frozen. Across 80 held-out demonstration observations, endpoint MSE against the teacher fell from 0.00993 with native two-step sampling to 0.00820 after distillation. [Trainer](../../scripts/distill_two_steps.py) · [Student weight, processor and validation record](student.json)

In a separate evaluation session on the same 30 development starts, the teacher, native two-step policy and student completed **25, 19 and 24** placements. The student gained five and lost zero against native two-step inference. It met the development rule and advanced to the 100 final starts. [Follow-up selection rule](../../configs/few_step_followup.json)

## Where the outcomes changed

Each row pairs the student with one reference on the same 100 starts. The teacher and student had identical total successes with 14 individual outcomes changing. Against native two-step inference the student gained eight starts and lost six. Six of the eight gains were cases in which the native policy lifted the target and failed to place it. [Episode outcomes and timings](episodes.jsonl)

| Reference | Both succeed | Only student succeeds | Only reference succeeds | Both fail |
| --- | ---: | ---: | ---: | ---: |
| Ten-step teacher | 63 | 7 | 7 | 23 |
| Native two-step | 62 | 8 | 6 | 24 |

| Task | Target location | Ten-step teacher | Native two-step | Distilled two-step |
| ---: | --- | ---: | ---: | ---: |
| 0 | Between plate and ramekin | 7/10 | 7/10 | 8/10 |
| 1 | Next to ramekin | 5/10 | 3/10 | 4/10 |
| 2 | Table center | 9/10 | 10/10 | 9/10 |
| 3 | On cookie box | 9/10 | 8/10 | 9/10 |
| 4 | Inside cabinet drawer | 8/10 | 9/10 | 9/10 |
| 5 | On ramekin | 3/10 | 3/10 | 3/10 |
| 6 | Next to cookie box | 10/10 | 8/10 | 10/10 |
| 7 | On stove | 6/10 | 8/10 | 4/10 |
| 8 | Next to plate | 6/10 | 7/10 | 7/10 |
| 9 | On cabinet | 7/10 | 5/10 | 7/10 |

Two final starts illustrate both directions of change. Each clip contains the external and wrist camera views at the recorded 20 Hz.

| Matched start | Native two-step | Distilled two-step |
| --- | --- | --- |
| Task 3, state 2 | Lifted the target and timed out at 280 actions. [Video](cases/task-03_state-02_native2.mp4) | Placed the target at action 78. [Video](cases/task-03_state-02_student2.mp4) |
| Task 7, state 17 | Placed the target at action 119. [Video](cases/task-07_state-17_native2.mp4) | Lifted the target and timed out at 280 actions. [Video](cases/task-07_state-17_student2.mp4) |

The final measurements cover these initial states, known instructions and one GPU. The equal 70/100 totals describe this sample. New instructions, physical robot execution and end-to-end cycle time would require separate measurements.

## Verify or reproduce

From the repository root, `python3 scripts/few_step_results.py` verifies all 510 few-step episode records, including the two development panels. Each line in `episodes.jsonl` identifies the panel, solver, task, start, outcome and executed action count. Its `inference_ms` array contains the latency of every action-chunk call in that rollout. The GPU workflow also saves full camera, action and contact traces. [Reproduction guide](../../docs/reproduce.md)
