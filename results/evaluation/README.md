# Closed-loop results on LIBERO-Spatial

The project-adapted checkpoint completed **70/100** placements, LeRobot’s released LIBERO checkpoint completed **66/100**, and the robot-pretrained base completed **0/100**. The two adapted policies both succeeded on 53 starts and both failed on 17. The project checkpoint alone succeeded on 17 starts and the released checkpoint alone succeeded on 13. [All 300 episode records](episodes.csv) · [Per-task counts](per-task.csv)

| Task | Target bowl location named in the instruction | Base | Adapted | Released |
| ---: | --- | ---: | ---: | ---: |
| 0 | Between the plate and the ramekin | 0/10 | 7/10 | 9/10 |
| 1 | Next to the ramekin | 0/10 | 5/10 | 8/10 |
| 2 | Table center | 0/10 | 9/10 | 7/10 |
| 3 | On the cookie box | 0/10 | 9/10 | 10/10 |
| 4 | Inside the cabinet’s top drawer | 0/10 | 8/10 | 3/10 |
| 5 | On the ramekin | 0/10 | 3/10 | 0/10 |
| 6 | Next to the cookie box | 0/10 | 10/10 | 9/10 |
| 7 | On the stove | 0/10 | 6/10 | 6/10 |
| 8 | Next to the plate | 0/10 | 6/10 | 7/10 |
| 9 | On the wooden cabinet | 0/10 | 7/10 | 7/10 |

## What the robot did

| Measurement | Base | Adapted | Released |
| --- | ---: | ---: | ---: |
| Failure with no sampled two-pad contact on the target | 99 | 19 | 20 |
| Failure with contact and less than 3 cm of lift during contact | 0 | 4 | 1 |
| Failure after reaching the lift threshold during contact | 1 | 7 | 13 |
| Median chunk inference on the RTX 4090 | 392.55 ms | 398.33 ms | 381.67 ms |

At each 20 Hz sample the recorder checks whether both fingerpad groups touch the target and measures the target’s height relative to its settled starting position. The 3 cm threshold describes a lift while in contact. LIBERO’s own bowl-on-plate predicate determines success. Brief contacts can occur between samples. The per-episode table includes the first contact and lift frames, the final horizontal bowl–plate distance and every synchronized chunk timing. [Aggregate results and runtime versions](summary.json)

On task 2, state 2, the adapted policy grasped the bowl from the table center and completed placement in 100 actions. On task 0, state 27, both the adapted and released policies lifted the requested bowl, and the adapted policy timed out during placement. The case study aligns the two camera views with the object measurements. [Successful adapted rollout](task-02_state-02_adapted.mp4) · [Matched placement failure](placement-case/README.md)

## Evaluation conditions

Each policy runs once on ten predefined initial states per task. The state IDs are 2, 7, 12, 17, 22, 27, 32, 37, 42 and 47. The seed is `1000 + 1000 × task_id + state_id`. Every triplet shares the same instruction, seed and post-settling simulator-state digest. Episodes end at success, termination, truncation or 280 control actions. These ten instructions also appear in training. The recorded comparison measures performance under additional starting configurations. The released checkpoint has a different training history and budget. [Frozen protocol and weight digests](comparison.json) · [Official training configuration](https://huggingface.co/lerobot/smolvla_libero/blob/a24f1f380461b665158d52acd140152c79e8b36d/train_config.json)

The selected checkpoint was run on all 100 final starts after development selection. The base and released-policy rows reuse their original closed-loop runs on those same starts. The base control’s model configuration, camera mapping and normalization files are byte-identical to the selected checkpoint’s interface files. The raw videos, executed actions, weights and initial-state digests were checked when assembling this comparison. [Selected checkpoint and interface digests](../training/checkpoint.json) · [Development selection](../training/selection.json)

Run `python3 scripts/results.py` from the project root to verify the compact tables. A fresh GPU run saves the camera streams, scene measurements and actions for each episode. [Reproduction guide](../../docs/reproduce.md)
