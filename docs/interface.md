# From a language instruction to a robot action

## Observations

LeRobot maps LIBERO’s observations into two RGB images and an eight-value robot state. The simulator images are rotated 180 degrees to match the dataset convention. The evaluator records the images at 256 × 256 and the policy resizes them with padding to 512 × 512. [Evaluator](../scripts/evaluate.py) · [LeRobot](https://github.com/huggingface/lerobot)

| Signal | Recorded input | Policy input |
| --- | --- | --- |
| External camera | `observation.images.image` | `observation.images.camera1` |
| Wrist camera | `observation.images.image2` | `observation.images.camera2` |
| Robot state | `observation.state` | End-effector XYZ, axis-angle orientation and two finger joint positions |
| Task instruction | LIBERO task description | Text tokens from the pinned SmolVLM2 tokenizer |

The preprocessor normalizes the robot state. For the project-adapted policy and its step-zero control, these statistics come from the training demonstrations. The released LIBERO checkpoint uses its saved processor. The recorder also reads object positions and fingerpad contacts from the simulator for offline failure analysis. The policy observes the images, robot state and language. [Training split and statistics](../results/training/spatial_split.json)

## Action chunks

One policy call uses ten flow integration steps to generate 50 seven-dimensional actions. The saved postprocessor restores the action scale component by component with `normalized_action * std + mean`. The first three components command relative translation, the next three relative rotation and the last the gripper. Positive gripper commands close the fingers.

The synchronous evaluator issues up to 50 commands at 20 Hz before making a new prediction from the latest observation. A full chunk therefore spans 2.5 simulated seconds. If LIBERO reports success mid-chunk, execution stops at that step. The demonstration data declares 10 fps. [Execution settings](../configs/final_spatial.json) · [Recorded dataset revision](https://huggingface.co/datasets/lerobot/libero/tree/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4)

## Read a recorded episode

Frame zero follows a hard reset to the indexed initial state and ten settling actions. Row `t` in `steps.jsonl` contains the state, the command between frames `t` and `t + 1`, the following state, reward and termination flags. Its `chunk_start` locates the prediction in `chunks.jsonl`, and its `chunk_offset` selects the executed action. Videos place the external camera on the left and the wrist camera on the right.

`episode.json` stores the instruction, seed, outcomes, control-step count and per-call inference times in milliseconds. Each inference time measures one GPU-synchronized, batch-one prediction including its ten flow steps. Preprocessing, action unnormalization, physics and video encoding run outside this timer. The first call can include warm-up. [Example with measurements](../results/evaluation/placement-case/README.md)
