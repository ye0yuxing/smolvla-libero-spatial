# SmolVLA on LIBERO-Spatial

SmolVLA completed **70 of 100 placements** after fine-tuning on 389 LIBERO-Spatial demonstrations with a single RTX 4090. On the same 100 starts, LeRobot’s released LIBERO checkpoint completed **66**, and the robot-pretrained base completed **0**.

Each instruction asks a simulated Franka Panda to place a particular black bowl on a plate. Phrases such as “on the cookie box” and “next to the cookie box” identify which bowl to pick up. The policy receives the instruction, an external camera view, a wrist camera view and the robot state.

| Policy | Training used for this comparison | Successful placements |
| --- | --- | ---: |
| Project-adapted SmolVLA | Frozen vision-language backbone, action expert and projections fine-tuned for 25,000 updates on Spatial demonstrations | **70/100** |
| LeRobot’s released LIBERO policy | Published LIBERO-trained checkpoint with its own saved processors | 66/100 |
| Robot-pretrained SmolVLA base | Zero Spatial updates, with the same LIBERO input and action processing as the project policy | 0/100 |

Four checkpoints were evaluated on 30 development starts. The 25,000-update checkpoint won with 25 successes and was then evaluated on ten separate starts for each of the ten training instructions. All three policies share the instruction, seed and settled simulator state for each comparison. [Training and checkpoint selection](results/training/README.md) · [Per-task results and failure analysis](results/evaluation/README.md)

## Watch the policy execute

The adapted policy picked up the bowl from the center of the table and placed it on the plate in **100 actions**. In the paired example below, both policies lifted the target. The released policy completed the placement in 79 actions and the adapted policy timed out. [Successful adapted rollout](results/evaluation/task-02_state-02_adapted.mp4) · [Paired videos and measured trajectories](results/evaluation/placement-case/README.md)

![External and wrist views of the same initial scene, action 75 and each policy’s final observation.](results/evaluation/placement-case/placement.jpg)

## Reproduce the results

To verify the checkpoint selection and the **300 recorded rollout summaries**, run this command from the project root with Python 3.12. It uses the standard library to check file digests, paired starts, success counts and inference timings.

```bash
python3 scripts/results.py
```

The GPU reproduction guide runs the full training and selection workflow from pinned public inputs and shows how to rerun the two controls. Each new run saves its loadable checkpoints, dual-camera videos, simulator commands and predicted actions under `outputs/`. [GPU reproduction guide](docs/reproduce.md) · [Observation and action conventions](docs/interface.md)

## Credits and license

The project implements the demonstration split, single-GPU adaptation and matched evaluation using the SmolVLA policy and processors from LeRobot, and the tasks, demonstrations and success predicates from LIBERO. [SmolVLA paper](https://arxiv.org/abs/2506.01844) · [LeRobot](https://github.com/huggingface/lerobot) · [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

The MIT license covers original project code and documentation. Upstream software, model weights, demonstrations and scene assets retain their respective licenses. [License](LICENSE)
