# SmolVLA on LIBERO-Spatial

Can a robot-pretrained SmolVLA policy learn ten language-conditioned placement tasks on a single RTX 4090? Using 389 demonstrations, 2,000 fine-tuning updates raised closed-loop success from **0/100 to 24/100**. LeRobot’s released LIBERO checkpoint completed **66/100** on the same starts.

Each instruction asks a simulated Franka Panda to place a particular black bowl on a plate. Phrases such as “on the cookie box” and “next to the cookie box” identify which bowl to pick up. The policy receives the instruction, an external camera view, a wrist camera view and the robot state.

| Policy | Training used for this comparison | Successful placements |
| --- | --- | ---: |
| Robot-pretrained SmolVLA base | Zero Spatial updates, with the same LIBERO input and action processing used after fine-tuning | 0/100 |
| Project-adapted SmolVLA | Base weights, frozen vision-language backbone, 2,000 updates on Spatial demonstrations | 24/100 |
| LeRobot’s released LIBERO policy | Published LIBERO-trained checkpoint with its own saved processors | 66/100 |

The evaluation pairs the policies by instruction, random seed and settled simulator state. It covers ten starts for each of the ten training instructions. [Per-task results and failure analysis](results/evaluation/README.md) · [Training curve and checkpoint verification](results/training/README.md)

## Watch the policy execute

The adapted policy picked up the bowl from the center of the table and placed it on the plate in **94 actions**. In a different scene, both the released and adapted policies lifted the target, but only the released policy reached the placement goal. The frames below show that second pair at reset, after action 68 and at the end of each episode. [Successful adapted rollout](results/evaluation/task-02_state-02_adapted.mp4) · [Placement case with both videos and measured trajectories](results/evaluation/placement-case/README.md)

![External and wrist views of two policies starting from the same scene. The released policy completes placement in 77 actions. The adapted policy reaches the 280-action limit.](results/evaluation/placement-case/placement.jpg)

## Reproduce the results

To check all **300 recorded rollouts**, run this command from the project root with Python 3.12. It uses the standard library to verify file digests, paired starts, success counts and inference timings.

```bash
python3 scripts/results.py
```

To train and evaluate your own checkpoint, the reproduction guide covers installation, the pinned data and models, training, and all three closed-loop evaluations. Each new run saves its full videos, simulator commands and predicted actions under `outputs/`. [GPU reproduction guide](docs/reproduce.md) · [Observation and action conventions](docs/interface.md)

## Credits and license

The project implements the demonstration split, single-GPU adaptation and matched evaluation using the SmolVLA policy and processors from LeRobot, and the tasks, demonstrations and success predicates from LIBERO. [SmolVLA paper](https://arxiv.org/abs/2506.01844) · [LeRobot](https://github.com/huggingface/lerobot) · [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

The MIT license covers original project code and documentation. Upstream software, model weights, demonstrations and scene assets retain their respective licenses. [License](LICENSE)
