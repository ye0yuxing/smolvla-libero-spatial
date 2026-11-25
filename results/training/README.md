# Adapting SmolVLA on a single RTX 4090

The training run initialized the full policy from `lerobot/smolvla_base`. It froze the vision-language backbone and updated the action expert and the projections for robot state, actions and flow time. The LIBERO interface uses two cameras, an eight-value state and seven-dimensional relative actions. SmolVLA pads the numeric inputs internally, so the base tensors load with their original shapes. [Model construction and training loop](../../scripts/train.py) · [Base weights](https://huggingface.co/lerobot/smolvla_base/tree/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6)

## Demonstrations

The ten Spatial instructions select **432 complete demonstrations** from the pinned LeRobot LIBERO dataset. Dataset task IDs and simulator task IDs use different orderings. The preparation script matches instructions and splits episodes within each task. It fits the state and action normalization on the **47,630 frames from 389 training episodes**. The other **43 episodes, with 5,340 frames**, supply validation samples. [Episode IDs, task mapping and normalization statistics](spatial_split.json) · [Data preparation](../../scripts/prepare_training.py)

Each sampled observation supplies its language instruction, two images, robot state and up to 50 consecutive action targets. The loss masks actions beyond the end of the episode. Validation uses eight fixed observation windows per task, fixed noise and fixed time sampling. This measures imitation loss on familiar instructions. [Training configuration](../../configs/spatial_train.json)

## Recorded run

| Measurement | Value |
| --- | ---: |
| Updates × batch size | 25,000 × 32 (800,000 sampled observation windows) |
| Trainable / total parameters | 99,880,992 / 450,046,176 |
| Optimizer loop time | 8,875.71 s (about 2 h 28 min) |
| Peak PyTorch GPU memory in that loop | 7.93 GiB allocated, 8.43 GiB reserved |

AdamW used FP32 trainable weights and optimizer moments, BF16 autocast, and gradient clipping at 10. The learning rate reached `1e-4` after 1,000 updates and decayed to `2.5e-6` by update 25,000. The measured time covers optimizer updates and their training batches. Checkpoints contain the FP32 training state for resumption and a LeRobot export in the model’s native inference dtypes. [Training and validation history](training_history.json)

## Choosing the checkpoint

Each candidate ran the same ten instructions from initial states 0, 1 and 3. The selection rule used the highest closed-loop success count, with ties going to the earlier checkpoint. The 25,000-update model won and completed 70 placements on the separate 100-start final evaluation. [Development scores by task](selection.json) · [Final evaluation](../evaluation/README.md)

| Updates | 5,000 | 10,000 | 15,000 | 25,000 |
| --- | ---: | ---: | ---: | ---: |
| Successful development placements | 14/30 | 17/30 | 21/30 | **25/30** |

The fixed-window validation loss reached 0.603 at 3,000 updates and rose to 2.106 at the end. Over the four candidate checkpoints, closed-loop success increased from 14 to 25 on the same 30 starts.

![Training and fixed-window validation loss over 25,000 updates, alongside the four development success counts.](learning-curve.png)

All 345 vision-language backbone tensors were bitwise identical to the base weights. All 145 action-expert tensors, both state-projection tensors and all eight action and time projection tensors changed. The exported processors matched the training-split statistics. [Checkpoint digest and tensor comparison](checkpoint.json) · [Reproduction commands](../../docs/reproduce.md)
