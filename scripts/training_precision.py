"""Train the action expert in FP32 and export the policy in its native inference dtypes."""

import torch
from safetensors import safe_open
from safetensors.torch import save_file


@torch.no_grad()
def promote_trainable(policy, checkpoint):
    promoted = {}
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad or parameter.dtype == torch.float32:
            continue
        if parameter.dtype != torch.bfloat16:
            raise ValueError(f"Unexpected trainable parameter dtype for {name}: {parameter.dtype}")
        parameter.data = parameter.detach().float()
        promoted[name] = parameter
    if not promoted:
        raise ValueError("Expected trainable BF16 expert parameters in the native SmolVLA model")
    if checkpoint is not None:
        with safe_open(checkpoint / "fp32_master.safetensors", framework="pt", device="cpu") as f:
            if set(f.keys()) != set(promoted):
                raise ValueError("FP32 master parameter names disagree with the model")
            for name, parameter in promoted.items():
                master = f.get_tensor(name)
                if master.dtype != torch.float32 or master.shape != parameter.shape:
                    raise ValueError(f"Invalid FP32 master for {name}")
                if not torch.equal(master.bfloat16().float(), parameter.cpu()):
                    raise ValueError(
                        f"FP32 master and native inference weights disagree for {name}"
                    )
                parameter.copy_(master)
    return promoted


@torch.no_grad()
def save_policy(policy, model, promoted):
    if not promoted:
        policy.save_pretrained(model)
        return
    # Keep exact training weights for resume. LeRobot loads the ordinary export for rollouts.
    master = {name: param.detach() for name, param in promoted.items()}
    save_file(
        {name: value.cpu() for name, value in master.items()},
        model.parent / "fp32_master.safetensors",
    )
    try:
        for name, parameter in promoted.items():
            parameter.data = master[name].bfloat16()
        policy.save_pretrained(model)
    finally:
        for name, parameter in promoted.items():
            parameter.data = master[name]


def check_adam_precision(optimizer):
    states = list(optimizer.state.values())
    if not states or any(
        state[key].dtype != torch.float32 for state in states for key in ("exp_avg", "exp_avg_sq")
    ):
        raise ValueError("AdamW moments must be FP32 for the long-budget recipe")
