"""Read simulator-only measurements after reset and after each control step."""

import hashlib

import numpy as np


class SpatialScene:
    def __init__(self, env):
        wrapper = env._env
        self.domain = wrapper.env
        goal = self.domain.parsed_problem["goal_state"]
        if goal != [["on", "akita_black_bowl_1", "plate_1"]]:
            raise ValueError(f"Unsupported Spatial placement goal: {goal}")
        _, target, plate = goal[0]
        self.target_body = self.domain.obj_body_id[target]
        self.plate_body = self.domain.obj_body_id[plate]
        self.target = self.domain.objects_dict[target]
        self.other_bowls = [
            obj
            for name, obj in self.domain.objects_dict.items()
            if name.startswith("akita_black_bowl_") and name != target
        ]
        self.gripper = self.domain.robots[0].gripper
        state_bytes = np.asarray(wrapper.get_sim_state(), dtype="<f8").tobytes()
        self.initial_sim_sha256 = hashlib.sha256(state_bytes).hexdigest()

    def sample(self, frame):
        xyz = self.domain.sim.data.body_xpos
        return {
            "frame": frame,
            "target_xyz": xyz[self.target_body].tolist(),
            "plate_xyz": xyz[self.plate_body].tolist(),
            "target_between_pads": bool(self.domain._check_grasp(self.gripper, self.target)),
            "other_bowl_between_pads": any(
                self.domain._check_grasp(self.gripper, bowl) for bowl in self.other_bowls
            ),
        }
