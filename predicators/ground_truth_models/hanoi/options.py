"""Ground-truth options for the Towers of Hanoi environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.hanoi import HanoiEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class HanoiGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the Towers of Hanoi environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"hanoi"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        del env_name, predicates  # unused

        robot_type = types["robot"]
        disk_type = types["disk"]
        peg_type = types["peg"]

        # Pick a disk (the topmost disk in its column).
        Pick = utils.SingletonParameterizedOption(
            # variables: [robot, disk to pick]
            "Pick",
            cls._create_pick_policy(action_space),
            types=[robot_type, disk_type])

        # Stack the held disk onto another (larger, clear) disk.
        Stack = utils.SingletonParameterizedOption(
            # variables: [robot, disk on which to stack the held disk]
            "Stack",
            cls._create_stack_policy(action_space),
            types=[robot_type, disk_type])

        # Place the held disk onto an empty peg.
        PutOnPeg = utils.SingletonParameterizedOption(
            # variables: [robot, peg on which to place the held disk]
            "PutOnPeg",
            cls._create_putonpeg_policy(action_space),
            types=[robot_type, peg_type])

        return {Pick, Stack, PutOnPeg}

    @classmethod
    def _create_pick_policy(cls, action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, disk = objects
            x = state.get(disk, "pose_x")
            z = state.get(disk, "pose_z")
            arr = np.array([x, z, HanoiEnv.closed_fingers], dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_stack_policy(cls, action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, disk = objects
            x = state.get(disk, "pose_x")
            z = state.get(disk, "pose_z") + HanoiEnv.disk_height
            arr = np.array([x, z, HanoiEnv.open_fingers], dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_putonpeg_policy(cls,
                                action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, peg = objects
            x = state.get(peg, "pose_x")
            z = HanoiEnv.base_z
            arr = np.array([x, z, HanoiEnv.open_fingers], dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

