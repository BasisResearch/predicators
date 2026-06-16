"""Motion Planning in PyBullet."""
from __future__ import annotations

from typing import Collection, Iterator, Optional, Sequence

import numpy as np
import pybullet as p
from gym.spaces import Box
from numpy.typing import NDArray

from predicators import utils
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.link import get_link_state
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG


def run_motion_planning(
    robot: SingleArmPyBulletRobot,
    initial_positions: JointPositions,
    target_positions: JointPositions,
    collision_bodies: Collection[int],
    seed: int,
    physics_client_id: int,
    held_object: Optional[int] = None,
    base_link_to_held_obj: Optional[NDArray] = None,
) -> Optional[Sequence[JointPositions]]:
    """Run BiRRT to find a collision-free sequence of joint positions.

    Note that this function changes the state of the robot.
    """
    rng = np.random.default_rng(seed)
    # BiRRT plans in the arm-joint space. For mobile robots, action_space also
    # includes base-delta dims (appended last); strip them so sampled configs
    # match the arm joints that set_joints / forward_kinematics expect. For
    # fixed-base robots (base_action_dim == 0) this is a no-op.
    joint_space = robot.action_space
    base_dim = int(getattr(robot, "base_action_dim", 0))
    if base_dim > 0:
        joint_space = Box(low=np.asarray(joint_space.low[:-base_dim]),
                          high=np.asarray(joint_space.high[:-base_dim]),
                          dtype=np.float32)
    joint_space.seed(seed)
    num_interp = CFG.pybullet_birrt_extend_num_interp

    def _sample_fn(pt: JointPositions) -> JointPositions:
        new_pt: JointPositions = list(joint_space.sample())
        # Don't change the fingers.
        new_pt[robot.left_finger_joint_idx] = pt[robot.left_finger_joint_idx]
        new_pt[robot.right_finger_joint_idx] = pt[robot.right_finger_joint_idx]
        return new_pt

    def _set_state(pt: JointPositions) -> None:
        robot.set_joints(pt)
        if held_object is not None:
            assert base_link_to_held_obj is not None
            world_to_base_link = get_link_state(
                robot.robot_id,
                robot.end_effector_id,
                physics_client_id=physics_client_id).com_pose
            world_to_held_obj = p.multiplyTransforms(world_to_base_link[0],
                                                     world_to_base_link[1],
                                                     base_link_to_held_obj[0],
                                                     base_link_to_held_obj[1])
            p.resetBasePositionAndOrientation(
                held_object,
                world_to_held_obj[0],
                world_to_held_obj[1],
                physicsClientId=physics_client_id)

    def _extend_fn(pt1: JointPositions,
                   pt2: JointPositions) -> Iterator[JointPositions]:
        pt1_arr = np.array(pt1)
        pt2_arr = np.array(pt2)
        num = int(np.ceil(max(abs(pt1_arr - pt2_arr)))) * num_interp
        if num == 0:
            yield pt2
        for i in range(1, num + 1):
            yield list(pt1_arr * (1 - i / num) + pt2_arr * i / num)

    def _collision_fn(pt: JointPositions) -> bool:
        _set_state(pt)
        p.performCollisionDetection(physicsClientId=physics_client_id)
        # Use a penetration margin to distinguish real collisions from
        # resting-on-surface contacts (e.g. a grasped object sitting on a
        # table, or a gripper touching a just-released object).
        # getContactPoints returns tuples where index 8 is contactDistance:
        # negative = penetration, ~0 = touching, positive = separation.
        # Only penetration deeper than the margin counts as a collision.
        margin = CFG.pybullet_birrt_contact_margin
        for body in collision_bodies:
            contacts = p.getContactPoints(robot.robot_id,
                                          body,
                                          physicsClientId=physics_client_id)
            if any(c[8] < margin for c in contacts):
                return True
            if held_object is not None:
                contacts = p.getContactPoints(
                    held_object, body, physicsClientId=physics_client_id)
                if any(c[8] < margin for c in contacts):
                    return True
        return False

    def _distance_fn(from_pt: JointPositions, to_pt: JointPositions) -> float:
        # NOTE: only using positions to calculate distance. Should use
        # orientations as well in the near future.
        from_ee = robot.forward_kinematics(from_pt).position
        to_ee = robot.forward_kinematics(to_pt).position
        return sum(np.subtract(from_ee, to_ee)**2)

    birrt = utils.BiRRT(_sample_fn,
                        _extend_fn,
                        _collision_fn,
                        _distance_fn,
                        rng,
                        num_attempts=CFG.pybullet_birrt_num_attempts,
                        num_iters=CFG.pybullet_birrt_num_iters,
                        smooth_amt=CFG.pybullet_birrt_smooth_amt)

    path = birrt.query(initial_positions, target_positions)
    if path is not None and CFG.pybullet_birrt_path_subsample_ratio > 1:
        ratio = CFG.pybullet_birrt_path_subsample_ratio
        last = path[-1]
        path = [path[i] for i in range(0, len(path), ratio)]
        # Always include the final waypoint.
        if path[-1] is not last:
            path.append(last)
    return path
