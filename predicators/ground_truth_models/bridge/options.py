"""Ground-truth options (skills) for the bridge environment.

Built on the shared skill factories: PickBlock / PickBottle (pick),
Place (generic place), MoveTo (generic move-through-pose), and Wait.
All geometry lives in continuous params supplied by the samplers in
``processes.py`` -- the skill set itself carries ZERO glue semantics.
In particular there is no ApplyGlue skill: the glue bottle is picked
and moved with the same generic skills as every other object.
"""

from typing import ClassVar, Dict, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
from gym.spaces import Box

from predicators.envs.pybullet_bridge import PyBulletBridgeEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.skill_factories import SkillConfig, \
    build_params_space, create_move_to_skill, create_pick_skill, \
    create_place_skill, create_wait_option, shared_skill_robot, \
    shared_skill_simulator
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, \
    Predicate, State, Type

# Place params: canonical (x, y, release_z, yaw) order. All three
# position params are HELD-OBJECT coordinates (live-compensated, see
# compensate_held_offset / compensate_held_z below): release_z is the
# held object's CENTER height at the END OF THE DESCENT -- the skill
# then settles to first contact before releasing (see
# settle_to_contact_depth below), so release_z only needs to clear the
# scene; the block touches down with essentially no free fall. A span
# descends over the table to ~0.428, a span seated on the leg tops to
# ~0.545.
_BRIDGE_PLACE_PARAMS = [
    ("target_x (world x position for the held object)",
     PyBulletBridgeEnv.workspace_x_lo, PyBulletBridgeEnv.workspace_x_hi),
    ("target_y (world y position for the held object)", 1.1, 1.6),
    ("release_z (world z height of the held object's center at release)", 0.41,
     0.60),
    ("target_yaw (placement orientation in radians)", -np.pi, np.pi),
]

# MoveTo params: an absolute world pose for the HELD OBJECT'S CENTER
# (the EE itself when empty-handed) -- all axes live-compensated by the
# EE-to-held offset, so the sampled target is exactly where the held
# object goes regardless of grasp depth or the pick's IK residual. The
# glue samplers use this to land the held bottle's tip on a face target
# (tip = center minus the bottle half-height).
#
# The x bounds extend one span half-length past the block workspace: a
# block STAGED near the workspace edge presents its end face up to
# span_half_x outside it, and clamping the target to the block
# workspace silently parked the bottle tip short of the face. With the
# wider box a genuinely unreachable target fails IK loudly (and
# triggers a replan) instead of "succeeding" at a clamped pose.
#
# NOTE (hidden-dynamics hygiene): this file is copied verbatim into the
# agent sandbox as reference/options.py. Do not document any hidden
# dynamics here -- discovering them is the agent's job.
_BRIDGE_MOVE_TO_PARAMS = [
    ("target_x (world x position for the held object, or the EE if "
     "empty-handed)",
     PyBulletBridgeEnv.workspace_x_lo - PyBulletBridgeEnv.span_half_extents[0],
     PyBulletBridgeEnv.workspace_x_hi +
     PyBulletBridgeEnv.span_half_extents[0]),
    ("target_y (world y position for the held object, or the EE if "
     "empty-handed)", 1.1, 1.6),
    ("target_z (world z height for the held object, or the EE if "
     "empty-handed)", 0.42, 0.72),
    ("target_yaw (wrist yaw in radians)", -np.pi, np.pi),
]


class PyBulletBridgeGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the bridge environment."""

    env_cls: ClassVar[TypingType[PyBulletBridgeEnv]] = PyBulletBridgeEnv

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_bridge"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        del env_name, predicates, action_space  # unused

        pybullet_robot = shared_skill_robot(cls.env_cls)

        robot_type = types["robot"]
        block_type = types["block"]
        bottle_type = types["bottle"]

        config = cls._build_skill_config(pybullet_robot)

        # -- PickBlock -------------------------------------------------------
        def _get_block_grasp_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            del params, cfg
            _, blk = objects
            half = cls.env_cls._world_half_extents(state, blk)  # pylint: disable=protected-access
            # Grasp near the top of the block. At wrist yaw = block yaw
            # the fingers straddle the 5 cm width for either orientation
            # (a quarter-turn on a lying block would straddle its 10 cm
            # length, wider than the ~8 cm finger opening, so the
            # fingers press the top face and shove the block into the
            # table instead of wrapping it).
            return (state.get(blk, "x"), state.get(blk, "y"),
                    state.get(blk, "z") + half[2], state.get(blk, "yaw"))

        PickBlock = create_pick_skill(
            name="PickBlock",
            types=[robot_type, block_type],
            config=config,
            get_target_pose_fn=_get_block_grasp_pose,
            # Open-fingered approach avoids dragging the block before
            # the grasp; anchored lift avoids the unreachable chase of
            # the held block's xy near the reach limit (see bond).
            approach_open=True,
            anchor_lift=True,
            # Staging is dense (grid slots ~11 cm apart), so end the
            # pick 3 cm up: a 1 cm lift can leave the gripper grazing
            # a neighboring block, poisoning the next option's BiRRT
            # start config.
            lift_dz=0.03,
            # A grasp_z_offset near the pads' upper engagement edge
            # (>= ~1 cm on a standing leg) makes the closing fingers cam
            # over the block's top corners: held detection can still
            # latch a degenerate constraint, and the table then drags
            # the block out of it during the lift. Verifying the lift
            # fails such picks honestly instead of handing a
            # ghost-held block to the place.
            verify_lift=True,
        )

        # -- PickBottle ------------------------------------------------------
        def _get_bottle_grasp_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            del params, cfg
            _, bottle = objects
            return (state.get(bottle,
                              "x"), state.get(bottle,
                                              "y"), state.get(bottle, "z") +
                    cls.env_cls.bottle_half_extents[2],
                    state.get(bottle, "rot"))

        PickBottle = create_pick_skill(
            name="PickBottle",
            types=[robot_type, bottle_type],
            config=config,
            get_target_pose_fn=_get_bottle_grasp_pose,
            approach_open=True,
            anchor_lift=True,
            # The default 1 cm lift is within the move-to acceptance
            # radius, so the pick can end with the bottle still at
            # table height; grasp-constraint droop then models it in
            # table contact at the next option's planning start.
            lift_dz=0.03,
        )

        # -- Place (generic; geometry via params) ---------------------------
        # use_move_above keeps the carry at transport_z so a held block
        # (or a whole multi-body assembly) clears staged objects and
        # the standing legs.
        Place = create_place_skill(
            name="Place",
            types=[robot_type],
            config=config,
            use_move_above=True,
            param_defs=_BRIDGE_PLACE_PARAMS,
            # Land the HELD BLOCK (not the gripper) on the sampled
            # target, on all three axes: blocks staged near the reach
            # limit grasp with up to ~2 cm of EE-to-block IK residual.
            # Uncompensated xy transfers that error into the butt
            # joints and seat alignment, past this domain's tight
            # tolerances; uncompensated z drives a deep-grasped block
            # into the table at the descend goal (BiRRT rejects it
            # forever).
            compensate_held_offset=True,
            compensate_held_z=True,
            # Guarded release: after the (collision-checked) descent to
            # release_z, settle straight down to FIRST contact of the
            # held assembly before opening. Drop-settle scatter is what
            # flips this domain's tight tolerances (a butt joint's
            # contact window, a 2:1 leg's sub-mm topple threshold, the
            # seat's chaotic landing). 3 cm covers the largest sampler descend
            # clearance (the seat's 20 mm) with margin; table places
            # settle only their 2-3 mm.
            settle_to_contact_depth=0.03,
            # Optional sag-discharge preload before release (see the
            # factory docstring); 0 = first-touch settle, the current
            # default behavior.
            settle_preload_force=(CFG.skill_place_settle_preload_force
                                  if CFG.skill_place_settle_preload_force > 0
                                  else None),
            # Verified release: the settle stroke ends at FIRST contact,
            # and plant sag (position control under gravity + payload)
            # was measured walking that contact point ~15 mm toward the
            # robot base -- a systematic landing bias that bends every
            # butt row. Before opening,
            # require the held block within 4 mm of the commanded xy;
            # otherwise lift back to release_z and re-descend, aiming
            # upstream of the measured (repeatable) drift.
            verify_xy_tol=0.004,
        )

        # -- MoveTo (generic move-through-pose) ------------------------------
        # Approach at transport height, descend (validated IK), retreat.
        # The target is where the HELD OBJECT'S CENTER ends up (the EE
        # itself when empty-handed): all three axes are compensated by
        # the live EE-to-held offset, read from the state at execution.
        # The z compensation is what Place's release_z budget papers
        # over with a drop: the pick's IK z-residual makes any
        # planning-time EE-height estimate wrong by up to ~5 mm, which
        # put the sampled descend goal in shallow contact and burned
        # execution replans. The retreat keeps the next option's
        # motion-planning start config clear of any surface the target
        # pose grazes.
        def _move_to_pose(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            del cfg
            robot = objects[0]
            x, y = float(params[0]), float(params[1])
            z, yaw = float(params[2]), float(params[3])
            off_x, off_y, off_z = 0.0, 0.0, 0.0
            for obj in state:
                if obj == robot or \
                        "is_held" not in obj.type.feature_names:
                    continue
                if state.get(obj, "is_held") > 0.5:
                    off_x = state.get(robot, "x") - state.get(obj, "x")
                    off_y = state.get(robot, "y") - state.get(obj, "y")
                    off_z = state.get(robot, "z") - state.get(obj, "z")
                    break
            return x + off_x, y + off_y, z + off_z, yaw

        move_to_space, move_to_description = build_params_space(
            _BRIDGE_MOVE_TO_PARAMS)
        MoveTo = create_move_to_skill(
            name="MoveTo",
            types=[robot_type],
            params_space=move_to_space,
            config=config,
            get_target_pose_fn=_move_to_pose,
            params_description=move_to_description,
            use_move_above=True,
            retreat=True,
            validate_ik=True,
            base_mode="home",
            # Hold at the reached target before retreating, so the tip
            # spends a stable interval at the target pose instead of a
            # single drive-by step between approach and retreat.
            dwell_steps=cls.env_cls.glue_dab_dwell_steps(),
        )

        return {
            PickBlock,
            PickBottle,
            Place,
            MoveTo,
            create_wait_option("Wait", config, robot_type),
        }

    @classmethod
    def _build_skill_config(
            cls, pybullet_robot: SingleArmPyBulletRobot) -> SkillConfig:
        simulator = shared_skill_simulator(cls.env_cls) \
            if CFG.skill_phase_use_motion_planning else None
        env_cls = cls.env_cls
        return SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=env_cls._fingers_state_to_joint,  # pylint: disable=protected-access
            ik_validate=CFG.pybullet_ik_validate,
            robot_init_tilt=env_cls.robot_init_tilt,
            robot_init_wrist=env_cls.robot_init_wrist,
            robot_home_pos=(env_cls.robot_init_x, env_cls.robot_init_y,
                            env_cls.robot_init_z),
            transport_z=env_cls.transport_z,
            simulator=simulator,
            # The carried block lags the end effector's mid-path swings
            # by up to centimetres, and the standing legs (2:1 aspect)
            # topple from a fraction-of-a-mm graze, so plans must keep
            # a real berth between the carried block and bodies the
            # path only passes by. Bodies within this clearance of the
            # held object at a path ENDPOINT (butt-joint neighbors,
            # seat legs, glue targets) are exempted by the planner, so
            # deliberately tight placements stay plannable.
            held_bystander_clearance=0.01,
        )
