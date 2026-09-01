"""Composed PyBullet domino environment.

This module provides the main environment class that composes multiple
components (dominoes, fans, balls, etc.) into a single environment.
"""

import logging
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_domino.cascade_certificate import StepOption, \
    check_cascade_legitimacy, count_movable_blocks_used
from predicators.envs.pybullet_domino.components.ball_component import \
    BallComponent
from predicators.envs.pybullet_domino.components.base_component import \
    DominoEnvComponent
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.envs.pybullet_domino.components.goal_region_component \
    import GoalRegionComponent
from predicators.envs.pybullet_domino.components.fan_component import \
    FanComponent
from predicators.envs.pybullet_domino.components.grid_component import \
    GridComponent
from predicators.envs.pybullet_domino.components.ramp_component import \
    RampComponent
from predicators.envs.pybullet_domino.components.stairs_component import \
    StairsComponent
# pylint: disable-next=line-too-long
from predicators.envs.pybullet_domino.task_generators.domino_task_generator import \
    DominoTaskGenerator
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import DECLARE_FINISHED_KEY, Action, \
    EnvironmentTask, GroundAtom, Object, ParameterizedOption, Predicate, \
    State, TaskEvaluator, Type


class DominoEvaluator(TaskEvaluator):
    """Task evaluator for min-block / system-ID tasks.

    ``terminated`` is the inherited goal-atom check (the target
    toppled, however that happened). The success bonus is gated on
    cascade legitimacy, and each movable (blue) domino the cascade
    consumes - toppled, or shoved off its stand as a slide-relay -
    costs ``CFG.domino_block_cost`` reward, so an over-built
    (denser-than-needed) chain succeeds at lower reward while an
    under-built one fails to topple the target. The oracle K* (the
    searched minimum blues at the true friction) deliberately does NOT
    live here: this object ships on the agent-facing ``Task``, so it
    must hold no oracle quantity - K* travels env-side via
    ``EnvironmentTask.offline_task_metrics``.

    The evaluator itself stores no env handle (which is what makes
    shipping it on the ``Task`` leak-free); the certificate's
    counterfactual push probe needs physics, so callers that own an
    env pass it per call as ``sim_env`` (``BaseEnv`` passes the true
    env, the sandbox's verdict path passes the agent's belief env) and
    the probe binding never outlives the call. Without a ``sim_env``
    the certificate runs its pure state/action rules only.
    """

    def __init__(self,
                 goal: Set[GroundAtom],
                 num_movables: Optional[int] = None,
                 trigger_option_name: str = "Push") -> None:
        """``num_movables`` is the number of movable (blue) dominoes staged in
        the task's scene, bounding the worst-case toppled-blue cost; it
        defaults to the min-block budget flag for the min-block / heavy task
        families, and the plain chain generator passes its actual count.

        ``trigger_option_name`` is the one option through which the
        robot may legitimately set the cascade going: "Push" where it
        shoves the green itself, "TurnFanOn" in a fan env, where it
        presses a switch and the wind does the toppling.
        """
        super().__init__(goal)
        self._trigger_option_name = trigger_option_name
        if num_movables is None:
            num_movables = CFG.domino_min_block_num_blues
        assert CFG.domino_block_cost * num_movables < 1.0, \
            "A legitimate success must outscore any failure."
        # Per-trajectory memo for _certify: reward/solved/_certify are
        # called back-to-back on the same states list and the probe is
        # a physics rollout, so recomputing per call would triple the
        # sim cost. Identity-keyed (never content-keyed): only reused
        # while the SAME states/labels objects with the SAME length are
        # scored.
        self._certify_memo: Optional[Tuple[Tuple[int, ...], Tuple[bool,
                                                                  str]]] = None

    def reward(self,
               states: Sequence[State],
               step_options: Optional[Sequence[StepOption]],
               sim_env: Optional[Any] = None) -> float:
        ok, _ = self._certify(states, step_options, sim_env=sim_env)
        bonus = float(self.terminated(states[-1]) and ok)
        return bonus - CFG.domino_block_cost * \
            count_movable_blocks_used(states)

    def _certify(self,
                 states: Sequence[State],
                 step_options: Optional[Sequence[StepOption]],
                 sim_env: Optional[Any] = None) -> Tuple[bool, str]:
        """Min-block episodes must be genuine start-block cascades.

        The final-state checks cannot see HOW the target fell; this
        rejects episodes where the robot toppled anything other than the
        green start block (via its Push), so place-knock / push-a-blue /
        flail-knock exploits earn no bonus even when the goal atoms
        hold. Consumed by ``BaseEnv.check_episode_trajectory`` /
        ``BaseEnv.evaluate_episode``. ``sim_env`` (a domino env, when
        the caller owns one) supplies the counterfactual push probe.
        """
        key = (id(states), len(states), id(states[-1]), id(step_options),
               id(sim_env))
        if self._certify_memo is not None and self._certify_memo[0] == key:
            return self._certify_memo[1]
        probe = getattr(sim_env, "run_counterfactual_cascade_probe", None)
        verdict = check_cascade_legitimacy(
            states,
            self.goal,
            step_options,
            probe=probe,
            trigger_option_name=self._trigger_option_name)
        self._certify_memo = (key, verdict)
        return verdict

    def offline_metrics(
            self, states: Sequence[State],
            step_options: Optional[Sequence[StepOption]]) -> Dict[str, float]:
        del step_options  # unused
        return {"k_used": float(count_movable_blocks_used(states))}

    def objective_description(self) -> str:
        c = CFG.domino_block_cost
        return ("The episode reward is EXACTLY:\n"
                f"  reward = (1.0 if certified success else 0.0) - {c} x "
                "(number of movable (blue) dominoes consumed)\n"
                "A blue is consumed when it ends the episode toppled, or "
                "was shoved off its stand while not held - whether or not "
                "you placed it, and regardless of success. Examples: a "
                f"certified success consuming 2 blues scores {1 - 2 * c:g}; "
                "a failed or rejected episode that consumed 1 blue scores "
                f"{-c:g}; no success and nothing consumed scores 0. There "
                "are no other reward terms.\n"
                "Certified success = the target domino topples via a "
                "legitimate cascade seeded by pushing the green start block. "
                "Only the blue dominoes may be rearranged: the green start "
                "block, the targets, and any gray blocks must stay "
                "untouched at their staged poses, upright and never held, "
                "until the green is pushed, and nothing may topple before "
                "that push. Only the green block may ever be pushed, so the "
                "cascade must bridge the gap with blue dominoes. Legitimacy "
                "is verified by re-simulating the push from the pre-push "
                "scene with the same Push skill but only the fingertips "
                "able to touch anything (the arm's body is intangible): the "
                "layout you built must cascade to the goal under the legal "
                "fingertip push alone - topples that needed the arm's body "
                "earn nothing. Extra consumed blues cost reward but never "
                "invalidate a success, so a robust over-built cascade "
                "always outscores a failed minimal one.")


class PyBulletDominoComposedEnv(PyBulletEnv):
    """A PyBullet domino environment composed of modular components.

    This environment supports:
    - Domino blocks that can topple through collisions
    - Fans that blow wind (optional)
    - Balls that can be moved by wind and collisions (optional)
    - Additional components can be added via the component system

    Components are initialized and composed at construction time.
    """

    # =========================================================================
    # TABLE / WORKSPACE CONFIGURATION
    # =========================================================================
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = tuple(
        p.getQuaternionFromEuler([0., 0., np.pi / 2]))
    table_width: ClassVar[float] = 1.0

    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.95

    # =========================================================================
    # ROBOT CONFIGURATION
    # =========================================================================
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub
    robot_base_pos: ClassVar[Optional[Tuple[float, float,
                                            float]]] = (0.75, 0.72, 0.0)
    robot_base_orn: ClassVar[Optional[Tuple[float, float, float, float]]] = \
        tuple(p.getQuaternionFromEuler([0.0, 0.0, np.pi / 2]))
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2

    # =========================================================================
    # CAMERA CONFIGURATION
    # =========================================================================
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = -70
    _camera_pitch: ClassVar[float] = -40
    _camera_target: ClassVar[Pose3D] = (0.75, 1.25, 0.42)

    # =========================================================================
    # DOMINO CONFIGURATION
    # =========================================================================
    # Domino shape properties
    domino_width: ClassVar[float] = 0.07
    domino_depth: ClassVar[float] = 0.015
    domino_height: ClassVar[float] = 0.15
    domino_mass: ClassVar[float] = 0.1
    domino_friction: ClassVar[float] = 0.5
    pos_gap: ClassVar[float] = 0.098  # domino_width * 1.4, computed value

    # Type definitions
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"],
                       angular_features=["roll", "tilt", "wrist"])
    _out_of_view_xy: ClassVar[Sequence[float]] = [10.0, 10.0]

    def __init__(self,
                 components: List[DominoEnvComponent],
                 use_gui: bool = False,
                 **kwargs: Any) -> None:
        """Initialize the composed domino environment.

        Args:
            components: List of components to include in the environment.
            use_gui: Whether to use PyBullet GUI.
        """
        self._components = components

        # Create robot object
        self._robot = Object("robot", self._robot_type)

        # Find specific component types for convenience
        # (must be done before _create_robot_predicates)
        self._domino_component: Optional[DominoComponent] = None
        self._fan_component: Optional[FanComponent] = None
        self._ball_component: Optional[BallComponent] = None

        for comp in components:
            if isinstance(comp, DominoComponent):
                self._domino_component = comp
            elif isinstance(comp, FanComponent):
                self._fan_component = comp
            elif isinstance(comp, BallComponent):
                self._ball_component = comp

        # Create predicates for robot (HandEmpty, Holding)
        self._create_robot_predicates()

        # Wire up fan -> ball wind connection if both present
        # (done after PyBullet init in _store_pybullet_bodies)

        super().__init__(use_gui, **kwargs)

        # Apply the configured domino friction to the live bodies. Two roles,
        # distinguished by how this instance was constructed:
        #   * eval/"real" env (skip_residual_dynamics=False, e.g. main.py) ->
        #     CFG.domino_true_friction;
        #   * planning base sim (skip_residual_dynamics=True — the approaches'
        #     base envs / option models, the same flag that already denies
        #     planners the ground-truth delayed dynamics) ->
        #     CFG.domino_planning_friction when set (else true friction).
        # Setting planning friction above true friction makes an uncalibrated
        # planner over-estimate topple reach (min-block / system-ID
        # experiments). Only applied when it differs from the built-in
        # default, so existing runs are physically untouched; re-applied
        # automatically after every reset_state.
        friction = CFG.domino_true_friction
        if self._skip_domain_specific_dynamics and \
                CFG.domino_planning_friction is not None and \
                not CFG.agent_sim_learn_oracle_sim_params:
            # agent_sim_learn_oracle_sim_params grants the planner the
            # TRUE friction (oracle upper bound) while task generation keeps
            # using domino_planning_friction for the differentiation filter.
            friction = CFG.domino_planning_friction
        if self._domino_component is not None and abs(
                friction - self._domino_component.domino_friction) > 1e-9:
            self.set_domino_physical_params(lateral_friction=friction)
        # Heavy-block tasks: planning sims BELIEVE the heavy gray blocks
        # are ordinary dominoes (normal mass), so their rollouts propagate
        # a chain straight through one. The eval env (and the oracle-
        # params planner) keeps the true heavy mass, asserted at reset.
        if CFG.domino_heavy_block_tasks \
                and self._domino_component is not None \
                and self._skip_domain_specific_dynamics \
                and not CFG.agent_sim_learn_oracle_sim_params:
            self.set_domino_physical_params(block_mass=self.domino_mass)
        # Snapshot the believed baseline AFTER the role adjustments above:
        # ``get_physical_param_info`` reports these values as the defaults,
        # and the sysID revert path restores dropped params to them (the
        # instance attrs alone miss init-time overrides such as a planning
        # friction that differs from the built-in).
        self._physical_param_baseline: Dict[str, float] = (
            self._domino_component.physical_param_override
            if self._domino_component is not None else {})
        # Dedicated world for the certificate's counterfactual push probe
        # (see run_counterfactual_cascade_probe); created on first use.
        self._cascade_probe_env: Optional[PyBulletDominoComposedEnv] = None
        # The real Push skill the probe replays; resolved lazily from
        # the ground-truth options on first probe.
        self._probe_push_option: Optional[ParameterizedOption] = None

    def _create_robot_predicates(self) -> None:
        """Create robot-specific predicates."""
        if self._domino_component is not None:
            domino_type = self._domino_component.domino_type
            self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                        self._HandEmpty_holds)
            self._Holding: Optional[Predicate] = Predicate(
                "Holding", [self._robot_type, domino_type],
                self._Holding_holds)
        else:
            # Create dummy predicates if no domino component
            self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                        lambda s, o: True)
            self._Holding = None

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_composed"

    # =========================================================================
    # PROPERTIES (Types, Predicates, etc.)
    # =========================================================================

    @property
    def types(self) -> Set[Type]:
        """Return all types from all components plus robot type."""
        all_types = {self._robot_type}
        for comp in self._components:
            all_types |= comp.get_types()
        return all_types

    @property
    def predicates(self) -> Set[Predicate]:
        """Return all predicates from all components plus robot predicates."""
        all_preds = {self._HandEmpty}
        if self._Holding is not None:
            all_preds.add(self._Holding)
        for comp in self._components:
            all_preds |= comp.get_predicates()
        if self._ball_component is not None:
            all_preds.add(self._ball_component.BallAtTarget)
        return all_preds

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Return goal predicates from all components."""
        goal_preds: Set[Predicate] = set()
        for comp in self._components:
            goal_preds |= comp.get_goal_predicates()
        if self._ball_component is not None:
            goal_preds.add(self._ball_component.BallAtTarget)
        return goal_preds

    # =========================================================================
    # PYBULLET INITIALIZATION
    # =========================================================================

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Initialize PyBullet simulation.

        Note: Component initialization happens in instance method since
        components are instance-specific.
        """
        # Reuse the base setup (connection, plane + studio floor, robot,
        # gravity, backdrop walls), then add this env's two tables. The tables
        # are textured centrally by _apply_studio_table_textures.
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # Two tables side by side for extra workspace.
        bodies["table_id"] = create_object(asset_path="urdf/table.urdf",
                                           position=cls.table_pos,
                                           orientation=cls.table_orn,
                                           scale=1.0,
                                           use_fixed_base=True,
                                           physics_client_id=physics_client_id)
        bodies["table_id2"] = create_object(
            asset_path="urdf/table.urdf",
            position=(cls.table_pos[0], cls.table_pos[1] + cls.table_width / 2,
                      cls.table_pos[2]),
            orientation=cls.table_orn,
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id)
        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Initialize and store PyBullet bodies for all components."""
        self._table_ids = [
            pybullet_bodies["table_id"], pybullet_bodies["table_id2"]
        ]
        # Initialize each component
        for comp in self._components:
            comp.set_physics_client_id(self._physics_client_id)
            comp_bodies = comp.initialize_pybullet(self._physics_client_id)
            comp.store_pybullet_bodies(comp_bodies)

        # Wire up fan -> ball connection if both present. Without a
        # ball the target is a domino, which changes per task, so it is
        # wired on every reset by _wire_wind_target instead.
        if self._fan_component is not None and self._ball_component is not None:
            self._fan_component.set_wind_target(self._ball_component.ball_id)

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return object IDs that can be held by robot."""
        ids = []
        for comp in self._components:
            ids.extend(comp.get_object_ids_for_held_check())
        return ids

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        """Extract state feature for an object."""
        # Try each component
        for comp in self._components:
            result = comp.extract_feature(obj, feature)
            if result is not None:
                return result

        # Grid helper objects (loc/angle/direction) are injected by the
        # ground-truth models during oracle / process planning and own no
        # live component here. GridComponent is the canonical home for the
        # grid logic, so reconstruct their features from their names. This
        # lets the _get_state round-trip in _set_state succeed even when the
        # env itself is built grid-free.
        result = GridComponent.reconstruct_feature_from_name(obj, feature)
        if result is not None:
            return result

        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _set_domain_specific_state(self, state: State) -> None:
        """Reset each component and update ball state reference."""
        for comp in self._components:
            comp.reset_state(state)

        if self._ball_component is not None:
            self._ball_component.set_current_state(state)
        self._wire_wind_target(state)

    def _wire_wind_target(self, state: State) -> None:
        """Point the fan at the body it is supposed to blow, for this task.

        With a ball in the scene the fan blows the ball, which is one
        body for the life of the env and is wired at startup. Without
        one it blows the START domino, and which body holds that role is
        a property of the task, not of the env: the roles are colours
        assigned per layout, so the green block is a different body id
        from one reset to the next. Re-resolving here - after the
        components have taken the new state - is what keeps the wind on
        the domino the task actually starts from.

        A task with no start block (none generated, or a scene where the
        chain begins elsewhere) leaves the previous target in place
        rather than silently blowing an arbitrary body.
        """
        if self._fan_component is None or self._ball_component is not None:
            return
        if self._domino_component is None:
            return
        for domino in self._domino_component.dominos:
            # The component allocates num_dominos_max bodies once and a
            # task instantiates a subset of them, so the tail of this
            # list is absent from the state; reading a colour off one
            # raises rather than returning False.
            if domino.id is None or domino not in state:
                continue
            # pylint: disable-next=protected-access
            if DominoComponent._StartBlock_holds(state, [domino]):
                # Push near the top of the domino so the wind tips it
                # rather than sliding it: 0.4 of its height above the
                # origin is comfortably above the centre and still on
                # the body.
                # Aim the fan down the chain, not merely in its
                # direction. Wind force is computed from orientation
                # alone, so a misplaced fan still topples the block -
                # which let this env ship with its fan 0.34 m to the
                # side of the chain, on a rail whose centre sits outside
                # the domino workspace. Correct in the picture and in
                # the state an agent reads.
                lateral = float(state.get(domino, "y"))
                self._fan_component.set_lateral_alignment(lateral)
                self._fan_component.set_wind_target(
                    domino.id,
                    z_offset=0.4 * self._domino_component.domino_height,
                    stop_when_toppled=True,
                    force=CFG.domino_fan_wind_force)
                return
        logging.warning(
            "Fan env has no start (green) domino in this task; leaving the "
            "wind target unchanged.")

    def _domain_specific_step(self) -> None:
        """Run component physics updates (e.g., fan wind simulation)."""
        for comp in self._components:
            comp.step()

        # Update ball component's state reference
        if self._ball_component is not None:
            state = self._get_state()
            self._ball_component.set_current_state(state)

    def set_domino_physical_params(self, **params: Optional[float]) -> None:
        """Override this env instance's domino PyBullet dynamics params.

        Thin delegate to ``DominoComponent.set_physical_params``
        (accepts ``mass``, ``lateral_friction``, ``restitution``,
        ``rolling_friction``, ``spinning_friction``). Lets a caller run
        two env instances with divergent physics in one process — e.g. a
        miscalibrated planning sim vs. the "real" env — for system-ID /
        sim-vs-real experiments, without touching the shared ClassVars.
        No-op if there is no domino component.
        """
        if self._domino_component is not None:
            self._domino_component.set_physical_params(**params)

    def get_physical_param_info(self) -> Dict[str, Dict[str, Any]]:
        """Tunable domino dynamics params (see BaseEnv docstring).

        These are the parameters ``set_domino_physical_params`` accepts;
        defaults mirror what ``create_domino_block`` bakes into fresh
        bodies. All are global scalars shared by every (identical)
        domino body.
        """
        comp = self._domino_component
        if comp is None:
            return {}
        # Defaults report the believed BASELINE of this instance: the
        # post-init override snapshot when present (e.g. a planning
        # friction differing from the built-in), else the built-in value.
        # The sysID revert path restores dropped params to these defaults,
        # so they must be the values the env would have without any fit.
        baseline = getattr(self, "_physical_param_baseline", {})
        lateral_friction = baseline.get("lateral_friction",
                                        comp.domino_friction)
        # ``scale: "log"`` marks positive scale-like parameters whose
        # behavioral effect is multiplicative: the sysID fit runs in
        # log-space for them (geometric grid sweep, relative LM steps,
        # log-normal prior). A linear parameterization has almost no
        # resolution at the low end of a box spanning decades —
        # linspace(0.01, 2, 8) has no candidate between 0.01 and 0.29,
        # which is how run_20260706_171526 fit friction 0.0114 for a
        # true 0.1. Params whose lo is 0 (restitution,
        # rolling_friction) stay linear.
        info: Dict[str, Dict[str, Any]] = {
            "lateral_friction": {
                "default":
                lateral_friction,
                "lo":
                0.01,
                "hi":
                2.0,
                "scale":
                "log",
                "description":
                "Lateral (sliding) friction of each domino against the "
                "table and other dominoes (PyBullet lateralFriction); "
                "governs how far a toppling domino slides/rotates and "
                "whether a cascade propagates.",
            },
            "restitution": {
                "default":
                baseline.get("restitution", 0.02),
                "lo":
                0.0,
                "hi":
                0.9,
                "description":
                "Bounciness of domino-domino impacts (the table's "
                "restitution is 0, and PyBullet combines them "
                "multiplicatively, so this only manifests in "
                "domino-on-domino collisions).",
            },
            "mass": {
                "default":
                baseline.get("mass", comp.domino_mass),
                "lo":
                0.005,
                "hi":
                1.0,
                "scale":
                "log",
                "description":
                "Mass of each (non-glued) domino in kg. Largely scales "
                "out of the topple condition for identical dominoes.",
            },
            "rolling_friction": {
                "default":
                baseline.get("rolling_friction", 0.006),
                "lo":
                0.0,
                "hi":
                0.1,
                "description":
                "Rolling-friction coefficient; damps edge-rolling of a "
                "tipping domino.",
            },
            "spinning_friction": {
                "default":
                # Bodies are created with spinningFriction = the built-in
                # lateral value; a lateral_friction override does NOT
                # retouch it, so the baseline follows the ClassVar.
                baseline.get("spinning_friction", comp.domino_friction),
                "lo":
                0.01,
                "hi":
                2.0,
                "scale":
                "log",
                "description":
                "Spin (yaw) friction against the table; defaults to the "
                "lateral friction value at body creation.",
            },
        }
        # Gray ``block``-typed bodies form their own parameter class:
        # the ``block_*`` family applies to them only (and beats the
        # global param for those bodies). Descriptions are deliberately
        # neutral - whether blocks differ physically from dominoes is
        # for the fit to establish, not the registry to reveal.
        if comp.blocks:
            info["block_mass"] = {
                "default":
                baseline.get("block_mass", comp.domino_mass),
                "lo":
                0.005,
                "hi":
                2000.0,
                "scale":
                "log",
                "description":
                "Mass in kg of each block (the gray block type); applies "
                "to block bodies only, independently of the dominoes' "
                "``mass``.",
            }
            info["block_lateral_friction"] = {
                "default":
                baseline.get(
                    "block_lateral_friction",
                    baseline.get("lateral_friction", comp.domino_friction)),
                "lo":
                0.01,
                "hi":
                2.0,
                "scale":
                "log",
                "description":
                "Lateral (sliding) friction of each block (the gray "
                "block type) against the table and other bodies; applies "
                "to block bodies only.",
            }
        return info

    def apply_physical_param_overrides(self, params: Dict[str, float]) -> None:
        """Sticky in-place dynamics override (delegates to the domino
        component's ``set_physical_params``, which re-applies after every reset
        and body recreation)."""
        unknown = set(params) - set(self.get_physical_param_info())
        if unknown:
            raise ValueError(f"Unknown physical param(s) {sorted(unknown)}.")
        self.set_domino_physical_params(**params)

    def dispose(self) -> None:
        """Disconnect every client this instance owns.

        The counterfactual-probe world is a second full PyBullet client
        created lazily by :meth:`_get_cascade_probe_env`; disconnecting
        only ``_physics_client_id`` (what generic callers used to do)
        leaked it - ~150MB per fresh validation env, enough to freeze a
        16GB machine across parallel runs. Probe world first: the main
        client may already be dead (crash recovery), and ``super()``
        raising must not strand the probe.
        """
        if self._cascade_probe_env is not None:
            probe = self._cascade_probe_env
            self._cascade_probe_env = None
            probe.dispose()
        super().dispose()

    def _get_cascade_probe_env(self) -> "PyBulletDominoComposedEnv":
        """The dedicated probe world for the counterfactual push probe.

        A fresh instance of this env's own class (created once, then
        reused) so probing never contaminates this world with residual
        velocities, solver state, or the finger body. Same-class
        construction gives the probe world the same body pool in the
        same order, which is what lets it ``_set_state`` this env's
        states (their objects carry this world's pybullet ids - the
        same id-coincidence contract the option model's belief env
        already relies on). The live physics overrides are re-mirrored
        on every call because sysID artifacts move between episodes.
        """
        if self._cascade_probe_env is None:
            # Concrete env classes take (use_gui, **kwargs) and build
            # their own component list; only this abstract composed base
            # takes `components`, and it is never instantiated directly.
            # pylint: disable-next=no-value-for-parameter
            self._cascade_probe_env = type(self)(  # type: ignore[call-arg]
                use_gui=False,
                skip_residual_dynamics=self._skip_domain_specific_dynamics)
        probe_env = self._cascade_probe_env
        # pylint: disable-next=protected-access
        probe_component = probe_env._domino_component
        if self._domino_component is not None and probe_component is not None:
            override = self._domino_component.physical_param_override
            if override:
                probe_component.set_physical_params(**override)
        return probe_env

    def run_counterfactual_cascade_probe(
            self,
            pre_push_state: State,
            greens: Sequence[Object],
            goal: Set[GroundAtom],
            push_params: Optional[Tuple[float,
                                        ...]] = None) -> Tuple[bool, str]:
        """Counterfactual clean-push probe for the cascade certificate.

        From ``pre_push_state``, re-runs the REAL Push skill on each
        green (in the given order) in the dedicated probe world - this
        env's physics, the episode's own ``push_params`` when recorded,
        every robot link except the fingertips collision-masked - and
        reports whether the push cascades to the goal atoms. When this
        env carries a ``probe_process_model_factory`` (a sim-learning
        approach's belief env), the replay runs on the combined
        substrate (learned residual rules applied per step); a passing
        combined verdict is then double-checked base-only with the same
        attempt count, purely as a diagnostic of whether the rules were
        load-bearing. See
        ``cascade_probe`` for the fidelity contract and the rationale.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.envs.pybullet_domino import cascade_probe
        if self._probe_push_option is None:
            # Lazy: the env layer imports no ground-truth models at
            # module scope; the probe needs the real Push skill (its
            # controller IS the fidelity contract), and get_gt_options
            # caches its skill simulator, so this costs one lookup.
            # pylint: disable-next=import-outside-toplevel
            from predicators.ground_truth_models import get_gt_options
            self._probe_push_option = next(
                opt for opt in get_gt_options(self.get_name())
                if opt.name == "Push")
        factory = self.probe_process_model_factory
        ok, detail = cascade_probe.run_counterfactual_push_probe(
            self._get_cascade_probe_env(),
            pre_push_state,
            greens,
            goal,
            push_params,
            push_option=self._probe_push_option,
            process_model_factory=factory)
        if factory is not None and ok:
            # Load-bearing-rules diagnostic: a combined-substrate pass
            # whose base-only replay fails means the learned rules
            # carried the verdict - fine when they model a process the
            # base sim lacks, a calibration smell when they compensate
            # for undeclared physical params. The base-only replay MUST
            # use the same attempt count as the combined probe: replay
            # attempts are nondeterministic (residual solver state in
            # the reused probe world), so an any-of-N pass compared
            # against a 1-of-1 pass flags knife-edge layouts as
            # "load-bearing" even when the rules are a physical no-op.
            # The note rides the harness-internal ``reason`` channel
            # only, and is evidence, not proof - a sufficiently
            # knife-edge layout can still fail all base-only attempts
            # by chance.
            base_ok, _ = cascade_probe.run_counterfactual_push_probe(
                self._get_cascade_probe_env(),
                pre_push_state,
                greens,
                goal,
                push_params,
                push_option=self._probe_push_option)
            if not base_ok:
                note = ("the learned residual rules appear load-bearing "
                        "for this verdict (no base-sim-only replay "
                        "attempt cascades)")
                logging.info("[cascade probe] %s", note)
                detail = f"{detail}; {note}"
        return ok, detail

    # =========================================================================
    # PREDICATE HOLD FUNCTIONS
    # =========================================================================

    def _HandEmpty_holds(self, state: State,
                         _objects: Sequence[Object]) -> bool:
        """Check if robot hand is empty."""
        if self._domino_component is None:
            return True
        dominos = state.get_objects(self._domino_component.domino_type)
        for domino in dominos:
            if state.get(domino, "is_held"):
                return False
        return True

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if robot is holding a specific domino."""
        _, domino = objects
        return state.get(domino, "is_held") > 0.5

    # =========================================================================
    # COMPONENT CONSTRUCTION HELPERS
    # =========================================================================

    @classmethod
    def _default_workspace_bounds(cls) -> Dict[str, float]:
        """Workspace bounds shared by all concrete domino environments."""
        return {
            "x_lb": cls.x_lb,
            "x_ub": cls.x_ub,
            "y_lb": cls.y_lb,
            "y_ub": cls.y_ub,
            "z_lb": cls.z_lb,
            "z_ub": cls.z_ub,
        }

    @classmethod
    def _make_domino_component(
            cls, workspace_bounds: Dict[str, float]) -> DominoComponent:
        """Build a domino component sized to the configured task ranges."""
        max_dominos = max(max(CFG.domino_train_num_dominos),
                          max(CFG.domino_test_num_dominos))
        max_targets = max(max(CFG.domino_train_num_targets),
                          max(CFG.domino_test_num_targets))
        max_pivots = max(max(CFG.domino_train_num_pivots),
                         max(CFG.domino_test_num_pivots))
        if CFG.domino_min_block_tasks or CFG.domino_heavy_block_tasks:
            # Need slots for the start + target + all staged blues, plus
            # one more for the heavy gray obstacle in heavy-block mode.
            extra = 3 if CFG.domino_heavy_block_tasks else 2
            max_dominos = max(max_dominos,
                              CFG.domino_min_block_num_blues + extra)
        return DominoComponent(num_dominos_max=max_dominos,
                               num_targets_max=max_targets,
                               num_pivots_max=max_pivots,
                               workspace_bounds=workspace_bounds)

    # =========================================================================
    # TASK GENERATION
    # =========================================================================

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        """Generate training tasks."""
        return self._make_tasks(
            num_tasks=CFG.num_train_tasks,
            possible_num_dominos=CFG.domino_train_num_dominos,
            possible_num_targets=CFG.domino_train_num_targets,
            possible_num_pivots=CFG.domino_train_num_pivots,
            turn_ratio=CFG.domino_train_turn_ratio,
            rng=self._train_rng,
            cache_tag="train")

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        """Generate test tasks."""
        return self._make_tasks(
            num_tasks=CFG.num_test_tasks,
            possible_num_dominos=CFG.domino_test_num_dominos,
            possible_num_targets=CFG.domino_test_num_targets,
            possible_num_pivots=CFG.domino_test_num_pivots,
            turn_ratio=CFG.domino_test_turn_ratio,
            rng=self._test_rng,
            cache_tag="test")

    def robot_init_state_dict(self) -> Dict[str, float]:
        """The robot's initial feature dict, shared by every task scene (the
        task generators stage the robot at this pose)."""
        return {
            "x": self.robot_init_x,
            "y": self.robot_init_y,
            "z": self.robot_init_z,
            "fingers": self.open_fingers,
            "roll": self.robot_init_roll,
            "tilt": self.robot_init_tilt,
            "wrist": self.robot_init_wrist,
        }

    def _make_tasks(self,
                    num_tasks: int,
                    possible_num_dominos: List[int],
                    possible_num_targets: List[int],
                    possible_num_pivots: List[int],
                    turn_ratio: float,
                    rng: np.random.Generator,
                    log_debug: bool = False,
                    cache_tag: str = "") -> List[EnvironmentTask]:
        """Generate tasks using task generator."""
        if self._domino_component is None:
            raise ValueError("Cannot generate tasks without domino component")

        # Create task generator
        robot_init_state = self.robot_init_state_dict()

        # Collect additional components for init dict (all except domino)
        additional_components = []
        for comp in self._components:
            if comp is not self._domino_component:
                additional_components.append(comp)

        generator = DominoTaskGenerator(
            domino_component=self._domino_component,
            robot=self._robot,
            robot_init_state=robot_init_state,
            additional_components=additional_components)

        # If ball component is present, place dominoes in upper half
        # to leave space for ball in lower half
        domino_in_upper_half = self._ball_component is not None

        def _generate_batch(n: int) -> List[EnvironmentTask]:
            return generator.generate_tasks(
                num_tasks=n,
                rng=rng,
                log_debug=log_debug,
                possible_num_dominos=possible_num_dominos,
                possible_num_targets=possible_num_targets,
                possible_num_pivots=possible_num_pivots,
                domino_in_upper_half=domino_in_upper_half,
                turn_ratio=turn_ratio)

        # Non-min-block mode: single generation pass, unchanged behaviour.
        if not (CFG.domino_min_block_tasks or CFG.domino_heavy_block_tasks):
            return self._add_pybullet_state_to_tasks(
                _generate_batch(num_tasks))

        # Min-block / system-ID mode (incl. the heavy-block task type):
        # the whole pipeline (quota loop, K* searches, differentiation
        # filters, disk cache) lives in
        # task_generators.min_block_generation. Imported lazily: that
        # module constructs DominoEvaluator from this one.
        # pylint: disable-next=import-outside-toplevel,line-too-long
        from predicators.envs.pybullet_domino.task_generators.min_block_generation import \
            make_min_block_tasks
        return make_min_block_tasks(self, generator, _generate_batch,
                                    num_tasks, rng, cache_tag, turn_ratio)


# =============================================================================
# BACKWARD-COMPATIBLE ENVIRONMENT CLASSES
# =============================================================================


class PyBulletDominoEnv(PyBulletDominoComposedEnv):
    """Backward-compatible domino environment class."""

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        bounds = self._default_workspace_bounds()
        domino_comp = self._make_domino_component(bounds)
        super().__init__(components=[domino_comp], use_gui=use_gui, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino"


# Where the domino-fan env's switch sits: in front of the robot, past
# the far end of the chain in x and a little NEARER in y, so the arm
# approaches it from outside the chain instead of over it. At
# (1.05, 1.30) the press reached across a finished bridge - the switch
# was 0.22 m from the target at the same y - and disturbed it.
#
# The inherited formula put it 0.41 m from the robot's base, folded in
# against the arm, and the press never completed - IK solves there, but
# the joint-limited arm lands centimetres short and the skill waits
# forever for an exact arrival. A sweep over candidate positions found
# the boundary sharp: 0.41 m never terminates, while every position
# from 0.59 m out presses in ~30 steps. The fan env, where this has
# always worked, presses at 0.73 m.
DOMINO_FAN_SWITCH_XY = (1.10, 1.20)


class PyBulletDominoFanBallEnv(PyBulletDominoComposedEnv):
    """Domino + fan + ball: the fan blows the BALL, which knocks dominoes.

    Formerly ``pybullet_domino_fan``. Renamed when the ball-free variant
    below took that name, because which body the wind pushes is the
    whole difference between the two tasks and the old name did not say.
    """

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        bounds = self._default_workspace_bounds()
        domino_comp = self._make_domino_component(bounds)
        fan_comp = FanComponent(workspace_bounds=bounds,
                                table_height=self.table_height,
                                table_width=self.table_width)
        ball_comp = BallComponent(workspace_bounds=bounds,
                                  table_height=self.table_height)
        super().__init__(components=[domino_comp, fan_comp, ball_comp],
                         use_gui=use_gui,
                         **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_fan_ball"


class PyBulletDominoFanEnv(PyBulletDominoComposedEnv):
    """Domino + fan, no ball: the fan blows the START DOMINO directly.

    The task is to arrange the dominoes so that switching the fan on
    topples the chain all the way to the target - the robot never
    pushes anything itself, so the whole plan is in the layout.

    Wind targets the green start block rather than the ball, resolved on
    every reset (see ``_wire_wind_target``): which body carries the
    start role changes from task to task, so it cannot be wired once at
    startup the way the ball's single body can.
    """

    # A deeper workspace than the other domino envs, and the chain
    # layout is why. The staging grid steps y by 1.5 * domino_width
    # inside margins of 1.5 and 3 widths, so the inherited y range
    # (1.1-1.6) leaves a domino band 0.185 m deep - room for exactly ONE
    # staging row, at y = 1.275. A wind-aligned chain runs along x at
    # y = 1.283, so that single row lands ON the chain: every blue is
    # parked 8 mm from the line it has to be built into, and Place
    # cannot reach a bridge slot without the gripper fouling a
    # neighbour ("BiRRT collision: target configuration in collision").
    # 1.70 gives a 0.285 m band, hence rows at 1.275 and 1.38 - one for
    # the chain, one to park in. The far row sits 0.66 m from the
    # robot's base, well inside the reach the switch sweep measured.
    y_ub: ClassVar[float] = 1.70

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        bounds = self._default_workspace_bounds()
        domino_comp = self._make_domino_component(bounds)
        # One fan, one switch. The four-sided layout is the ball task's,
        # where the ball must be blown any of four ways across a grid; a
        # domino chain runs one way, and the aligned generator lays it
        # along side 0. The other three fans would be distractors the
        # planner still has to ground, and being opposed they cancel.
        fan_comp = self._make_fan_component(bounds)
        components = [domino_comp, fan_comp]
        components += self._extra_components(bounds, domino_comp)
        super().__init__(components=components, use_gui=use_gui, **kwargs)

    def _extra_components(self, bounds: Dict[str, float],
                          domino_comp: DominoComponent) -> List[Any]:
        """Components beyond the dominoes and the fan. None by default."""
        del bounds, domino_comp
        return []

    def _make_fan_component(self, bounds: Dict[str, float]) -> FanComponent:
        """The fan bank for this env. Overridden where the switch is
        not something the robot can reach."""
        return FanComponent(workspace_bounds=bounds,
                            table_height=self.table_height,
                            table_width=self.table_width,
                            num_sides=CFG.domino_fan_num_sides,
                            fans_per_side=1,
                            switch_xy=DOMINO_FAN_SWITCH_XY)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_fan"


class PyBulletDominoDeclareEnv(PyBulletDominoFanEnv):
    """The fan env with the button removed: the robot DECLARES finished.

    Same scene and same physics as ``pybullet_domino_fan`` -- arrange
    the blues so the wind carries a cascade from the green start block
    to the purple target -- with one thing changed: there is no switch
    to press. The robot runs a ``DeclareFinished`` skill, and the fan
    comes on.

    Two reasons the change is worth its own env rather than a flag.

    **It is the version that survives contact with real hardware.** A
    physical button has to be reachable from every staging pose, the
    arm has to approach it without sweeping through the chain it just
    built, and a missed press looks exactly like a press that did not
    take. A declaration has none of that.

    **It makes the causal question clean.** With a button, "what starts
    the wind" has a mechanical answer an agent can stumble into: the
    gripper touched a thing. Here nothing is touched. An agent that
    works out that the wind follows its declaration has found a
    relation that is causal and nothing else -- which is precisely what
    the predicate-invention rung is meant to be testing.

    The switch body still exists and still stores the bit (a fan's
    ``is_on`` is read off its joint), but it is parked two metres
    outside the workspace, so a press is not merely unnecessary here:
    it is impossible.
    """

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        super().__init__(use_gui=use_gui, **kwargs)
        # Latched by the declaration, cleared on reset. Kept on the env
        # rather than read from the switch each step because the
        # question it answers is "has the robot declared yet", which is
        # about the episode, not about the bodies.
        self._declared: bool = False

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_declare"

    def _make_fan_component(self, bounds: Dict[str, float]) -> FanComponent:
        """A fan whose switch nothing can reach."""
        return FanComponent(workspace_bounds=bounds,
                            table_height=self.table_height,
                            table_width=self.table_width,
                            num_sides=CFG.domino_fan_num_sides,
                            fans_per_side=1,
                            switch_xy=DOMINO_FAN_SWITCH_XY,
                            switch_reachable=False)

    def _set_domain_specific_state(self, state: State) -> None:
        super()._set_domain_specific_state(state)
        # A reset is a new episode: the declaration does not carry over,
        # or the second task of a run would begin with the wind already
        # blowing on a chain nobody has built yet.
        self._declared = bool(self._fan_component is not None
                              and self._fan_component.any_fan_on())

    def _domain_specific_step(self) -> None:
        # The declaration, before the wind: the flag it sets is what
        # makes any wind happen at all this step.
        action = self._last_action
        info = getattr(action, "extra_info", None)
        if isinstance(info, dict) and info.get(DECLARE_FINISHED_KEY):
            self._declared = True
            if self._fan_component is not None:
                self._fan_component.set_fans_on(True)
        super()._domain_specific_step()


class PyBulletDominoBlowEnv(PyBulletDominoDeclareEnv):
    """Place a block so the wind carries it INTO a goal region.

    The scene reads left to right: fan, goal patch, staged block. The
    robot picks the block up, puts it down somewhere between the fan
    and the patch, and declares finished; the fan then blows for a
    bounded number of steps and the block slides. It scores if the
    block comes to rest inside the patch.

    This task exists because of what pybullet_domino_fan could NOT
    teach. There the wind tips a standing domino in about two steps,
    so every force above the tipping threshold produces the same
    observation and ``wind_force`` is unfittable - 1.5 N and 2.0 N give
    identical trajectories (measured by
    scripts/domino_debug/probe_wind_identifiability.py). Here the wind
    pushes through the block's CENTRE OF MASS, so it slides rather than
    tips and the distance it travels is a continuous, monotone function
    of the force. That is the same reason pybullet_fan can fit this
    parameter and the domino env cannot: what the wind is pushing
    decides whether its strength leaves a trace.

    And the goal is a bounded REGION, not a point, which is what stops
    the degenerate policy. Placing the block as close to the fan as
    possible - the obvious way to avoid learning anything - overshoots
    the far edge. Placing it safely far never arrives. Only a band of
    placements works, and its position depends on how hard this fan
    blows, so the robot cannot reach the goal without having learned
    that.
    """

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._goal_region_component: Optional[GoalRegionComponent] = None
        super().__init__(use_gui=use_gui, **kwargs)
        self._wind_steps_left: int = 0

    def _extra_components(self, bounds: Dict[str, float],
                          domino_comp: DominoComponent) -> List[Any]:
        """The patch the block has to end up in."""
        self._goal_region_component = GoalRegionComponent(
            workspace_bounds=bounds,
            table_height=self.table_height,
            domino_type=domino_comp.domino_type)
        return [self._goal_region_component]

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_blow"

    def _set_domain_specific_state(self, state: State) -> None:
        super()._set_domain_specific_state(state)
        # A fresh episode gets a fresh gust budget, and the wind aims at
        # the movable block through its centre so the block slides.
        self._wind_steps_left = CFG.domino_blow_wind_steps
        self._wire_blow_target(state)

    def _wire_blow_target(self, state: State) -> None:
        """Aim the fan at the block it is supposed to move.

        z_offset 0.0 is the whole difference from the cascade envs: a
        force through the centre of mass is pure translation. In
        pybullet_domino_fan that same setting was a BUG - the start
        block slid a metre and a half into the target without ever
        tipping, "solving" a bridge task with no bridge - and here it
        is the mechanism.
        """
        if self._fan_component is None or self._domino_component is None:
            return
        movable = [
            obj for obj in state
            if obj.type == self._domino_component.domino_type
            # pylint: disable-next=protected-access
            and self._domino_component._MovableBlock_holds(state, [obj])
        ]
        if not movable:
            return
        block = movable[0]
        self._fan_component.set_lateral_alignment(float(
            state.get(block, "y")))
        self._fan_component.set_wind_target(
            block.id,
            z_offset=0.0,
            stop_when_toppled=False,
            force=CFG.domino_blow_wind_force)

    def _domain_specific_step(self) -> None:
        super()._domain_specific_step()
        # The gust is finite. Without a budget the block is pushed until
        # the episode horizon and its resting place says nothing about
        # the force - every force large enough to move it at all ends up
        # against the far wall, which is the saturation this env was
        # built to avoid.
        if self._fan_component is None:
            return
        if self._fan_component.any_fan_on() and self._wind_steps_left > 0:
            self._wind_steps_left -= 1
            if self._wind_steps_left == 0:
                self._fan_component.set_fans_on(False)


class PyBulletDominoFanRampEnv(PyBulletDominoComposedEnv):
    """Domino + fan + ball + ramp environment class."""

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        bounds = self._default_workspace_bounds()
        domino_comp = self._make_domino_component(bounds)
        fan_comp = FanComponent(workspace_bounds=bounds,
                                table_height=self.table_height,
                                table_width=self.table_width)
        ball_comp = BallComponent(workspace_bounds=bounds,
                                  table_height=self.table_height)
        ramp_comp = RampComponent(workspace_bounds=bounds,
                                  table_height=self.table_height,
                                  max_ramps=5)
        super().__init__(
            components=[domino_comp, fan_comp, ball_comp, ramp_comp],
            use_gui=use_gui,
            **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_fan_ramp"


class PyBulletDominoFanRampStairsEnv(PyBulletDominoComposedEnv):
    """Domino + fan + ball + ramp + stairs environment class."""

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        bounds = self._default_workspace_bounds()
        domino_comp = self._make_domino_component(bounds)
        fan_comp = FanComponent(workspace_bounds=bounds,
                                table_height=self.table_height,
                                table_width=self.table_width)
        ball_comp = BallComponent(workspace_bounds=bounds,
                                  table_height=self.table_height)
        ramp_comp = RampComponent(workspace_bounds=bounds,
                                  table_height=self.table_height,
                                  max_ramps=5)
        # Stairs component needs reference to domino type for positioning
        stairs_comp = StairsComponent(workspace_bounds=bounds,
                                      table_height=self.table_height,
                                      domino_type=domino_comp.domino_type,
                                      enabled=True)
        super().__init__(components=[
            domino_comp, fan_comp, ball_comp, ramp_comp, stairs_comp
        ],
                         use_gui=use_gui,
                         **kwargs)

        # Store reference to stairs component
        self._stairs_component = stairs_comp

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_fan_ramp_stairs"


if __name__ == "__main__":
    import sys
    import time

    from predicators import utils

    # Choose which environment to test
    # Options: "domino", "domino_fan", "domino_fan_ball",
    # "domino_fan_ramp",
    # "domino_fan_ramp_stairs"
    # Change this to test different environments
    test_env = "domino_fan_ramp_stairs"
    test_env = "domino"
    if len(sys.argv) > 1:
        test_env = sys.argv[1]

    CFG.domino_min_block_tasks = False
    CFG.domino_true_friction = 0.1
    CFG.domino_min_block_span_lo = 0.13
    CFG.domino_min_block_span_hi = 0.30
    CFG.domino_min_block_num_blues = 4

    # Configure environment
    CFG.seed = 1
    CFG.num_train_tasks = 0
    CFG.num_test_tasks = 10

    # Domino configuration
    CFG.domino_initialize_at_finished_state = True
    CFG.domino_use_domino_blocks_as_target = True
    CFG.domino_has_glued_dominos = False
    CFG.domino_test_num_dominos = [3]
    CFG.domino_test_num_targets = [1, 2]
    CFG.domino_test_num_pivots = [0]

    # Fan/ball configuration
    CFG.domino_fan_ball_position_tolerance = 0.04
    CFG.fan_known_controls_relation = True
    CFG.fan_fans_blow_opposite_direction = False

    # Create environment based on selection
    demo_env: PyBulletDominoComposedEnv
    if test_env == "domino":
        print("Creating PyBulletDominoEnv...")
        CFG.env = "pybullet_domino"
        demo_env = PyBulletDominoEnv(use_gui=True)
    elif test_env == "domino_fan":
        print("Creating PyBulletDominoFanEnv...")
        CFG.env = "pybullet_domino_fan"
        demo_env = PyBulletDominoFanEnv(use_gui=True)
    elif test_env == "domino_fan_ball":
        print("Creating PyBulletDominoFanBallEnv...")
        CFG.env = "pybullet_domino_fan_ball"
        demo_env = PyBulletDominoFanBallEnv(use_gui=True)
    elif test_env == "domino_fan_ramp":
        print("Creating PyBulletDominoFanRampEnv...")
        CFG.env = "pybullet_domino_fan_ramp"
        demo_env = PyBulletDominoFanRampEnv(use_gui=True)
    elif test_env == "domino_fan_ramp_stairs":
        print("Creating PyBulletDominoFanRampStairsEnv...")
        CFG.env = "pybullet_domino_fan_ramp_stairs"
        demo_env = PyBulletDominoFanRampStairsEnv(use_gui=True)
    else:
        raise ValueError(f"Unknown environment: {test_env}")

    # Generate test tasks
    print("Generating test tasks...")
    test_tasks = demo_env._generate_test_tasks()  # pylint: disable=protected-access

    print(f"\nGenerated {len(test_tasks)} tasks")
    print(f"Types: {[t.name for t in demo_env.types]}")
    print(f"Predicates: {[p.name for p in demo_env.predicates]}")

    # Test each task
    for i, task in enumerate(test_tasks):
        print(f"\n{'=' * 60}")
        print(f"Task {i + 1}")
        print(f"{'=' * 60}")

        # Reset to initial state
        demo_env._set_state(task.init)  # pylint: disable=protected-access

        print("\nGoal atoms:")
        for atom in task.goal:
            print(f"  {atom}")

        # Print the initial abstract atoms (what the agent sees).
        init_atoms = utils.abstract(task.init, demo_env.predicates)
        print("\nInitial atoms (abstract state seen by the agent):")
        for atom in sorted(init_atoms, key=str):
            print(f"  {atom}")

        # Print task pretty_str
        print("\n Initial state:")
        print(task.init.pretty_str())

        try:
            for step in range(50):
                # pylint: disable=protected-access
                cur_action = Action(
                    np.array(demo_env._pybullet_robot.initial_joint_positions))
                cur_state = demo_env.step(cur_action)

                if all(atom.holds(cur_state) for atom in task.goal):
                    print(f"Goal reached at step {step}!")
                    time.sleep(2)
                    break

                time.sleep(0.02)
        except KeyboardInterrupt:
            continue

    print("\nDone!")
