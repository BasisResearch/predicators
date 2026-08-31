"""Task generator for domino-based tasks."""

from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import numpy as np

from predicators import utils
from predicators.envs.pybullet_domino import geometry
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent, PlacementResult
from predicators.envs.pybullet_domino.task_generators import goal_text
from predicators.envs.pybullet_domino.task_generators.base_generator import \
    TaskGenerator
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, GroundAtom, Object


def _dist_to_segment(pt: Tuple[float, float], a: Tuple[float, float],
                     b: Tuple[float, float]) -> float:
    """Perpendicular distance from ``pt`` to segment ``a``-``b``."""
    px, py = pt
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 0.0:
        return float(np.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return float(np.hypot(px - (ax + t * dx), py - (ay + t * dy)))


class DominoTaskGenerator(TaskGenerator):
    """Generates tasks involving domino sequences.

    Creates tasks where dominoes must be arranged to topple targets.
    Supports pivots for direction changes.
    """

    def __init__(self,
                 domino_component: DominoComponent,
                 robot: Object,
                 robot_init_state: Dict[str, float],
                 additional_components: Optional[List[Any]] = None) -> None:
        """Initialize the task generator.

        Args:
            domino_component: The domino component to use.
            robot: The robot object.
            robot_init_state: Initial state dict for the robot.
            additional_components: Other components to include in state.
        """
        self.domino = domino_component
        self.robot = robot
        self.robot_init_state = robot_init_state
        self.additional_components = additional_components or []
        # Fan side the last generated chain was aligned to (None when
        # domino_fan_aligned_tasks is off, or before the first chain).
        self.last_fan_side: Optional[int] = None

    def generate_tasks(
            self,
            num_tasks: int,
            rng: np.random.Generator,
            log_debug: bool = False,
            possible_num_dominos: Optional[List[int]] = None,
            possible_num_targets: Optional[List[int]] = None,
            possible_num_pivots: Optional[List[int]] = None,
            domino_in_upper_half: bool = False,
            turn_ratio: Optional[float] = None) -> List[EnvironmentTask]:
        """Generate domino sequence tasks.

        Args:
            domino_in_upper_half: If True, shift dominoes to upper
                half of workspace (useful when ball needs space
                in lower half).
            turn_ratio: Fraction of tasks that must contain a turn90
                corner (the caller picks the train or test flag).
        """
        if possible_num_dominos is None:
            possible_num_dominos = CFG.domino_test_num_dominos
        if possible_num_targets is None:
            possible_num_targets = CFG.domino_test_num_targets
        if possible_num_pivots is None:
            possible_num_pivots = CFG.domino_test_num_pivots
        if turn_ratio is None:
            turn_ratio = CFG.domino_test_turn_ratio

        # Turn/straight quota (domino_{train,test}_turn_ratio, shared with
        # min-block generation): the first n_turn tasks must contain a
        # turn90 corner and the rest are generated straight-only.
        n_turn = int(round(num_tasks * turn_ratio))

        tasks = []
        for i_task in range(num_tasks):
            task = self._generate_single_task(i_task,
                                              rng,
                                              possible_num_dominos,
                                              possible_num_targets,
                                              possible_num_pivots,
                                              log_debug,
                                              domino_in_upper_half,
                                              force_turn=i_task < n_turn)
            if task is not None:
                tasks.append(task)

        return tasks

    def _wind_triggered(self) -> bool:
        """True when the cascade is started by wind, not by a push.

        A fan is the only additional component that can start a cascade
        legitimately without the arm touching a domino, so it decides
        both the trigger the certificate sanctions and what the goal
        text may ask for. Those two must never disagree: an agent told
        to push the green in a fan env has no Push skill to push with,
        and every episode it runs is rejected for having no TurnFanOn
        on the record. A ball is a second body the robot can throw at
        the chain, so ball variants are not wind-triggered.
        """
        comp_names = {type(c).__name__ for c in self.additional_components}
        return bool(comp_names) and comp_names <= {"FanComponent"}

    def _generate_single_task(
            self,
            task_idx: int,
            rng: np.random.Generator,
            possible_num_dominos: List[int],
            possible_num_targets: List[int],
            possible_num_pivots: List[int],
            log_debug: bool = False,
            domino_in_upper_half: bool = False,
            force_turn: bool = False) -> Optional[EnvironmentTask]:
        """Generate a single domino task.

        ``force_turn`` is this task's slot in the turn-ratio quota
        (``domino_{train,test}_turn_ratio``): True means the chain must
        contain a turn90 corner (chains
        without one are resampled), False means it is generated
        straight-only. Ignored on the min-block path, which fills its
        own quota from the same ratio.
        """
        if CFG.domino_min_block_tasks:
            return self._generate_min_block_task(task_idx, rng)

        init_dict: Dict[Object, Dict[str, Any]] = {}

        # Robot initial state
        init_dict[self.robot] = self.robot_init_state.copy()

        # Generate domino sequence
        n_dominos = rng.choice(possible_num_dominos)
        n_targets = rng.choice(possible_num_targets)
        n_pivots = rng.choice(possible_num_pivots)

        obj_dict = None
        max_attempts = 1000
        for attempt_num in range(max_attempts):
            if log_debug:
                print(f"\nAttempt {attempt_num} for task {task_idx}")
            candidate_obj_dict = self._generate_domino_sequence(
                rng, n_dominos, n_targets, n_pivots, log_debug, task_idx,
                domino_in_upper_half, force_turn)
            if candidate_obj_dict is None:
                continue

            # Make the chain's terminal block(s) the target(s). The placement
            # loop can otherwise mark a mid-chain block as the target, leaving
            # movable blocks after the goal -- which makes the bridge length
            # ambiguous (an agent over-builds past the target, e.g. a 2-gap
            # task that admits one intermediate but is planned with two).
            # Blocks are placed start-first along the chain, so the
            # highest-index ones are the chain end; re-designating those keeps
            # the target last.
            if CFG.domino_use_domino_blocks_as_target:
                self._retarget_terminal_dominoes(candidate_obj_dict, n_targets)

            # Move intermediate objects if needed. This can fail if the
            # unfinished staging area is too full after collision checking, so
            # keep it inside the attempt loop and resample the solved chain.
            if not CFG.domino_initialize_at_finished_state:
                candidate_obj_dict = \
                    self.stage_movable_blocks(
                        candidate_obj_dict)
                if candidate_obj_dict is None:
                    continue

            obj_dict = candidate_obj_dict
            if log_debug:
                print("Found satisfying domino sequence")
            break

        if obj_dict is None:
            return None

        init_dict.update(obj_dict)

        # Aim the fan down the chain BEFORE it reports its own init
        # state. Doing it at reset instead leaves the task carrying the
        # fan's un-aimed coordinate and the reset check comparing two
        # different positions ("fan_0.y: requested=1.708000
        # reconstructed=1.386534"). The chain's lateral coordinate is
        # known here and nowhere earlier.
        chain_lateral = self._chain_lateral(obj_dict)
        for component in self.additional_components:
            if chain_lateral is not None and hasattr(component,
                                                     "set_lateral_alignment"):
                component.set_lateral_alignment(chain_lateral)

        # Add entries from additional components
        for component in self.additional_components:
            if hasattr(component, 'get_init_dict_entries'):
                component_entries = component.get_init_dict_entries(rng)
                init_dict.update(component_entries)

        init_state = utils.create_state_from_dict(init_dict)

        # Create goal atoms
        goal_atoms = set()
        if CFG.domino_use_domino_blocks_as_target:
            for domino_obj in init_state.get_objects(self.domino.domino_type):
                # pylint: disable=protected-access
                if self.domino._TargetDomino_holds(init_state, [domino_obj]):
                    goal_atoms.add(
                        GroundAtom(self.domino.Toppled, [domino_obj]))
        else:
            for target_obj in init_state.get_objects(self.domino.target_type):
                goal_atoms.add(GroundAtom(self.domino.Toppled, [target_obj]))

        if len(goal_atoms) == 1:
            target_word, target_verb = "the purple domino", "is"
        else:
            target_word, target_verb = "the purple dominoes", "are"
        fan_only = self._wind_triggered()
        # Which skill is sanctioned to start the cascade. The two fan
        # envs differ only here: one presses a button, one declares.
        if not fan_only:
            trigger = "Push"
        elif CFG.env == "pybullet_domino_declare":
            trigger = "DeclareFinished"
        else:
            trigger = "TurnFanOn"
        if fan_only and trigger == "DeclareFinished":
            goal_nl = (
                f"Arrange the blue dominoes as needed (possibly none) such "
                f"that once you declare you have finished building, "
                f"{target_word} {target_verb} toppled. Only the blue "
                f"dominoes may be "
                f"rearranged: the green and purple dominoes must stay "
                f"untouched at their staged poses, upright and never held, "
                f"until you declare finished, and nothing may topple "
                f"before that. There is no switch to press and the robot "
                f"must never push a domino - declaring finished is the "
                f"only thing that starts the cascade.")
        elif fan_only:
            goal_nl = (
                f"Arrange the blue dominoes as needed (possibly none) such "
                f"that when the fan is switched on, {target_word} "
                f"{target_verb} toppled. "
                f"Only the blue dominoes may be rearranged: the green and "
                f"purple dominoes must stay untouched at their staged "
                f"poses, upright and never held, until the fan is switched "
                f"on, and nothing may topple before that. The robot must "
                f"never push a domino - the only way to start the cascade "
                f"is to press the fan's switch.")
        else:
            goal_nl = (
                f"Arrange the blue dominoes as needed (possibly none) such "
                f"that when the green domino is pushed, {target_word} "
                f"{target_verb} toppled. Only the blue dominoes may be "
                f"rearranged: the green and purple dominoes must stay "
                f"untouched at their staged poses, upright and never "
                f"held, until the green is pushed, and nothing may "
                f"topple before that push. Only the green domino may "
                f"ever be pushed.")

        # Cascade-legitimacy evaluator (reward = certified success minus a
        # per-toppled-blue cost), same as the min-block tasks. Attached only
        # where its causal model is valid: targets must be roll-tracked
        # dominoes (the separate ``target_type`` has no roll feature, so the
        # certificate is blind to a direct robot knock on such a target and
        # would certify it at zero blue cost), and dominoes must be the only
        # dynamic component (ball/fan variants topple dominoes legitimately
        # without a robot Push, which the certificate would reject).
        # A fan is a dynamic component, but a legitimate one: it
        # topples dominoes only through the wind, and the certificate
        # can score that as long as it is told the trigger is TurnFanOn
        # rather than Push. A BALL is not - it is a second body the
        # robot can throw at the chain - so ball variants still get no
        # evaluator.
        evaluator = None
        if CFG.domino_use_domino_blocks_as_target and \
                (not self.additional_components or fan_only):
            # Imported lazily: env.py imports this module at load time.
            # pylint: disable-next=import-outside-toplevel
            from predicators.envs.pybullet_domino.env import DominoEvaluator
            num_movables = sum(
                1 for obj in init_state.get_objects(self.domino.domino_type)
                # pylint: disable-next=protected-access
                if DominoComponent._MovableBlock_holds(init_state, [obj]))
            evaluator = DominoEvaluator(goal_atoms,
                                        num_movables,
                                        trigger_option_name=trigger)
            # State the reward structure so a rejected goal-reaching
            # attempt reads as "no solve bonus", not as a fatal
            # per-blue penalty: run_20260716_215533 burned its budget
            # theorizing that any disturbed blue disqualifies a solve.
            cost = CFG.domino_block_cost
            goal_nl += (f" Scoring: a solve earns +1 reward, and each blue "
                        f"domino the cascade consumes (toppled or shoved "
                        f"out of place) costs {cost:g}, so a solve that "
                        f"uses one blue scores +{1.0 - cost:g}. Using "
                        f"blues never disqualifies a solve.")
            # Same reasoning for the legitimacy rule (see
            # goal_text.CASCADE_VERIFICATION_NL): an arm-assisted layout
            # otherwise fails with verdicts the agent cannot explain.
            if not fan_only:
                goal_nl += goal_text.CASCADE_VERIFICATION_NL
            elif trigger == "DeclareFinished":
                goal_nl += goal_text.DECLARE_VERIFICATION_NL
            else:
                goal_nl += goal_text.WIND_VERIFICATION_NL

        return EnvironmentTask(init_state,
                               goal_atoms,
                               goal_nl=goal_nl,
                               evaluator=evaluator)

    def _generate_min_block_task(
            self, task_idx: int,
            rng: np.random.Generator) -> Optional[EnvironmentTask]:
        """Reach-limited "minimum-blocks" task.

        A green start block and a purple target sit a sampled distance
        apart (``CFG.domino_min_block_span_range``), with a generous
        pile of staged blue blocks. The span is chosen so bridging
        start->target must happen near the topple-reach limit; the per-
        task ``DominoEvaluator`` (offline k_star = K*) is attached
        afterwards with physics in the loop — see
        ``min_block_generation._assign_min_blocks``. The goal is simply
        to topple the target; each blue the cascade consumes costs
        reward, so a solver that over-estimates reach under-builds and
        fails while an over-builder pays the cost. Returns ``None`` if
        no in-bounds placement is found.
        """
        dominos = self.domino.dominos
        num_blues = min(CFG.domino_min_block_num_blues, len(dominos) - 2)
        if num_blues < 1:
            return None
        span_lo = CFG.domino_min_block_span_lo
        span_hi = CFG.domino_min_block_span_hi
        x_lb, x_ub = self.domino.domino_x_lb, self.domino.domino_x_ub
        y_lb, y_ub = self.domino.domino_y_lb, self.domino.domino_y_ub

        placement = None
        for _ in range(500):
            rotation = float(rng.choice([0.0, np.pi / 2, -np.pi / 2]))
            span = float(rng.uniform(span_lo, span_hi))
            # Chain travels along `rotation`, matching _place_straight_domino
            # (dx=sin, dy=cos); the block faces this way so a Push topples it
            # toward the target.
            tx = rng.uniform(x_lb, x_ub)
            ty = rng.uniform(y_lb, y_ub)
            sx = tx - span * np.sin(rotation)
            sy = ty - span * np.cos(rotation)
            if x_lb < sx < x_ub and y_lb < sy < y_ub:
                placement = (rotation, sx, sy, tx, ty)
                break
        if placement is None:
            return None
        rotation, sx, sy, tx, ty = placement

        # Build the domino-only dict first: the staging collision-check reads
        # object yaw, which the robot's feature dict lacks, so the robot must
        # be added only after staging (matching the main generator).
        obj_dict: Dict[Object, Dict[str, Any]] = {}
        obj_dict[dominos[0]] = self.domino.place_domino(0,
                                                        sx,
                                                        sy,
                                                        rotation,
                                                        is_start_block=True,
                                                        rng=rng,
                                                        task_idx=task_idx)
        obj_dict[dominos[1]] = self.domino.place_domino(1,
                                                        tx,
                                                        ty,
                                                        rotation,
                                                        is_target_block=True,
                                                        rng=rng,
                                                        task_idx=task_idx)
        for i in range(num_blues):
            # Placed at the start for now; staged to a pickable spot below.
            obj_dict[dominos[2 + i]] = self.domino.place_domino(
                2 + i, sx, sy, rotation, rng=rng, task_idx=task_idx)

        staged = self.stage_movable_blocks(obj_dict)
        if staged is None:
            return None

        init_dict: Dict[Object, Dict[str, Any]] = {
            self.robot: self.robot_init_state.copy()
        }
        init_dict.update(staged)
        init_state = utils.create_state_from_dict(init_dict)
        goal_atoms = set()
        for domino_obj in init_state.get_objects(self.domino.domino_type):
            # pylint: disable=protected-access
            if self.domino._TargetDomino_holds(init_state, [domino_obj]):
                goal_atoms.add(GroundAtom(self.domino.Toppled, [domino_obj]))
        return EnvironmentTask(init_state,
                               goal_atoms,
                               goal_nl=(goal_text.MIN_BLOCK_WIND_GOAL_NL
                                        if self._wind_triggered() else
                                        goal_text.MIN_BLOCK_GOAL_NL))

    # A chain's travel direction is (sin rotation, cos rotation) -- see
    # _place_straight_domino -- so rotation is measured from +y, turning
    # toward +x. A fan on side_idx blows along its own yaw, world
    # (cos yaw, sin yaw): left(0) +x, right(1) -x, back(2) +y,
    # front(3) -y. These are those two conventions reconciled, which is
    # the only place the fan's frame and the chain's frame meet.
    _FAN_SIDE_TO_ROTATION: ClassVar[Dict[int, float]] = {
        0: np.pi / 2,  # left fan blows +x
        1: -np.pi / 2,  # right fan blows -x
        2: 0.0,  # back fan blows +y
        3: np.pi,  # front fan blows -y
    }

    def _fan_aligned_start(self, rng: np.random.Generator, x_lb: float,
                           x_ub: float, y_lb: float,
                           y_ub: float) -> Tuple[float, float, float, int]:
        """Start pose and travel direction for a wind-started chain.

        Picks a fan side, points the chain downwind, and puts the start
        block in the upwind fifth of that axis so the rest of the chain
        has the workspace to run into. Free across the crosswind axis:
        the wind is uniform, so where the chain sits sideways does not
        change whether it cascades, and varying it keeps the task set
        from collapsing onto one line.
        """
        # Only sides that actually carry a fan (see
        # domino_fan_num_sides); aligning a chain to a fan that is not
        # there makes the task unsolvable.
        side = int(rng.integers(0, max(1, CFG.domino_fan_num_sides)))
        rotation = self._FAN_SIDE_TO_ROTATION[side]
        dx, dy = np.sin(rotation), np.cos(rotation)
        lead = 0.2  # fraction of the axis reserved upwind of the start
        if abs(dx) > abs(dy):  # travelling along x
            x = (x_lb + lead * (x_ub - x_lb) if dx > 0 else x_ub - lead *
                 (x_ub - x_lb))
            y = rng.uniform(y_lb, y_ub)
        else:  # travelling along y
            x = rng.uniform(x_lb, x_ub)
            y = (y_lb + lead * (y_ub - y_lb) if dy > 0 else y_ub - lead *
                 (y_ub - y_lb))
        return x, y, rotation, side

    def _generate_domino_sequence(self,
                                  rng: np.random.Generator,
                                  n_dominos: int,
                                  n_targets: int,
                                  n_pivots: int,
                                  _log_debug: bool = False,
                                  task_idx: Optional[int] = None,
                                  domino_in_upper_half: bool = False,
                                  force_turn: bool = False) -> Optional[Dict]:
        """Generate a sequence of dominoes, targets, and pivots.

        With ``force_turn`` True the completed chain must contain at
        least one turn90 (otherwise ``None``, so the caller's attempt
        loop resamples); with False the per-step choice is restricted to
        straight placements. See ``domino_{train,test}_turn_ratio``.
        """
        obj_dict: Dict[Object, Dict[str, Any]] = {}
        domino_count = 0
        target_count = 0
        pivot_count = 0
        just_placed_target = False
        just_turned_90 = False

        y_lb, y_ub = self.domino.domino_y_lb, self.domino.domino_y_ub
        x_lb, x_ub = self.domino.domino_x_lb, self.domino.domino_x_ub
        if domino_in_upper_half:
            y_lb += 0.4  # domino_in_upper_half_shift
            y_ub += 0.4

        def _in_bounds(nx: float, ny: float) -> bool:
            return x_lb < nx < x_ub and y_lb < ny < y_ub

        # Initial position and orientation. A wind-started chain has to
        # run downwind from the upwind edge; a robot-pushed one can start
        # anywhere and face any of three ways (the fourth, -y, has never
        # been in this list).
        self.last_fan_side = None
        if CFG.domino_fan_aligned_tasks:
            x, y, rotation, self.last_fan_side = self._fan_aligned_start(
                rng, x_lb, x_ub, y_lb, y_ub)
        else:
            x = rng.uniform(x_lb, x_ub)
            y = rng.uniform(y_lb, y_ub)
            rotation = rng.choice([0, np.pi / 2, -np.pi / 2])
        gap = self.domino.pos_gap

        # Place first domino (start block)
        obj_dict[self.domino.dominos[domino_count]] = self.domino.place_domino(
            domino_count,
            x,
            y,
            rotation,
            is_start_block=True,
            rng=rng,
            task_idx=task_idx)
        domino_count += 1

        expected_count = self._get_expected_domino_count(n_dominos, n_targets)

        # When targets are domino blocks, they are re-designated as the
        # chain's terminal block(s) after generation (see
        # _retarget_terminal_dominoes), so here we just fill the chain to
        # length with regular blocks. This also avoids overrunning the
        # fixed-size dominos[] list: the interleaved loop below could let a
        # turn (which places two blocks at once) push the count past the last
        # slot when a max-size task leaves no slack for the targets — the
        # index-out-of-range crash. The turn90 guard
        # (domino_count + 1 >= expected_count -> straight) keeps this loop in
        # bounds.
        if CFG.domino_use_domino_blocks_as_target:
            # ``block_yaw`` tracks the smooth 45-deg-per-turn yaw increment so
            # straight runs after a turn keep one constant yaw; positions still
            # follow ``rotation`` (the travel direction).
            block_yaw = rotation
            had_turn = False
            while domino_count < expected_count:
                result = self._place_next_domino(
                    rng, obj_dict, x, y, rotation, gap, domino_count,
                    pivot_count, target_count, n_pivots, n_dominos, n_targets,
                    just_placed_target, just_turned_90, _in_bounds, task_idx,
                    block_yaw, force_turn)
                if not result.success:
                    return None
                x, y, rotation = result.x, result.y, result.rotation
                domino_count = result.domino_count
                pivot_count = result.pivot_count
                just_turned_90 = result.just_turned_90
                had_turn = had_turn or result.just_turned_90
                block_yaw = (result.block_yaw
                             if result.block_yaw is not None else rotation)
            if domino_count == expected_count and pivot_count == n_pivots:
                if force_turn and not had_turn:
                    return None
                return obj_dict
            return None

        # Separate target objects (use_domino_blocks_as_target=False):
        # interleave regular dominoes and target-typed objects.
        had_turn = False
        while self._should_continue_placement(domino_count, target_count,
                                              n_dominos, n_targets):
            can_place_target = (domino_count >= 2 and target_count < n_targets
                                and not just_placed_target)
            can_place_domino = domino_count < expected_count

            should_place_domino = (not can_place_target
                                   or rng.random() > 0.5) and can_place_domino

            if should_place_domino:
                result = self._place_next_domino(rng,
                                                 obj_dict,
                                                 x,
                                                 y,
                                                 rotation,
                                                 gap,
                                                 domino_count,
                                                 pivot_count,
                                                 target_count,
                                                 n_pivots,
                                                 n_dominos,
                                                 n_targets,
                                                 just_placed_target,
                                                 just_turned_90,
                                                 _in_bounds,
                                                 task_idx,
                                                 force_turn=force_turn)
                if not result.success:
                    return None
                x, y, rotation = result.x, result.y, result.rotation
                domino_count = result.domino_count
                pivot_count = result.pivot_count
                target_count += result.target_count
                just_turned_90 = result.just_turned_90
                had_turn = had_turn or result.just_turned_90
                just_placed_target = result.just_placed_target
            else:
                result = self._place_next_target(rng, obj_dict, x, y, rotation,
                                                 gap, domino_count,
                                                 target_count, _in_bounds,
                                                 task_idx)
                if not result.success:
                    return None
                x, y, rotation = result.x, result.y, result.rotation
                domino_count = result.domino_count
                target_count = result.target_count
                just_placed_target = True
                just_turned_90 = False

        if self._check_placement_complete(domino_count, target_count,
                                          pivot_count, n_dominos, n_targets,
                                          n_pivots):
            if force_turn and not had_turn:
                return None
            return obj_dict
        return None

    def _retarget_terminal_dominoes(self, obj_dict: Dict[Object, Any],
                                    n_targets: int) -> None:
        """Recolor so the last ``n_targets`` placed blocks are the target(s).

        Mutates ``obj_dict`` in place. Dominoes are placed start-first
        along the chain, so ``self.domino.dominos`` index order is chain
        order: the terminal ``n_targets`` blocks become targets (purple)
        and every other non-start block becomes movable (blue). No-op
        for ``n_targets <= 0``. (Glue state is not preserved; it only
        applies when ``domino_has_glued_dominos`` is set, which is off
        by default.)
        """
        if n_targets <= 0:
            return
        placed = [d for d in self.domino.dominos if d in obj_dict]
        terminal = set(placed[-n_targets:])
        target_color = self.domino.target_domino_color
        movable_color = self.domino.domino_color
        for idx, domino_obj in enumerate(placed):
            if idx == 0:
                continue  # start block keeps its color
            color = target_color if domino_obj in terminal else movable_color
            entry = obj_dict[domino_obj]
            entry["r"], entry["g"], entry["b"] = color[0], color[1], color[2]

    def _get_expected_domino_count(self, n_dominos: int,
                                   n_targets: int) -> int:
        if CFG.domino_use_domino_blocks_as_target:
            return n_dominos + n_targets
        return n_dominos

    def _should_continue_placement(self, domino_count: int, target_count: int,
                                   n_dominos: int, n_targets: int) -> bool:
        expected = self._get_expected_domino_count(n_dominos, n_targets)
        if CFG.domino_use_domino_blocks_as_target:
            return domino_count < expected or target_count < n_targets
        return domino_count < n_dominos or target_count < n_targets

    def _check_placement_complete(self, domino_count: int, target_count: int,
                                  pivot_count: int, n_dominos: int,
                                  n_targets: int, n_pivots: int) -> bool:
        expected = self._get_expected_domino_count(n_dominos, n_targets)
        if CFG.domino_use_domino_blocks_as_target:
            return (domino_count == expected and target_count == n_targets
                    and pivot_count == n_pivots)
        return (domino_count == n_dominos and target_count == n_targets
                and pivot_count == n_pivots)

    def _place_next_domino(self,
                           rng: np.random.Generator,
                           obj_dict: Dict,
                           x: float,
                           y: float,
                           rotation: float,
                           gap: float,
                           domino_count: int,
                           pivot_count: int,
                           target_count: int,
                           n_pivots: int,
                           n_dominos: int,
                           n_targets: int,
                           just_placed_target: bool,
                           just_turned_90: bool,
                           _in_bounds: Callable[[float, float], bool],
                           task_idx: Optional[int] = None,
                           block_yaw: Optional[float] = None,
                           force_turn: bool = False) -> PlacementResult:
        """Place the next domino using various strategies."""
        turn_choices = self.domino.turn_choices.copy()
        if pivot_count >= n_pivots and "pivot180" in turn_choices:
            turn_choices.remove("pivot180")
        if just_turned_90 and "turn90" in turn_choices:
            turn_choices.remove("turn90")
        if just_placed_target or not force_turn:
            # Straight-only slot in the turn-ratio quota (or a
            # cooldown step right after a target).
            turn_choices = ["straight"]

        choice = rng.choice(turn_choices)

        should_place_target_at_end = False
        if CFG.domino_use_domino_blocks_as_target and choice in [
                "turn90", "pivot180"
        ]:
            if target_count < n_targets and rng.random() > 0.5:
                should_place_target_at_end = True

        if choice == "straight":
            return self._place_straight_domino(rng, obj_dict, x, y, rotation,
                                               gap, domino_count, _in_bounds,
                                               task_idx, block_yaw)
        if choice == "turn90":
            return self._place_turn90_domino(rng, obj_dict, x, y, rotation,
                                             gap, domino_count, n_dominos,
                                             n_targets, _in_bounds, task_idx,
                                             should_place_target_at_end,
                                             block_yaw)
        if choice == "pivot180":
            return self._place_pivot180_domino(rng, obj_dict, x, y, rotation,
                                               gap, domino_count, pivot_count,
                                               _in_bounds, task_idx,
                                               should_place_target_at_end)
        return self._place_straight_domino(rng, obj_dict, x, y, rotation, gap,
                                           domino_count, _in_bounds, task_idx,
                                           block_yaw)

    def _place_straight_domino(
            self,
            rng: np.random.Generator,
            obj_dict: Dict[Object, Any],
            x: float,
            y: float,
            rotation: float,
            gap: float,
            domino_count: int,
            _in_bounds: Callable[[float, float], bool],
            task_idx: Optional[int],
            block_yaw: Optional[float] = None) -> PlacementResult:
        # Travel direction (positions) follows ``rotation``; the block is laid
        # at ``block_yaw`` (the smooth turn increment) when one has been
        # established, else at ``rotation``. They are the same box, so a run
        # after a turn reads as one constant yaw instead of flipping 180 deg.
        yaw = rotation if block_yaw is None else block_yaw
        dx = gap * np.sin(rotation)
        dy = gap * np.cos(rotation)
        new_x, new_y = x + dx, y + dy

        if not _in_bounds(new_x, new_y):
            return PlacementResult(success=False,
                                   x=x,
                                   y=y,
                                   rotation=rotation,
                                   domino_count=domino_count,
                                   block_yaw=block_yaw)

        obj_dict[self.domino.dominos[domino_count]] = self.domino.place_domino(
            domino_count,
            new_x,
            new_y,
            yaw,
            is_start_block=False,
            rng=rng,
            task_idx=task_idx)

        return PlacementResult(success=True,
                               x=new_x,
                               y=new_y,
                               rotation=rotation,
                               domino_count=domino_count + 1,
                               block_yaw=block_yaw)

    def _place_turn90_domino(
            self,
            rng: np.random.Generator,
            obj_dict: Dict[Object, Any],
            x: float,
            y: float,
            rotation: float,
            gap: float,
            domino_count: int,
            n_dominos: int,
            n_targets: int,
            _in_bounds: Callable[[float, float], bool],
            task_idx: Optional[int],
            should_place_target_at_end: bool,
            block_yaw: Optional[float] = None) -> PlacementResult:
        expected_count = self._get_expected_domino_count(n_dominos, n_targets)
        if domino_count + 1 >= expected_count:
            return self._place_straight_domino(rng, obj_dict, x, y, rotation,
                                               gap, domino_count, _in_bounds,
                                               task_idx, block_yaw)

        # The two turn blocks' yaws step 45 deg per block off the running block
        # yaw (``block_yaw``, = ``rotation`` before any turn), so successive
        # turns keep incrementing rather than resetting and a 90 deg turn reads
        # as a smooth increment (yaw, yaw +/- 45, yaw +/- 90). Positions are
        # independent of this representation and follow ``rotation`` (the
        # travel direction): ``d1_dir`` is the chain's toppling direction one
        # 45 deg step into the turn; d1 sits one gap ahead of the current block
        # along the entry direction (no lateral shift, so it stays on the
        # previous block's fall line) and d2 one gap ahead of d1 along d1_dir.
        base_yaw = rotation if block_yaw is None else block_yaw
        turn_direction = rng.choice([-1, 1])
        d1_dir = rotation - turn_direction * np.pi / 4
        d1_yaw = base_yaw + turn_direction * np.pi / 4
        d1_x = x + gap * np.sin(rotation)
        d1_y = y + gap * np.cos(rotation)
        # Lateral "side" offset for the first turn block, kept at 0 (matching
        # the legacy generator, which only nudged the turn-completing block).
        # Exposed here as an explicit tunable knob -- raise it to also shift
        # the first block orthogonal to its post-turn travel direction
        # ``d1_dir`` if future tuning needs more overlap entering the bend.
        d1_side_offset = -self.domino.domino_width / 2
        # d1_side_offset = 0
        d1_x += turn_direction * d1_side_offset * np.cos(d1_dir)
        d1_y -= turn_direction * d1_side_offset * np.sin(d1_dir)

        if not _in_bounds(d1_x, d1_y):
            return PlacementResult(success=False,
                                   x=x,
                                   y=y,
                                   rotation=rotation,
                                   domino_count=domino_count)

        obj_dict[self.domino.dominos[domino_count]] = self.domino.place_domino(
            domino_count,
            d1_x,
            d1_y,
            d1_yaw,
            is_start_block=False,
            rng=rng,
            task_idx=task_idx)
        domino_count += 1

        # Second turn block: one gap ahead of d1 along the chain direction,
        # completing the 90 deg turn. Its yaw continues the +/-45 increment;
        # ``d2_rot`` (the same cardinal orientation, 180 deg off) is returned
        # as the travel direction so subsequent straight blocks lay out
        # correctly, while ``d2_yaw`` is threaded as the running block yaw so
        # those blocks keep this orientation instead of flipping.
        d2_yaw = base_yaw + turn_direction * np.pi / 2
        d2_rot = rotation - turn_direction * np.pi / 2
        d2_x = d1_x + gap * np.sin(d1_dir)
        d2_y = d1_y + gap * np.cos(d1_dir)
        # Lateral "side" offset (ported from the legacy turn generator): in
        # addition to stepping the turn-completing block one gap *along* the
        # chain, nudge it a half-width *orthogonal* to its own travel
        # direction. Without this sideways shift the falling chain only moves
        # along one axis and clips past the corner block, so the cascade
        # stalls; the inward nudge keeps the toppling dominoes overlapping
        # through the bend. ``(cos d2_rot, -sin d2_rot)`` is the unit vector
        # perpendicular to the block's facing, signed by the turn direction.
        side_offset = -self.domino.domino_width / 2
        # side_offset = 0
        d2_x += turn_direction * side_offset * np.cos(d2_rot)
        d2_y -= turn_direction * side_offset * np.sin(d2_rot)

        if not _in_bounds(d2_x, d2_y):
            return PlacementResult(success=False,
                                   x=x,
                                   y=y,
                                   rotation=rotation,
                                   domino_count=domino_count)

        obj_dict[self.domino.dominos[domino_count]] = self.domino.place_domino(
            domino_count,
            d2_x,
            d2_y,
            d2_yaw,
            is_start_block=False,
            is_target_block=should_place_target_at_end,
            rng=rng,
            task_idx=task_idx)

        target_inc = 1 if should_place_target_at_end else 0
        return PlacementResult(success=True,
                               x=d2_x,
                               y=d2_y,
                               rotation=d2_rot,
                               domino_count=domino_count + 1,
                               target_count=target_inc,
                               just_turned_90=True,
                               just_placed_target=should_place_target_at_end,
                               block_yaw=d2_yaw)

    def _place_pivot180_domino(
            self, rng: np.random.Generator, obj_dict: Dict[Object, Any],
            x: float, y: float, rotation: float, gap: float, domino_count: int,
            pivot_count: int, _in_bounds: Callable[[float, float], bool],
            task_idx: Optional[int],
            should_place_target_at_end: bool) -> PlacementResult:
        pivot_direction = rng.choice([-1, 1])
        side_offset = self.domino.pivot_width / 2

        pivot_x = x + gap * (2 / 3) * np.sin(rotation)
        pivot_y = y + gap * (2 / 3) * np.cos(rotation)
        pivot_x -= pivot_direction * side_offset * np.cos(rotation)
        pivot_y -= pivot_direction * side_offset * np.sin(rotation)

        if not _in_bounds(pivot_x, pivot_y):
            return PlacementResult(success=False,
                                   x=x,
                                   y=y,
                                   rotation=rotation,
                                   domino_count=domino_count,
                                   pivot_count=pivot_count)

        obj_dict[self.domino.
                 pivots[pivot_count]] = self.domino.place_pivot_or_target(
                     pivot_x, pivot_y, rotation)

        domino_x = pivot_x - (gap * (2 / 3)) * np.sin(rotation)
        domino_y = pivot_y - (gap * (2 / 3)) * np.cos(rotation)
        domino_x -= pivot_direction * side_offset * np.cos(rotation)
        domino_y += pivot_direction * side_offset * -np.sin(rotation)

        if not _in_bounds(domino_x, domino_y):
            return PlacementResult(success=False,
                                   x=x,
                                   y=y,
                                   rotation=rotation,
                                   domino_count=domino_count,
                                   pivot_count=pivot_count)

        new_rotation = rotation + np.pi
        obj_dict[self.domino.dominos[domino_count]] = self.domino.place_domino(
            domino_count,
            domino_x,
            domino_y,
            new_rotation,
            is_start_block=False,
            is_target_block=should_place_target_at_end,
            rng=rng,
            task_idx=task_idx)

        target_inc = 1 if should_place_target_at_end else 0
        return PlacementResult(success=True,
                               x=domino_x,
                               y=domino_y,
                               rotation=new_rotation,
                               domino_count=domino_count + 1,
                               pivot_count=pivot_count + 1,
                               target_count=target_inc,
                               just_placed_target=should_place_target_at_end)

    def _place_next_target(self, rng: np.random.Generator,
                           obj_dict: Dict[Object, Any], x: float, y: float,
                           rotation: float, gap: float, domino_count: int,
                           target_count: int,
                           _in_bounds: Callable[[float, float], bool],
                           task_idx: Optional[int]) -> PlacementResult:
        dx = gap * np.sin(rotation)
        dy = gap * np.cos(rotation)
        target_x, target_y = x + dx, y + dy

        if not _in_bounds(target_x, target_y):
            return PlacementResult(success=False,
                                   x=x,
                                   y=y,
                                   rotation=rotation,
                                   domino_count=domino_count,
                                   target_count=target_count)

        if CFG.domino_use_domino_blocks_as_target:
            obj_dict[
                self.domino.dominos[domino_count]] = self.domino.place_domino(
                    domino_count,
                    target_x,
                    target_y,
                    rotation,
                    is_target_block=True,
                    rng=rng,
                    task_idx=task_idx)
            return PlacementResult(success=True,
                                   x=target_x,
                                   y=target_y,
                                   rotation=rotation,
                                   domino_count=domino_count + 1,
                                   target_count=target_count + 1)
        obj_dict[self.domino.
                 targets[target_count]] = self.domino.place_pivot_or_target(
                     target_x, target_y, rotation)
        return PlacementResult(success=True,
                               x=target_x,
                               y=target_y,
                               rotation=rotation,
                               domino_count=domino_count,
                               target_count=target_count + 1)

    def stage_movable_blocks(self, obj_dict: Dict) -> Optional[Dict]:
        """Scatter the movable (blue) dominoes and pivots onto pickable staging
        spots, leaving start/target/heavy blocks in place; None if the staging
        grid can't fit them all clear of collisions and grasp footprints.

        Public: the min-block / heavy generators (``min_block_utils`` /
        ``min_block_generation``) call this to stage their scenes, so it is
        part of this generator's layout API rather than an internal helper.
        """
        intermediate_objects = []
        eps = 1e-3

        for domino in self.domino.dominos:
            if domino in obj_dict:
                data = obj_dict[domino]
                is_start = (abs(
                    data.get("r", 0.0) -
                    self.domino.start_domino_color[0]) < eps and abs(
                        data.get("g", 0.0) - self.domino.start_domino_color[1])
                            < eps and abs(
                                data.get("b", 0.0) -
                                self.domino.start_domino_color[2]) < eps)

                is_target = False
                if CFG.domino_use_domino_blocks_as_target:
                    is_target = (
                        (abs(
                            data.get("r", 0.0) -
                            self.domino.target_domino_color[0]) < eps and abs(
                                data.get("g", 0.0) -
                                self.domino.target_domino_color[1]) < eps
                         and abs(
                             data.get("b", 0.0) -
                             self.domino.target_domino_color[2]) < eps)
                        or (abs(
                            data.get("r", 0.0) -
                            self.domino.glued_domino_color[0]) < eps and abs(
                                data.get("g", 0.0) -
                                self.domino.glued_domino_color[1]) < eps
                            and abs(
                                data.get("b", 0.0) -
                                self.domino.glued_domino_color[2]) < eps))

                # Heavy (gray) blocks are scenery, not workpieces: they stay
                # where the task placed them (and count as occupied below).
                is_heavy = self.domino.is_heavy_color(data.get("r", 0.0),
                                                      data.get("g", 0.0),
                                                      data.get("b", 0.0))

                if not is_start and not is_target and not is_heavy:
                    intermediate_objects.append((domino, "domino"))

        for pivot in self.domino.pivots:
            if pivot in obj_dict:
                intermediate_objects.append((pivot, "pivot"))

        if not intermediate_objects:
            return obj_dict

        occupied = {
            obj: data
            for obj, data in obj_dict.items()
            if all(obj != intermediate[0]
                   for intermediate in intermediate_objects)
        }

        x_margin = self.domino.domino_width
        y_margin = self.domino.domino_width
        spacing = self.domino.domino_width * 1.5

        # Gripper swept-footprint half-extents for a top-down grasp of a
        # staged (yaw=0) domino. The open fingers span the domino's depth axis
        # (local y) and reach ~1.45x the domino width from the grasp center;
        # the hand spans ~0.85x along the long axis (local x). Measured from
        # the Fetch gripper at the descend pose. A staged domino must keep this
        # footprint clear of every other object, otherwise it lands placed but
        # *un-pickable* -- BiRRT finds no collision-free descent because a
        # neighbor (especially a perpendicular one a few cm away in y) sits
        # inside the finger sweep even though the footprints don't overlap.
        grasp_clear_hand = self.domino.domino_width * 0.85
        grasp_clear_finger = self.domino.domino_width * 1.45
        x_values = np.arange(self.domino.domino_x_lb + x_margin,
                             self.domino.domino_x_ub - x_margin + eps, spacing)
        y_values = np.arange(self.domino.domino_y_lb + y_margin,
                             self.domino.domino_y_ub - y_margin + eps, spacing)
        candidate_xy = [(float(x), float(y)) for y in y_values
                        for x in x_values]

        # The corridor the robot has to build through: the segment from
        # the start block to the far target. A blue parked inside it is
        # not merely untidy - it sits within the gripper's finger sweep
        # of a bridge slot, and Place then has no collision-free
        # descent. Whether that happens is pure luck about where the
        # chain landed: a uniformly-placed chain usually sits to one
        # side and leaves whole staging cells free, while a
        # wind-ALIGNED chain starts in the upwind fifth and runs through
        # the middle of the workspace, straight across the staging row.
        # Measured: plain domino parks its blues at x = 0.470 / 0.575
        # against a chain spanning 0.697-0.991 (clear), the fan env at
        # 0.470 / 0.680 against 0.540-0.834 - the second one 66 mm from
        # a slot, inside a 100 mm finger sweep.
        corridor = self._chain_corridor(occupied)

        for obj, obj_type in intermediate_objects:
            placed = False
            for new_x, new_y in candidate_xy:
                candidate: Dict[str, float]
                if obj_type == "domino":
                    candidate = {
                        "x": new_x,
                        "y": new_y,
                        "z": self.domino.z_lb + self.domino.domino_height / 2,
                        "yaw": 0.0,
                        "roll": 0.0,
                        "r": self.domino.domino_color[0],
                        "g": self.domino.domino_color[1],
                        "b": self.domino.domino_color[2],
                        "is_held": 0.0,
                    }
                else:
                    candidate = {
                        "x": new_x,
                        "y": new_y,
                        "z": self.domino.z_lb,
                        "yaw": 0.0,
                    }
                if self._placement_collides(obj, candidate, occupied):
                    continue
                if corridor is not None and _dist_to_segment(
                    (new_x, new_y), *corridor) < grasp_clear_finger:
                    continue
                if obj_type == "domino" and self._grasp_clearance_blocked(
                        candidate, occupied, grasp_clear_hand,
                        grasp_clear_finger):
                    continue
                obj_dict[obj] = candidate
                occupied[obj] = candidate
                placed = True
                break
            if not placed:
                return None

        return obj_dict

    def _chain_lateral(self, obj_dict: Dict) -> Optional[float]:
        """The y the chain was laid at, or None if there is no chain.

        Only meaningful for a wind-aligned layout, where every block
        shares one lateral coordinate; the staged blues sit on their own
        row and are excluded by taking the START block's.
        """
        for obj, data in obj_dict.items():
            if obj.type != self.domino.domino_type or "y" not in data:
                continue
            eps = 1e-3
            if all(
                    abs(data.get(c, 0.0) -
                        self.domino.start_domino_color[i]) < eps
                    for i, c in enumerate(("r", "g", "b"))):
                return float(data["y"])
        return None

    def _chain_corridor(
        self, occupied: Dict[Object, Dict[str, float]]
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Endpoints of the line the chain will be built along, or None.

        The fixed blocks at staging time are the start block and the
        target(s); the bridge runs between them, so the two extreme
        fixed positions bound the corridor. None when fewer than two
        are present and there is nothing to keep clear of.
        """
        pts = [(float(d["x"]), float(d["y"])) for o, d in occupied.items()
               if o.type == self.domino.domino_type and "x" in d]
        if len(pts) < 2:
            return None
        far = max(
            ((a, b) for i, a in enumerate(pts) for b in pts[i + 1:]),
            key=lambda ab: np.hypot(ab[0][0] - ab[1][0], ab[0][1] - ab[1][1]),
            default=None)
        return far

    def _placement_collides(self, obj: Object, candidate: Dict[str, float],
                            occupied: Dict[Object, Dict[str, float]]) -> bool:
        """Check whether ``candidate`` overlaps any occupied object."""
        candidate_rect = self._placement_rect(obj, candidate)
        for other_obj, other_data in occupied.items():
            if geometry.rects_overlap(
                    candidate_rect,
                    self._placement_rect(other_obj, other_data)):
                return True
        return False

    def _grasp_clearance_blocked(self, candidate: Dict[str, float],
                                 occupied: Dict[Object, Dict[str, float]],
                                 half_hand: float, half_finger: float) -> bool:
        """Whether the gripper's swept grasp footprint at ``candidate`` would
        overlap another object, leaving the staged domino un-pickable.

        ``half_hand``/``half_finger`` are the gripper footprint half-
        extents along the domino's width axis (``(cos, sin)``) and
        depth/finger-span axis (``(-sin, cos)``). The check is the same
        oriented-rectangle overlap test used for placement, but against
        the larger gripper footprint.
        """
        clear_rect = geometry.domino_footprint(candidate["x"], candidate["y"],
                                               candidate.get("yaw", 0.0),
                                               half_hand, half_finger)
        for other_obj, other_data in occupied.items():
            if geometry.rects_overlap(
                    clear_rect, self._placement_rect(other_obj, other_data)):
                return True
        return False

    def _placement_rect(self, obj: Object, data: Dict[str,
                                                      float]) -> geometry.Rect:
        """Conservative oriented footprint (corner list) for collision
        checks."""
        if obj.type == self.domino.domino_type:
            width = self.domino.domino_width
            depth = self.domino.domino_depth
        elif obj.type == self.domino.pivot_type:
            width = self.domino.pivot_width
            depth = self.domino.pivot_width
        else:
            width = self.domino.domino_width
            depth = self.domino.domino_width

        padding = 0.003
        return geometry.domino_footprint(data["x"], data["y"], data["yaw"],
                                         width / 2 + padding,
                                         depth / 2 + padding)
