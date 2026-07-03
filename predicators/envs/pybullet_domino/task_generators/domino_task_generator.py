"""Task generator for domino-based tasks."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from predicators import utils
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent, PlacementResult
from predicators.envs.pybullet_domino.task_generators.base_generator import \
    TaskGenerator
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, GroundAtom, Object


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

    def generate_tasks(
            self,
            num_tasks: int,
            rng: np.random.Generator,
            log_debug: bool = False,
            possible_num_dominos: Optional[List[int]] = None,
            possible_num_targets: Optional[List[int]] = None,
            possible_num_pivots: Optional[List[int]] = None,
            domino_in_upper_half: bool = False) -> List[EnvironmentTask]:
        """Generate domino sequence tasks.

        Args:
            domino_in_upper_half: If True, shift dominoes to upper
                half of workspace (useful when ball needs space
                in lower half).
        """
        if possible_num_dominos is None:
            possible_num_dominos = CFG.domino_test_num_dominos
        if possible_num_targets is None:
            possible_num_targets = CFG.domino_test_num_targets
        if possible_num_pivots is None:
            possible_num_pivots = CFG.domino_test_num_pivots

        tasks = []
        for i_task in range(num_tasks):
            task = self._generate_single_task(i_task, rng,
                                              possible_num_dominos,
                                              possible_num_targets,
                                              possible_num_pivots, log_debug,
                                              domino_in_upper_half)
            if task is not None:
                tasks.append(task)

        return tasks

    def _generate_single_task(
            self,
            task_idx: int,
            rng: np.random.Generator,
            possible_num_dominos: List[int],
            possible_num_targets: List[int],
            possible_num_pivots: List[int],
            log_debug: bool = False,
            domino_in_upper_half: bool = False) -> Optional[EnvironmentTask]:
        """Generate a single domino task."""
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
                domino_in_upper_half)
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
                    self._move_intermediate_objects_to_unfinished_state(
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
        goal_nl = (
            f"Move the blue dominoes such that when the green domino is "
            f"pushed, {target_word} {target_verb} toppled. Do NOT directly "
            f"push or topple {target_word} yourself.")

        return EnvironmentTask(init_state, goal_atoms, goal_nl=goal_nl)

    def _generate_min_block_task(
            self, task_idx: int,
            rng: np.random.Generator) -> Optional[EnvironmentTask]:
        """Reach-limited "minimum-blocks" task.

        A green start block and a purple target sit a sampled distance apart
        (``CFG.domino_min_block_span_range``), with a generous pile of staged
        blue blocks. The span is chosen so bridging start->target must happen
        near the topple-reach limit; the per-task ``MinBlockReward`` (budget =
        K*) is attached afterwards with physics in the loop — see
        ``min_block_generation._assign_min_blocks``. The goal is simply to
        topple the target, but the reward additionally caps the blues used at
        K*, so a solver that over-estimates reach under-builds and fails.
        Returns ``None`` if no in-bounds placement is found.
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

        staged = self._move_intermediate_objects_to_unfinished_state(obj_dict)
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
        goal_nl = (
            "Move the blue dominoes so that when the green domino is pushed, "
            "the purple domino is toppled — using AS FEW blue dominoes as "
            "possible. Do NOT directly push or topple the purple domino "
            "yourself.")
        return EnvironmentTask(init_state, goal_atoms, goal_nl=goal_nl)

    def _generate_domino_sequence(
            self,
            rng: np.random.Generator,
            n_dominos: int,
            n_targets: int,
            n_pivots: int,
            _log_debug: bool = False,
            task_idx: Optional[int] = None,
            domino_in_upper_half: bool = False) -> Optional[Dict]:
        """Generate a sequence of dominoes, targets, and pivots."""
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

        # Initial position and orientation
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
            while domino_count < expected_count:
                result = self._place_next_domino(
                    rng, obj_dict, x, y, rotation, gap, domino_count,
                    pivot_count, target_count, n_pivots, n_dominos, n_targets,
                    just_placed_target, just_turned_90, _in_bounds, task_idx,
                    block_yaw)
                if not result.success:
                    return None
                x, y, rotation = result.x, result.y, result.rotation
                domino_count = result.domino_count
                pivot_count = result.pivot_count
                just_turned_90 = result.just_turned_90
                block_yaw = (result.block_yaw
                             if result.block_yaw is not None else rotation)
            if domino_count == expected_count and pivot_count == n_pivots:
                return obj_dict
            return None

        # Separate target objects (use_domino_blocks_as_target=False):
        # interleave regular dominoes and target-typed objects.
        while self._should_continue_placement(domino_count, target_count,
                                              n_dominos, n_targets):
            can_place_target = (domino_count >= 2 and target_count < n_targets
                                and not just_placed_target)
            can_place_domino = domino_count < expected_count

            should_place_domino = (not can_place_target
                                   or rng.random() > 0.5) and can_place_domino

            if should_place_domino:
                result = self._place_next_domino(
                    rng, obj_dict, x, y, rotation, gap, domino_count,
                    pivot_count, target_count, n_pivots, n_dominos, n_targets,
                    just_placed_target, just_turned_90, _in_bounds, task_idx)
                if not result.success:
                    return None
                x, y, rotation = result.x, result.y, result.rotation
                domino_count = result.domino_count
                pivot_count = result.pivot_count
                target_count += result.target_count
                just_turned_90 = result.just_turned_90
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

    def _place_next_domino(
            self,
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
            block_yaw: Optional[float] = None) -> PlacementResult:
        """Place the next domino using various strategies."""
        turn_choices = self.domino.turn_choices.copy()
        if pivot_count >= n_pivots and "pivot180" in turn_choices:
            turn_choices.remove("pivot180")
        if just_turned_90 and "turn90" in turn_choices:
            turn_choices.remove("turn90")
        if just_placed_target:
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

    def _move_intermediate_objects_to_unfinished_state(
            self, obj_dict: Dict) -> Optional[Dict]:
        """Move intermediate dominoes and pivots to unfinished positions."""
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

                if not is_start and not is_target:
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

    def _placement_collides(self, obj: Object, candidate: Dict[str, float],
                            occupied: Dict[Object, Dict[str, float]]) -> bool:
        """Check whether ``candidate`` overlaps any occupied object."""
        candidate_rect = self._placement_rect(obj, candidate)
        for other_obj, other_data in occupied.items():
            other_rect = self._placement_rect(other_obj, other_data)
            if self._rectangles_overlap(candidate_rect, other_rect):
                return True
        return False

    def _grasp_clearance_blocked(self, candidate: Dict[str, float],
                                 occupied: Dict[Object, Dict[str, float]],
                                 half_hand: float, half_finger: float) -> bool:
        """Whether the gripper's swept grasp footprint at ``candidate`` would
        overlap another object, leaving the staged domino un-pickable.

        ``half_hand``/``half_finger`` are the gripper footprint half-
        extents along the domino's long axis (local x) and depth/finger-
        span axis (local y). The check is the same oriented-rectangle
        overlap test used for placement, but against the larger gripper
        footprint.
        """
        clear_rect = self._oriented_rect_corners(candidate["x"],
                                                 candidate["y"],
                                                 candidate.get("yaw", 0.0),
                                                 half_hand, half_finger)
        for other_obj, other_data in occupied.items():
            if self._rectangles_overlap(
                    clear_rect, self._placement_rect(other_obj, other_data)):
                return True
        return False

    @staticmethod
    def _oriented_rect_corners(x: float, y: float, yaw: float, half_w: float,
                               half_d: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (center, corners) of an oriented rectangle with the given
        half-extents along its local x (``half_w``) and y (``half_d``) axes."""
        center = np.array([x, y], dtype=np.float64)
        local = np.array(
            [[-half_w, -half_d], [-half_w, half_d], [half_w, half_d],
             [half_w, -half_d]],
            dtype=np.float64,
        )
        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]],
                       dtype=np.float64)
        return center, center + local @ rot.T

    def _placement_rect(
            self, obj: Object,
            data: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Return center and corners for an object's conservative footprint."""
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
        half_w = width / 2 + padding
        half_d = depth / 2 + padding
        yaw = data["yaw"]
        center = np.array([data["x"], data["y"]], dtype=np.float64)
        local = np.array(
            [[-half_w, -half_d], [-half_w, half_d], [half_w, half_d],
             [half_w, -half_d]],
            dtype=np.float64,
        )
        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]],
                       dtype=np.float64)
        return center, center + local @ rot.T

    @staticmethod
    def _rectangles_overlap(rect1: Tuple[np.ndarray, np.ndarray],
                            rect2: Tuple[np.ndarray, np.ndarray]) -> bool:
        """Separating-axis overlap test for two oriented rectangles."""

        def _axes(corners: np.ndarray) -> List[np.ndarray]:
            edges = [corners[1] - corners[0], corners[2] - corners[1]]
            return [edge / np.linalg.norm(edge) for edge in edges]

        _, corners1 = rect1
        _, corners2 = rect2
        for axis in _axes(corners1) + _axes(corners2):
            proj1 = corners1 @ axis
            proj2 = corners2 @ axis
            if max(proj1) <= min(proj2) or max(proj2) <= min(proj1):
                return False
        return True
