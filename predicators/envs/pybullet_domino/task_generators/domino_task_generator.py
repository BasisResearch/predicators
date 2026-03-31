"""Task generator for domino-based tasks."""

from typing import Any, Callable, Dict, List, Optional

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
            obj_dict = self._generate_domino_sequence(rng, n_dominos,
                                                      n_targets, n_pivots,
                                                      log_debug, task_idx,
                                                      domino_in_upper_half)
            if obj_dict is not None:
                if log_debug:
                    print("Found satisfying domino sequence")
                break

        if obj_dict is None:
            return None

        # Move intermediate objects if needed
        if not CFG.domino_initialize_at_finished_state:
            obj_dict = self._move_intermediate_objects_to_unfinished_state(
                obj_dict)

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

        goal_nl = (
            "Arrange the moveable domino blocks into a chain so that when "
            "the start domino is pushed, the chain reaction topples the "
            "target(s). Do NOT directly push "
            "or topple the target dominoes yourself.")

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

        # Main placement loop
        while self._should_continue_placement(domino_count, target_count,
                                              n_dominos, n_targets):
            can_place_target = (domino_count >= 2 and target_count < n_targets
                                and not just_placed_target)
            expected_count = self._get_expected_domino_count(
                n_dominos, n_targets)
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
                           task_idx: Optional[int] = None) -> PlacementResult:
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
                                               task_idx)
        if choice == "turn90":
            return self._place_turn90_domino(rng, obj_dict, x, y, rotation,
                                             gap, domino_count, n_dominos,
                                             n_targets, _in_bounds, task_idx,
                                             should_place_target_at_end)
        if choice == "pivot180":
            return self._place_pivot180_domino(rng, obj_dict, x, y, rotation,
                                               gap, domino_count, pivot_count,
                                               _in_bounds, task_idx,
                                               should_place_target_at_end)
        return self._place_straight_domino(rng, obj_dict, x, y, rotation, gap,
                                           domino_count, _in_bounds, task_idx)

    def _place_straight_domino(self, rng: np.random.Generator,
                               obj_dict: Dict[Object, Any], x: float, y: float,
                               rotation: float, gap: float, domino_count: int,
                               _in_bounds: Callable[[float, float], bool],
                               task_idx: Optional[int]) -> PlacementResult:
        dx = gap * np.sin(rotation)
        dy = gap * np.cos(rotation)
        new_x, new_y = x + dx, y + dy

        if not _in_bounds(new_x, new_y):
            return PlacementResult(success=False,
                                   x=x,
                                   y=y,
                                   rotation=rotation,
                                   domino_count=domino_count)

        obj_dict[self.domino.dominos[domino_count]] = self.domino.place_domino(
            domino_count,
            new_x,
            new_y,
            rotation,
            is_start_block=False,
            rng=rng,
            task_idx=task_idx)

        return PlacementResult(success=True,
                               x=new_x,
                               y=new_y,
                               rotation=rotation,
                               domino_count=domino_count + 1)

    def _place_turn90_domino(
            self, rng: np.random.Generator, obj_dict: Dict[Object, Any],
            x: float, y: float, rotation: float, gap: float, domino_count: int,
            n_dominos: int, n_targets: int,
            _in_bounds: Callable[[float, float],
                                 bool], task_idx: Optional[int],
            should_place_target_at_end: bool) -> PlacementResult:
        expected_count = self._get_expected_domino_count(n_dominos, n_targets)
        if domino_count + 1 >= expected_count:
            return self._place_straight_domino(rng, obj_dict, x, y, rotation,
                                               gap, domino_count, _in_bounds,
                                               task_idx)

        turn_direction = rng.choice([-1, 1])
        dx = gap * np.sin(rotation)
        dy = gap * np.cos(rotation)
        d1_base_x, d1_base_y = x + dx, y + dy
        d1_rot = rotation - turn_direction * np.pi / 4

        shift_magnitude = self.domino.domino_width * self.domino.turn_shift_frac
        shift_dx = shift_magnitude * (turn_direction * np.cos(rotation) -
                                      np.sin(rotation))
        shift_dy = shift_magnitude * (-turn_direction * np.sin(rotation) -
                                      np.cos(rotation))
        d1_x = d1_base_x + shift_dx
        d1_y = d1_base_y + shift_dy

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
            d1_rot,
            is_start_block=False,
            rng=rng,
            task_idx=task_idx)
        domino_count += 1

        d2_rot = d1_rot - turn_direction * np.pi / 4
        sin_d1 = np.sin(d1_rot)
        cos_d1 = np.cos(d1_rot)
        disp_x = (gap * turn_direction * cos_d1 +
                  (2 * shift_magnitude - gap) * sin_d1) / np.sqrt(2)
        disp_y = (-gap * turn_direction * sin_d1 +
                  (2 * shift_magnitude - gap) * cos_d1) / np.sqrt(2)
        d2_x = d1_x + disp_x
        d2_y = d1_y + disp_y

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
            d2_rot,
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
                               just_placed_target=should_place_target_at_end)

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

    def _move_intermediate_objects_to_unfinished_state(self,
                                                       obj_dict: Dict) -> Dict:
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

        start_x = self.domino.domino_x_lb + self.domino.domino_width
        spacing = self.domino.domino_width * 1.5
        y_position = (self.domino.domino_y_lb + self.domino.domino_y_ub) / 2

        for i, (obj, obj_type) in enumerate(intermediate_objects):
            new_x = start_x + i * spacing
            if obj_type == "domino":
                obj_dict[obj] = {
                    "x": new_x,
                    "y": y_position,
                    "z": self.domino.z_lb + self.domino.domino_height / 2,
                    "yaw": 0.0,
                    "roll": 0.0,
                    "r": self.domino.domino_color[0],
                    "g": self.domino.domino_color[1],
                    "b": self.domino.domino_color[2],
                    "is_held": 0.0,
                }
            elif obj_type == "pivot":
                obj_dict[obj] = {
                    "x": new_x,
                    "y": y_position,
                    "z": self.domino.z_lb,
                    "yaw": 0.0,
                }

        return obj_dict
