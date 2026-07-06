"""Towers of Hanoi puzzle. There are three pegs and 3 disks of distinct sizes
stacked on the first peg, largest on the bottom. The goal is to move the stack
to the third peg while never placing a larger disk on top of a smaller one.
"""

from typing import ClassVar, Dict, List, Optional, Sequence, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from matplotlib import patches

from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, GroundAtom, \
    Object, Predicate, State, Type


class HanoiEnv(BaseEnv):
    """Towers of Hanoi domain."""
    # Number of pegs ("places").
    num_pegs: ClassVar[int] = 3
    # Geometry (shares the blocks table frame).
    table_height: ClassVar[float] = 0.4
    x_lb: ClassVar[float] = 1.2
    x_ub: ClassVar[float] = 1.5
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1
    peg_spacing: ClassVar[float] = 0.1
    disk_height: ClassVar[float] = 0.1
    base_z: ClassVar[float] = table_height + disk_height / 2
    pick_z: ClassVar[float] = 0.9
    robot_init_x: ClassVar[float] = (x_lb + x_ub) / 2
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2
    robot_init_z: ClassVar[float] = pick_z
    # Disk widths must stay below peg_spacing so a disk never overlaps a
    # neighboring peg's column when rendered.
    min_disk_width: ClassVar[float] = 0.04
    disk_width_step: ClassVar[float] = 0.02
    # Tolerances / gripper.
    held_tol: ClassVar[float] = 0.5
    pos_tol: ClassVar[float] = 1e-3
    open_fingers: ClassVar[float] = 0.04
    closed_fingers: ClassVar[float] = 0.01

    def __init__(self, use_gui: bool = False) -> None:
        super().__init__(use_gui)

        # Types
        self._disk_type = Type("disk", [
            "pose_x", "pose_z", "held", "width", "color_r", "color_g",
            "color_b"
        ])
        self._peg_type = Type("peg", ["pose_x"])
        self._robot_type = Type("robot",
                                ["pose_x", "pose_y", "pose_z", "fingers"])
        # Predicates
        self._On = Predicate("On", [self._disk_type, self._disk_type],
                             self._On_holds)
        self._OnPeg = Predicate("OnPeg", [self._disk_type, self._peg_type],
                                self._OnPeg_holds)
        self._Clear = Predicate("Clear", [self._disk_type], self._Clear_holds)
        self._ClearPeg = Predicate("ClearPeg", [self._peg_type],
                                   self._ClearPeg_holds)
        self._Holding = Predicate("Holding", [self._disk_type],
                                  self._Holding_holds)
        self._GripperOpen = Predicate("GripperOpen", [self._robot_type],
                                      self._GripperOpen_holds)
        self._Smaller = Predicate("Smaller",
                                  [self._disk_type, self._disk_type],
                                  self._Smaller_holds)
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        self._pegs = [
            Object(f"peg{i}", self._peg_type) for i in range(3)
        ]
        self._disks: List[Object] = []
        self._create_disks()
        # Hyperparameters from CFG.
        self._num_disks_train = CFG.hanoi_num_disks_train
        self._num_disks_test = CFG.hanoi_num_disks_test

    @classmethod
    def get_name(cls) -> str:
        return "hanoi"

    @classmethod
    def _peg_x(cls, peg_idx: int) -> float:
        """The x coordinate of the peg with the given index."""
        return [1.25, 1.35, 1.45][peg_idx]

    def _disk_width(self, disk_idx: int) -> float:
        """The width of the disk with the given index (index 0 is smallest)."""
        return self.min_disk_width + disk_idx * self.disk_width_step

    def _create_disks(self) -> None:
        num_disks = max(max(CFG.hanoi_num_disks_train),
                        max(CFG.hanoi_num_disks_test))
        for i in range(num_disks):
            disk = Object(f"disk{i}", self._disk_type)
            self._disks.append(disk)

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        # The z coordinate of the action is unused: a pick always grabs the
        # topmost disk in its column, and a place always drops onto the top of
        # the target column's stack.
        x, _, fingers = action.arr
        # Infer which transition to follow based on whether the fingers value
        # is closer to closed (pick) or open (place).
        fingers_closing = abs(fingers - self.closed_fingers) < \
            abs(fingers - self.open_fingers)
        if fingers_closing:
            return self._transition_pick(state, x)
        return self._transition_place(state, x)

    def _transition_pick(self, state: State, x: float) -> State:
        next_state = state.copy()
        # Can only pick if the gripper is currently open.
        if not self._GripperOpen_holds(state, [self._robot]):
            return next_state
        peg_x = self._snap_to_peg_x(x)
        disk = self._get_top_disk_at_x(state, peg_x)
        if disk is None:  # no disk in this column
            return next_state
        # Execute pick: lift the disk up.
        next_state.set(disk, "pose_x", peg_x)
        next_state.set(disk, "pose_z", self.pick_z)
        next_state.set(disk, "held", 1.0)
        next_state.set(self._robot, "fingers", self.closed_fingers)
        return next_state

    def _transition_place(self, state: State, x: float) -> State:
        next_state = state.copy()
        # Can only place if the gripper is currently holding a disk.
        if self._GripperOpen_holds(state, [self._robot]):
            return next_state
        disk = self._get_held_disk(state)
        assert disk is not None
        peg_x = self._snap_to_peg_x(x)
        top_disk = self._get_top_disk_at_x(state, peg_x)
        if top_disk is None:
            # Empty peg: place the disk directly on the peg.
            target_z = self.base_z
        else:
            # Towers of Hanoi rule: can only place onto a strictly larger disk.
            if state.get(disk, "width") >= state.get(top_disk, "width"):
                return next_state  # illegal move; no-op
            target_z = state.get(top_disk, "pose_z") + self.disk_height
        # Execute place.
        next_state.set(disk, "pose_x", peg_x)
        next_state.set(disk, "pose_z", target_z)
        next_state.set(disk, "held", 0.0)
        next_state.set(self._robot, "fingers", self.open_fingers)
        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num_tasks=CFG.num_train_tasks,
                               possible_num_disks=self._num_disks_train,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks,
                               possible_num_disks=self._num_disks_test,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._On, self._OnPeg, self._Clear, self._ClearPeg, self._Holding,
            self._GripperOpen, self._Smaller
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._On, self._OnPeg}

    @property
    def types(self) -> Set[Type]:
        return {self._disk_type, self._peg_type, self._robot_type}

    @property
    def action_space(self) -> Box:
        # dimensions: [x, z, fingers]
        x_lb = self._peg_x(0) - self.peg_spacing / 2
        x_ub = self._peg_x(self.num_pegs - 1) + self.peg_spacing / 2
        lowers = np.array([x_lb, 0.0, self.closed_fingers], dtype=np.float32)
        uppers = np.array([x_ub, self.pick_z + self.disk_height,
                           self.open_fingers],
                          dtype=np.float32)
        return Box(lowers, uppers)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        del task, action  # unused
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        x_lb = self._peg_x(0) - self.peg_spacing / 2
        x_ub = self._peg_x(self.num_pegs - 1) + self.peg_spacing / 2
        ax.set_xlim(x_lb, x_ub)
        ax.set_ylim(0, self.pick_z + 2 * self.disk_height)
        ax.set_xlabel("x", fontsize=16)
        ax.set_ylabel("z", fontsize=16)

        # Draw the pegs as vertical lines.
        for peg in self._pegs:
            if peg not in state:
                continue
            px = state.get(peg, "pose_x")
            ax.plot([px, px], [0.0, self.pick_z],
                    color="black",
                    linewidth=2,
                    zorder=0)

        # Draw the disks as rectangles centered on their poses.
        held = "None"
        disks = [o for o in state if o.is_instance(self._disk_type)]
        for disk in sorted(disks):
            x = state.get(disk, "pose_x")
            z = state.get(disk, "pose_z")
            w = state.get(disk, "width")
            color = (state.get(disk, "color_r"), state.get(disk, "color_g"),
                     state.get(disk, "color_b"))
            if state.get(disk, "held") > self.held_tol:
                held = disk.name
            rect = patches.Rectangle((x - w / 2, z - self.disk_height / 2),
                                     w,
                                     self.disk_height,
                                     linewidth=1,
                                     edgecolor="black",
                                     facecolor=color,
                                     zorder=1)
            ax.add_patch(rect)

        title = f"Held: {held}"
        if caption is not None:
            title += f"; {caption}"
        plt.suptitle(title, fontsize=16, wrap=True)
        plt.tight_layout()
        return fig

    def _get_tasks(self, num_tasks: int, possible_num_disks: List[int],
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []
        target_peg_idx = self.num_pegs - 1  # the "third" (final) place
        for _ in range(num_tasks):
            num_disks = int(rng.choice(possible_num_disks))
            # Start the stack on a peg other than the target peg.
            source_peg_idx = int(rng.choice(target_peg_idx))
            init_state = self._sample_state(num_disks, source_peg_idx)
            goal = self._make_stack_goal(num_disks, target_peg_idx)
            assert not all(atom.holds(init_state) for atom in goal)
            tasks.append(EnvironmentTask(init_state, goal))
        return tasks

    def _sample_state(self, num_disks: int, source_peg_idx: int) -> State:
        data: Dict[Object, Array] = {}
        # Distinct colors per disk (evenly spaced hues via a simple palette).
        palette = [
            (0.85, 0.32, 0.31),  # red
            (0.34, 0.63, 0.83),  # blue
            (0.45, 0.72, 0.42),  # green
            (0.90, 0.68, 0.29),  # orange
            (0.60, 0.44, 0.78),  # purple
        ]
        peg_x = self._peg_x(source_peg_idx)
        # Stack the disks with the largest on the bottom: the disk with the
        # highest index (widest) sits at the base, disk0 (narrowest) on top.
        for stack_level, disk_idx in enumerate(reversed(range(num_disks))):
            disk = self._disks[disk_idx]
            z = self.base_z + stack_level * self.disk_height
            r, g, b = palette[disk_idx % len(palette)]
            width = self._disk_width(disk_idx)
            # [pose_x, pose_z, held, width, color_r, color_g, color_b]
            data[disk] = np.array([peg_x, z, 0.0, width, r, g, b])
        # Pegs (static positions).
        for peg_idx, peg in enumerate(self._pegs):
            data[peg] = np.array([self._peg_x(peg_idx)])
        # Robot: fingers start out open.
        data[self._robot] = np.array([
            self.robot_init_x, self.robot_init_y, self.robot_init_z,
            self.open_fingers
        ])
        return State(data)

    def _make_stack_goal(self, num_disks: int,
                         target_peg_idx: int) -> Set[GroundAtom]:
        # Goal: full stack (largest on the bottom) on the target peg.
        target_peg = self._pegs[target_peg_idx]
        goal = set()
        bottom_disk = self._disks[num_disks - 1]
        goal.add(GroundAtom(self._OnPeg, [bottom_disk, target_peg]))
        # disk i on top of disk i+1 for i in [0, num_disks - 1).
        for i in range(num_disks - 1):
            goal.add(GroundAtom(self._On, [self._disks[i], self._disks[i + 1]]))
        return goal

    def _snap_to_peg_x(self, x: float) -> float:
        """Snap a continuous x coordinate to the nearest peg's x coordinate."""
        peg_xs = [self._peg_x(i) for i in range(self.num_pegs)]
        return min(peg_xs, key=lambda px: abs(px - x))

    def _get_disks_at_x(self, state: State, x: float) -> List[Object]:
        """All (non-held) disks whose column matches the given x coordinate."""
        disks = []
        for disk in state:
            if not disk.is_instance(self._disk_type):
                continue
            if state.get(disk, "held") >= self.held_tol:
                continue
            if abs(state.get(disk, "pose_x") - x) < self.pos_tol:
                disks.append(disk)
        return disks

    def _get_top_disk_at_x(self, state: State, x: float) -> Optional[Object]:
        """The highest (topmost) non-held disk in the column at x, if any."""
        disks = self._get_disks_at_x(state, x)
        if not disks:
            return None
        return max(disks, key=lambda d: state.get(d, "pose_z"))

    def _get_held_disk(self, state: State) -> Optional[Object]:
        for disk in state:
            if not disk.is_instance(self._disk_type):
                continue
            if state.get(disk, "held") >= self.held_tol:
                return disk
        return None

    def _On_holds(self, state: State, objects: Sequence[Object]) -> bool:
        disk1, disk2 = objects
        if state.get(disk1, "held") >= self.held_tol or \
           state.get(disk2, "held") >= self.held_tol:
            return False
        x1 = state.get(disk1, "pose_x")
        z1 = state.get(disk1, "pose_z")
        x2 = state.get(disk2, "pose_x")
        z2 = state.get(disk2, "pose_z")
        return abs(x1 - x2) < self.pos_tol and \
            abs(z1 - (z2 + self.disk_height)) < self.pos_tol

    def _OnPeg_holds(self, state: State, objects: Sequence[Object]) -> bool:
        disk, peg = objects
        if state.get(disk, "held") >= self.held_tol:
            return False
        x = state.get(disk, "pose_x")
        z = state.get(disk, "pose_z")
        peg_x = state.get(peg, "pose_x")
        return abs(x - peg_x) < self.pos_tol and \
            abs(z - self.base_z) < self.pos_tol

    def _Clear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        disk, = objects
        if state.get(disk, "held") >= self.held_tol:
            return False
        for other in state:
            if other.type != self._disk_type or other == disk:
                continue
            if self._On_holds(state, [other, disk]):
                return False
        return True

    def _ClearPeg_holds(self, state: State, objects: Sequence[Object]) -> bool:
        peg, = objects
        peg_x = state.get(peg, "pose_x")
        return not self._get_disks_at_x(state, peg_x)

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        disk, = objects
        return state.get(disk, "held") >= self.held_tol

    def _GripperOpen_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        robot, = objects
        rf = state.get(robot, "fingers")
        return abs(rf - self.open_fingers) < abs(rf - self.closed_fingers)

    def _Smaller_holds(self, state: State, objects: Sequence[Object]) -> bool:
        disk1, disk2 = objects
        return state.get(disk1, "width") < state.get(disk2, "width")
