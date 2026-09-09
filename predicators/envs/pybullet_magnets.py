"""The magnets environment: the field law, tasks, predicates.

The observable simulation core (mat, wand, pieces, slots) lives in
:mod:`predicators.envs.pybullet_magnets_base`, which may be surfaced
to learning agents as reference source. This module holds everything
an agent must LEARN or must not see:

* the FIELD - each colour's polarity (the wand pulls it or pushes it)
  and its range, and how fast a piece inside the range moves: toward
  the point under the wand's tip, or away from it;
* task generation (the train/test distribution) and goal semantics.

Design notes, and why this domain favours a model of the physics.

**Action at a distance.** The wand never touches a piece. Hovering it
somewhere moves every piece within that colour's range, at once, and
the pieces it repels go the other way. Carrying a piece to its slot
means hovering over it and sliding slowly enough that it follows;
parking it means leaving fast enough that it does not. A path that
passes within range of another piece drags that piece too, and a
repelled piece near the mat's edge is pushed off it for good.

**Per colour, a sign and a distance.** The hidden content per colour
is a polarity and a range, plus one speed law they share. An agent
that has identified them from a few hovers rolls the base simulator
forward and sees, for a whole hover path, which pieces move where and
which fall off. An agent reasoning from a table of hovers has to try
the path, and the piece it loses is lost.

**Test extends train.** Test mats carry more pieces, including a
colour training never showed near a slot it must reach; the colours
keep their polarity and range.
"""
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from predicators import utils
from predicators.code_sim_learning.commands import PhysicsCommand, SetVelocity
from predicators.envs.pybullet_magnets_base import PyBulletMagnetsBaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, StepOption, TaskEvaluator, Type


# =============================================================================
# CLASSIFIERS
# =============================================================================
def piece_in_slot(state: State, piece: Object, slot: Object) -> bool:
    """Whether ``piece`` rests within the goal tolerance of ``slot``."""
    dx = abs(state.get(piece, "x") - state.get(slot, "x"))
    dy = abs(state.get(piece, "y") - state.get(slot, "y"))
    return dx < CFG.magnets_in_tol and dy < CFG.magnets_in_tol


def piece_lost(state: State, piece: Object) -> bool:
    """Whether ``piece`` has left the mat."""
    return not PyBulletMagnetsBaseEnv.on_mat(state.get(piece, "x"),
                                             state.get(piece, "y"))


def piece_at_rest(state: State, piece: Object) -> bool:
    """Whether ``piece`` has stopped moving."""
    return state.get(piece, "speed") < CFG.magnets_settle_speed


def tip_over(state: State, wand: Object, piece: Object) -> bool:
    """Whether the wand's tip hovers over the piece."""
    tx, ty, _ = PyBulletMagnetsBaseEnv.tip_position(state, wand)
    return bool(
        np.hypot(tx - state.get(piece, "x"), ty -
                 state.get(piece, "y")) < CFG.magnets_over_tol)


class MagnetsEvaluator(TaskEvaluator):
    """Win when every goal piece rests in its slot; lose a piece, lose the
    level."""

    def terminated(self, state: State) -> bool:
        pieces = [o for o in state if o.type.name == "piece"]
        if any(piece_lost(state, pc) for pc in pieces):
            return True
        if not all(atom.holds(state) for atom in self.goal):
            return False
        return all(piece_at_rest(state, pc) for pc in pieces)

    def _certify(self,
                 states: Sequence[State],
                 step_options: Optional[Sequence[StepOption]],
                 sim_env: Optional[Any] = None) -> Tuple[bool, str]:
        del step_options, sim_env  # unused
        final = states[-1]
        for piece in sorted((o for o in final if o.type.name == "piece"),
                            key=lambda o: o.name):
            if piece_lost(final, piece):
                return False, f"{piece.name} slid off the mat"
        return True, ""

    def objective_description(self) -> str:
        return ("The level is won when every piece with a slot rests in the "
                "slot of its colour. A piece that leaves the mat cannot be "
                "recovered and ends the level.")


class PyBulletMagnetsEnv(PyBulletMagnetsBaseEnv):
    """A magnet puzzle whose colours' polarity and range must be learned."""

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        super().__init__(use_gui, **kwargs)
        self._In = Predicate("In", [self._piece_type, self._slot_type],
                             self._In_holds,
                             natural_language_assertion=lambda os:
                             f"piece {os[0]} rests in slot {os[1]}")
        self._OnMat = Predicate("OnMat", [self._piece_type],
                                self._OnMat_holds,
                                natural_language_assertion=lambda os:
                                f"piece {os[0]} is on the mat")
        self._Lost = Predicate("Lost", [self._piece_type],
                               self._Lost_holds,
                               natural_language_assertion=lambda os:
                               f"piece {os[0]} has slid off the mat")
        self._AtRest = Predicate("AtRest", [self._piece_type],
                                 self._AtRest_holds,
                                 natural_language_assertion=lambda os:
                                 f"piece {os[0]} is not moving")
        self._TipOver = Predicate(
            "TipOver", [self._wand_type, self._piece_type],
            self._TipOver_holds,
            natural_language_assertion=lambda os:
            f"the tip of {os[0]} hovers over piece {os[1]}")
        self._Holding = Predicate(
            "Holding", [self._robot_type, self._wand_type],
            self._Holding_holds,
            natural_language_assertion=lambda os: f"{os[0]} holds {os[1]}")

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_magnets"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._In, self._OnMat, self._Lost, self._AtRest, self._TipOver,
            self._Holding
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._In}

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._wand_type, self._piece_type,
            self._slot_type
        }

    # =========================================================================
    # HIDDEN FIELD
    # =========================================================================
    @classmethod
    def polarity(cls, color_index: int) -> int:
        """+1 if the wand pulls this colour, -1 if it pushes it."""
        return int(CFG.magnets_polarities[int(round(color_index))])

    @classmethod
    def field_range(cls, color_index: int) -> float:
        """How far from the tip's ground point this colour still moves."""
        return float(CFG.magnets_ranges[int(round(color_index))])

    def _domain_specific_step(self) -> None:
        """The field: every piece inside its colour's range moves along the
        line to the point under the tip, toward it or away from it, at a speed
        that falls linearly to zero at the range's edge and is zero right under
        the tip.

        A velocity command held for the next action, queued through the
        same channel a learned rule uses.
        """
        state = self._get_state()
        if self._wand not in state.data:
            return
        tx, ty, tz = self.tip_position(state, self._wand)
        commands: List[PhysicsCommand] = []
        for piece in self._pieces:
            if piece.name not in self._piece_colors or piece not in state.data:
                continue
            px, py, pz = (state.get(piece, "x"), state.get(piece, "y"),
                          state.get(piece, "z"))
            if tz - pz > CFG.magnets_field_height or tz < pz:
                continue
            color = self._piece_colors[piece.name]
            reach = self.field_range(color)
            dx, dy = tx - px, ty - py
            r = float(np.hypot(dx, dy))
            if r >= reach or r < CFG.magnets_dead_zone:
                continue
            speed = float(CFG.magnets_max_speed) * (1.0 - r / reach)
            sign = self.polarity(color)
            vx, vy = sign * speed * dx / r, sign * speed * dy / r
            commands.append(SetVelocity(piece.name, (vx, vy, 0.0), None))
        if commands:
            self.queue_residual_commands(commands)

    # =========================================================================
    # PREDICATES
    # =========================================================================
    @staticmethod
    def _In_holds(state: State, objects: Sequence[Object]) -> bool:
        piece, slot = objects
        return piece_in_slot(state, piece, slot)

    @staticmethod
    def _OnMat_holds(state: State, objects: Sequence[Object]) -> bool:
        piece, = objects
        return not piece_lost(state, piece)

    @staticmethod
    def _Lost_holds(state: State, objects: Sequence[Object]) -> bool:
        piece, = objects
        return piece_lost(state, piece)

    @staticmethod
    def _AtRest_holds(state: State, objects: Sequence[Object]) -> bool:
        piece, = objects
        return piece_at_rest(state, piece)

    @staticmethod
    def _TipOver_holds(state: State, objects: Sequence[Object]) -> bool:
        wand, piece = objects
        return tip_over(state, wand, piece)

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        _, wand = objects
        return state.get(wand, "is_held") > 0.5

    # =========================================================================
    # TASK GENERATION
    # =========================================================================
    _grid_margin: ClassVar[float] = 0.06
    _spacing: ClassVar[float] = 0.14

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks,
                                rng=self._train_rng,
                                train=True)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks,
                                rng=self._test_rng,
                                train=False)

    def _robot_init_dict(self) -> Dict[str, float]:
        return {
            "x": self.robot_init_x,
            "y": self.robot_init_y,
            "z": self.robot_init_z,
            "fingers": self.closed_fingers,
            "roll": self.robot_init_roll,
            "tilt": self.robot_init_tilt,
            "wrist": self.robot_init_wrist,
        }

    def level_state(self, pieces: Sequence[Tuple[float, float, int]],
                    slots: Sequence[Tuple[float, float, int]]) -> State:
        """A level: pieces at ``(x, y, color)``, slots likewise, the wand in
        hand at its home hover."""
        init: Dict[Object, Dict[str, float]] = {
            self._robot: self._robot_init_dict()
        }
        init[self._wand] = {
            "x": self.robot_init_x,
            "y": self.robot_init_y,
            "z": self.robot_init_z - self.wand_length / 2,
            "is_held": 1.0,
        }
        for i, (x, y, color) in enumerate(pieces):
            init[self._pieces[i]] = {
                "x": x,
                "y": y,
                "z": self.piece_z,
                "rot": 0.0,
                "color": float(color),
                "speed": 0.0,
            }
        for i, (x, y, color) in enumerate(slots):
            init[self._slots[i]] = {
                "x": x,
                "y": y,
                "z": self.slot_z,
                "color": float(color),
            }
        return utils.create_state_from_dict(init)

    def _sample_positions(
            self, count: int,
            rng: np.random.Generator) -> List[Tuple[float, float]]:
        x_min, x_max, y_min, y_max = self.mat_bounds()
        m = self._grid_margin
        out: List[Tuple[float, float]] = []
        for _ in range(500):
            x = float(rng.uniform(x_min + m, x_max - m))
            y = float(rng.uniform(y_min + m, y_max - m))
            if all(
                    np.hypot(x - ox, y - oy) >= self._spacing
                    for ox, oy in out):
                out.append((x, y))
            if len(out) == count:
                return out
        return out

    def _make_tasks(self, num_tasks: int, rng: np.random.Generator,
                    train: bool) -> List[EnvironmentTask]:
        # pylint: disable-next=import-outside-toplevel
        from predicators.ground_truth_models.magnets.oracle import solve_level
        counts = list(CFG.magnets_num_pieces_train if train else CFG.
                      magnets_num_pieces_test)
        tasks = []
        for _ in range(num_tasks):
            num_pieces = int(rng.choice(counts))
            found = None
            for _ in range(int(CFG.magnets_max_sampling_attempts)):
                colors = [
                    int(c) for c in rng.choice(len(self.COLOR_PALETTE),
                                               size=num_pieces,
                                               replace=False)
                ]
                attracted = [
                    i for i, c in enumerate(colors) if self.polarity(c) > 0
                ]
                if not attracted:
                    continue
                positions = self._sample_positions(num_pieces + len(attracted),
                                                   rng)
                if len(positions) < num_pieces + len(attracted):
                    continue
                pieces = [(x, y, c)
                          for (x, y), c in zip(positions[:num_pieces], colors)]
                slots = [(x, y, colors[i])
                         for (x,
                              y), i in zip(positions[num_pieces:], attracted)]
                state = self.level_state(pieces, slots)
                if solve_level(self, state) is not None:
                    found = state
                    break
            if found is None:
                raise RuntimeError(
                    f"No {num_pieces}-piece mat the oracle can clear in "
                    f"{CFG.magnets_max_sampling_attempts} draws.")
            goal = self.goal_for(found)
            pieces_, slots_ = self._active_objects(found)
            parts = [
                f"the {self.color_name(found.get(s, 'color'))} piece into the "
                f"{self.color_name(found.get(s, 'color'))} slot ({s.name})"
                for s in slots_
            ]
            others = [
                pc.name for pc in pieces_
                if self.polarity(found.get(pc, "color")) < 0
            ]
            goal_nl = "Using the wand in your hand, bring " + ", ".join(
                parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
            goal_nl += "; every piece must stay on the mat."
            if others:
                goal_nl += (
                    f" The other piece{'s' if len(others) > 1 else ''} "
                    f"({', '.join(others)}) need not go anywhere.")
            tasks.append(
                EnvironmentTask(found,
                                goal,
                                goal_nl=goal_nl,
                                evaluator=MagnetsEvaluator(goal)))
        return self._add_pybullet_state_to_tasks(tasks)

    def goal_for(self, state: State) -> Set[GroundAtom]:
        """Each slot filled by the piece of its colour."""
        pieces, slots = self._active_objects(state)
        goal = set()
        for slot in slots:
            color = int(round(state.get(slot, "color")))
            piece = next(pc for pc in pieces
                         if int(round(state.get(pc, "color"))) == color)
            goal.add(GroundAtom(self._In, [piece, slot]))
        return goal

    # =========================================================================
    # PROBE
    # =========================================================================
    def run_option(self, state: State, option: Any,
                   max_steps: int) -> Optional[State]:
        """Execute a grounded option from ``state`` on this instance and hold
        until the pieces settle; None if it cannot run."""
        self._pybullet_robot.set_joints(
            self._pybullet_robot.initial_joint_positions)
        self._set_state(state)
        obs = self._get_state()
        self._current_observation = obs
        if not option.initiable(obs):
            return None
        try:
            for _ in range(max_steps):
                if option.terminal(obs):
                    break
                obs = self._step_once(option.policy(obs))
        except utils.OptionExecutionFailure:
            return None
        hold = Action(
            np.array(self._pybullet_robot.get_joints(), dtype=np.float32))
        pieces, _ = self._active_objects(state)
        for _ in range(max_steps):
            if all(
                    self._piece_speed(pc) < CFG.magnets_settle_speed
                    for pc in pieces):
                break
            obs = self._step_once(hold)
        return self._get_state()


_PROBE_ENV: Optional[PyBulletMagnetsEnv] = None


def probe_env() -> PyBulletMagnetsEnv:
    """A dedicated instance for probes, so a helper never disturbs the live
    episode's world."""
    global _PROBE_ENV  # pylint: disable=global-statement
    if _PROBE_ENV is None:
        _PROBE_ENV = PyBulletMagnetsEnv(use_gui=False)
    return _PROBE_ENV
