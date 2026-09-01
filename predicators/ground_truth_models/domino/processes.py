"""Ground-truth processes for the domino environment."""

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch

from predicators.ground_truth_models import GroundTruthProcessFactory, \
    GroundTruthSamplerFactory
from predicators.settings import CFG
from predicators.structs import Array, CausalProcess, EndogenousProcess, \
    ExogenousProcess, GroundAtom, LiftedAtom, Object, ParameterizedOption, \
    ParameterizedSampler, Predicate, State, Type, Variable
from predicators.utils import ConstantDelay, DiscreteGaussianDelay, \
    null_sampler, wrap_angle

# Fixed parameter values for domino environment. Both z offsets were tuned on
# the Fetch; see _hand_z_correction for what that means on another arm.
_DOMINO_GRASP_Z_OFFSET = 0.0825  # domino_height * 0.55
# Slightly above the legacy drop height. With the skill-factory Pick grasp
# transform, 0.5695 leaves the held domino penetrating the table at the
# collision-aware Place goal; 0.58 clears the table and still settles to the
# intended upright pose.
_DOMINO_DROP_Z = 0.58
_DOMINO_OFFSET_Z = 0.0825  # domino_height * 0.55

# How far each hand reaches below its tool frame, measured at home as the
# lowest finger-link AABB against the tool link.
_FINGERTIP_REACH_BELOW_TOOL = {
    "fetch": 0.0320,
    "mobile_fetch": 0.0320,
    "panda": 0.0152,
}


def _hand_z_correction() -> float:
    """How much lower to command the tool frame on a shorter-fingered hand.

    The z offsets above position the TOOL frame, but what has to clear or
    contact the domino is the hand hanging below it -- and the Fetch reaches
    1.68cm further down than the Panda. Left uncorrected, the same number puts
    the Fetch's fingertips at 84% of a 0.15m domino's height and the Panda's at
    95%, the very top edge: the Panda barely catches the top on a grasp and
    skims over it on a push.

    Zero for the Fetch, so every value tuned on it is preserved exactly.
    """
    fetch_reach = _FINGERTIP_REACH_BELOW_TOOL["fetch"]
    return fetch_reach - _FINGERTIP_REACH_BELOW_TOOL.get(
        CFG.pybullet_robot, fetch_reach)


def _domino_depth() -> float:
    """Thickness of the domino actually in play, in metres.

    ``pybullet_domino_real`` sizes its component from
    ``CFG.domino_real_domino_dims`` (L, W, thickness); every other
    domino env takes the class ClassVar.
    """
    # pylint: disable=import-outside-toplevel  # local: avoid import cycle
    from predicators.envs.pybullet_domino import PyBulletDominoEnv
    if CFG.env == "pybullet_domino_real":
        return float(CFG.domino_real_domino_dims[2])
    return float(PyBulletDominoEnv.domino_depth)


def _push_approach_distance() -> float:
    """How far behind the block the gripper descends before pushing it."""
    return 3.0 * _domino_depth()


def _grasp_z_offset() -> float:
    """Pick grasp height, corrected for the hand in use."""
    return _DOMINO_GRASP_Z_OFFSET - _hand_z_correction()


def _push_contact_z_offset() -> float:
    """Push contact height, corrected for the hand in use."""
    return _DOMINO_OFFSET_Z - _hand_z_correction()


def _pick_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    """Return fixed grasp_z_offset for domino pick."""
    del state, goal, rng, objs
    return np.array([_grasp_z_offset()], dtype=np.float32)


def _push_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    """Return fixed push params for domino push."""
    if not CFG.domino_use_skill_factories:
        return np.array([], dtype=np.float32)
    del state, goal, rng, objs
    return np.array([_push_approach_distance(),
                     _push_contact_z_offset()],
                    dtype=np.float32)


def _declare_sampler(state: State, goal: Set[GroundAtom],
                     rng: np.random.Generator,
                     objs: Sequence[Object]) -> Array:
    """No parameters: a declaration has nothing to aim.

    Its option's params_space is empty, so an empty array is what the
    option expects. Written out rather than reaching for null_sampler
    to keep the contrast with _switch_push_sampler on the page: the
    press needs an approach distance and a contact offset because it
    has to arrive somewhere, and this does not.
    """
    del state, goal, rng, objs
    return np.array([], dtype=np.float32)


def _switch_push_sampler(state: State, goal: Set[GroundAtom],
                         rng: np.random.Generator,
                         objs: Sequence[Object]) -> Array:
    """Approach distance and contact offset for pressing a switch.

    TurnFanOn is a push skill, so it wants the same two params every
    push does - null_sampler hands it an empty array and the option's
    clip against a 2-vector bounds raises. The values are
    fan/processes.py's, measured there against this same switch model:
    0.075 clears an end-of-row switch on the approach without stalling
    at the arm's reach on the far-side press.
    """
    del state, goal, rng, objs
    if not CFG.domino_use_skill_factories:
        return np.array([], dtype=np.float32)
    return np.array([0.075, 0.1], dtype=np.float32)


def _place_sampler(state: State, goal: Set[GroundAtom],
                   rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    """Return a generator-faithful placement for the open-loop oracle.

    ``objs = [robot, domino1, domino2, target_pos, rotation]``. The process
    planner picks a discrete grid cell (``target_pos``) and angle (``rotation``)
    for the held ``domino1`` next to the reference ``domino2``. The grid is a
    uniform lattice (see ``augment_task_with_helper_objects``), so a turn block
    lands at the *same* cell a straight block would, differing only in angle --
    the generator's inward ``domino_width/2`` corner offset is absent from the
    lattice. Placing the held domino at the bare cell stalls corner cascades.

    Instead pick from the placements the generator would lay next to ``domino2``
    (``_generator_placements``, which carry the corner offset), rank-summing
    three signals that each, alone, mishandle one case -- future-target bridge
    (greedy: pulls a straight run onto the target), grid-cell distance (a
    uniform-grid turn cell sits on the straight position, missing corners), and
    angle error (the planner stamps spurious turn angles on straight runs). The
    cascade-correct candidate is top-ranked on >=2 of the three. Deterministic;
    final tiebreak is the planner's cell; bare cell if no candidate at all.
    """
    if not CFG.domino_use_skill_factories:
        return np.array([], dtype=np.float32)
    del goal, rng
    # objs = [robot, domino1, domino2, target_pos, rotation]
    held = objs[1]
    ref = objs[2]
    target_pos = objs[3]
    rotation = objs[4]
    gx = float(target_pos.name.split("_")[1])
    gy = float(target_pos.name.split("_")[2])
    gyaw = np.radians(float(rotation.name.split("_")[-1]))

    rx = state.get(ref, "x")
    ry = state.get(ref, "y")
    ryaw = state.get(ref, "yaw")
    candidates = _generator_placements(rx, ry, ryaw)
    if not candidates:
        # Fallback: bare lattice cell (no generator candidate available).
        return np.array([gx, gy, _DOMINO_DROP_Z, gyaw], dtype=np.float32)
    bridges = [
        _future_target_bridge_score(state, held, c[0], c[1], c[2])
        for c in candidates
    ]
    dgrids = [float(np.hypot(c[0] - gx, c[1] - gy)) for c in candidates]
    angerrs = [abs(wrap_angle(c[2] - gyaw)) for c in candidates]

    def _rank(vals: List[float], i: int, higher_better: bool = False) -> int:
        # Number of candidates strictly better than ``i`` (ties share a rank).
        if higher_better:
            return sum(1 for v in vals if v > vals[i] + 1e-9)
        return sum(1 for v in vals if v < vals[i] - 1e-9)

    def _total(i: int) -> Tuple[int, float]:
        rank_sum = (_rank(bridges, i, higher_better=True) + _rank(dgrids, i) +
                    _rank(angerrs, i))
        return (rank_sum, dgrids[i])

    best_i = min(range(len(candidates)), key=_total)
    cx, cy, cyaw = candidates[best_i]
    return np.array([cx, cy, _DOMINO_DROP_Z, cyaw], dtype=np.float32)


class PyBulletDominoGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the domino grid environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "pybullet_domino_grid", "pybullet_domino", "pybullet_domino_real",
            "pybullet_domino_real_geometry", "pybullet_domino_fan",
            "pybullet_domino_declare",
            "pybullet_domino_blow"
        }

    @classmethod
    def get_processes(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        if env_name == "pybullet_domino_blow":
            # A different task, so a different model rather than the
            # cascade one with pieces disabled: no chain, no topple, no
            # grid. One block, one gust, one patch to land it in.
            return _get_blow_processes(types, predicates, options)

        # These processes are defined over the grid (loc/angle/direction).
        # Only oracle / process-planning approaches request them, and they do
        # so unconditionally, so the grid is intrinsic to those approaches.

        # Types
        robot_type = types["robot"]
        domino_type = types["domino"]
        position_type = types["loc"]
        rotation_type = types["angle"]

        # Predicates
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        InFront = predicates["InFront"]
        Upright = predicates["Upright"]
        StartBlock = predicates["InitialBlock"]
        Toppled = predicates["Toppled"]
        Tilting = predicates["Tilting"]
        DominoAtPos = predicates["DominoAtPos"]
        DominoAtRot = predicates["DominoAtRot"]
        MovableBlock = predicates["MovableBlock"]
        PosClear = predicates["PosClear"]
        AdjacentTo = predicates["AdjacentTo"]
        if CFG.domino_has_glued_dominos:
            DominoNotGlued = predicates["DominoNotGlued"]
        # Note: Tilting predicate exists but represents the goal state
        # Note: The "Falling" predicate from the sketch is not implemented in the current environment  # pylint: disable=line-too-long
        # We would need to add it to the environment for the DominoFall
        # exogenous process

        # Options. Push is absent in a fan env (see below), so it is
        # looked up defensively rather than by subscript.
        Push = options.get("Push")
        Pick = options["Pick"]
        Place = options["Place"]
        Wait = options["Wait"]

        processes: Set[CausalProcess] = set()

        # --- Endogenous Processes / Actions ---

        # PushStartBlock: Push the start block to initiate the domino chain
        robot = Variable("?robot", robot_type)
        domino = Variable("?domino", domino_type)
        parameters = [robot, domino]
        # With restricted push the "Push" option finds the start block from
        # the state itself, so it takes only the robot. The unrestricted
        # option also takes the domino to push.
        if CFG.domino_restricted_push:
            option_vars = [robot]
        else:
            option_vars = [robot, domino]
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(StartBlock, [domino]),
            LiftedAtom(Upright, [domino]),
        }
        add_effects = {
            LiftedAtom(Tilting, [domino]),
        }
        delete_effects: Set[LiftedAtom] = {
            LiftedAtom(Upright, [domino]),
        }
        ignore_effects = {DominoAtPos, DominoAtRot, PosClear, AdjacentTo}
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                                   sigma=torch.tensor(0.1))
        if Push is not None:
            push_start_block_process = EndogenousProcess(
                "PushStartBlock", parameters, condition_at_start, set(), set(),
                add_effects, delete_effects, delay_distribution,
                torch.tensor(1.0), Push, option_vars, _push_sampler,
                ignore_effects)
            # Withheld in a fan env, matching the option: the wind starts
            # the chain there, and a planner left a Push will use it and
            # never touch a switch.
            processes.add(push_start_block_process)

        # PickDomino: Position-based pick process
        robot = Variable("?robot", robot_type)
        domino = Variable("?domino", domino_type)
        position = Variable("?pos", position_type)
        rotation = Variable("?rot", rotation_type)
        parameters = [robot, domino, position, rotation]
        option_vars = [robot, domino]
        option = Pick
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(DominoAtPos, [domino, position]),
            LiftedAtom(DominoAtRot, [domino, rotation]),
            LiftedAtom(MovableBlock, [domino]),
            LiftedAtom(Upright, [domino]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, domino]),
            LiftedAtom(PosClear, [position]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(DominoAtPos, [domino, position]),
            LiftedAtom(DominoAtRot, [domino, rotation]),
        }
        ignore_effects = {
            Tilting, Upright, DominoAtRot, DominoAtPos, PosClear, Toppled
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(4.0),
                                                   sigma=torch.tensor(0.1))
        pick_domino_process = EndogenousProcess("PickDomino",
                                                parameters, condition_at_start,
                                                set(), set(), add_effects,
                                                delete_effects,
                                                delay_distribution,
                                                torch.tensor(1.0), option,
                                                option_vars, _pick_sampler,
                                                ignore_effects)
        processes.add(pick_domino_process)

        # PlaceDomino: Place domino at specific position and rotation
        # Not in will still be in front to something
        robot = Variable("?robot", robot_type)
        domino1 = Variable("?domino1", domino_type)
        domino2 = Variable("?domino2", domino_type)
        target_pos = Variable("?pos1", position_type)
        rotation = Variable("?rot", rotation_type)
        parameters = [robot, domino1, domino2, target_pos, rotation]
        option_vars = [robot]
        option = Place
        condition_at_start = {
            LiftedAtom(Holding, [robot, domino1]),
            LiftedAtom(PosClear, [target_pos]),
            LiftedAtom(Upright, [domino2]),
            LiftedAtom(AdjacentTo, [target_pos, domino2]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(DominoAtPos, [domino1, target_pos]),
            LiftedAtom(DominoAtRot, [domino1, rotation]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, domino1]),
            LiftedAtom(PosClear, [target_pos]),
        }
        ignore_effects = {
            DominoAtRot, DominoAtPos, PosClear, Tilting, AdjacentTo
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                   sigma=torch.tensor(0.1))
        place_domino_process = EndogenousProcess("PlaceDomino", parameters,
                                                 condition_at_start, set(),
                                                 set(), add_effects,
                                                 delete_effects,
                                                 delay_distribution,
                                                 torch.tensor(1.0), option,
                                                 option_vars, _place_sampler,
                                                 ignore_effects)
        processes.add(place_domino_process)

        # Wait
        robot = Variable("?robot", robot_type)
        parameters = [robot]
        option_vars = [robot]
        option = Wait
        wait_delay_distribution = ConstantDelay(1)
        ignore_effects = {DominoAtRot, DominoAtPos, PosClear, AdjacentTo}
        wait_process = EndogenousProcess("Wait", parameters, set(), set(),
                                         set(), set(), set(),
                                         wait_delay_distribution,
                                         torch.tensor(1.0), option,
                                         option_vars, null_sampler,
                                         ignore_effects)
        processes.add(wait_process)

        # --- Exogenous Processes ---

        # Note: The DominoFall process from the sketch requires a "Falling" predicate
        # which is not currently implemented in the environment.
        # This process would look like:
        domino1 = Variable("?d1", domino_type)
        domino2 = Variable("?d2", domino_type)
        parameters = [domino1, domino2]
        condition_at_start = {
            LiftedAtom(InFront, [domino1, domino2]),
            LiftedAtom(Tilting, [domino2]),
        }
        if CFG.domino_oracle_knows_glued_dominos:
            condition_at_start.update({
                LiftedAtom(DominoNotGlued, [domino1]),
            })
        condition_overall = condition_at_start.copy()
        add_effects = {
            LiftedAtom(Tilting, [domino1]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                                   sigma=torch.tensor(0.1))
        domino_fall_process = ExogenousProcess(
            "DominoFallFromBeingInFrontOfTilting", parameters,
            condition_at_start, condition_overall, set(), add_effects, set(),
            delay_distribution, torch.tensor(1.0))
        processes.add(domino_fall_process)

        # Individual Domino Fall from Tilting to Fall flat
        domino1 = Variable("?d1", domino_type)
        parameters = [domino1]
        condition_at_start = {
            LiftedAtom(Tilting, [domino1]),
        }
        condition_overall = condition_at_start.copy()
        add_effects = {
            LiftedAtom(Toppled, [domino1]),
        }
        delete_effects = {
            LiftedAtom(Tilting, [domino1]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(2.0),
                                                   sigma=torch.tensor(0.1))
        domino_tilting_delete_process = ExogenousProcess(
            "DominoTiltingDelete", parameters, condition_at_start,
            condition_overall, set(), add_effects, delete_effects,
            delay_distribution, torch.tensor(1.0))
        processes.add(domino_tilting_delete_process)

        # --- Wind, when the env has fans -------------------------------
        # A composed env carrying a FanComponent brings switches and
        # fans; a plain domino env does not, and its predicate dict has
        # none of these names.
        if "FanOn" in predicates:
            processes |= cls._get_fan_processes(types, predicates, options)

        return processes

    @classmethod
    def _get_fan_processes(
            cls, types: Dict[str, Type], predicates: Dict[str, Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        """Turning a fan on, and the wind that follows.

        Two processes are enough, and that is the point. The fan env
        needs a grid because a ball's whole trajectory is wind; a domino
        chain's is not. Only the FIRST block is pushed by the wind -
        ``DominoFallFromBeingInFrontOfTilting`` and
        ``DominoTiltingDelete`` above carry the cascade from there. So
        the wind needs exactly one bridging rule into the vocabulary the
        chain already speaks.

        Written over FANS rather than switches because that is the
        vocabulary the env exposes: under
        ``fan_known_controls_relation`` FanComponent hides
        SwitchOn/SwitchOff and publishes FanOn/FanOff, and the switch is
        an implementation detail the option resolves for itself.
        """
        robot_type = types["robot"]
        domino_type = types["domino"]
        fan_type = types["fan"]

        FanOn = predicates["FanOn"]
        FanOff = predicates["FanOff"]
        Upright = predicates["Upright"]
        StartBlock = predicates["InitialBlock"]
        Tilting = predicates["Tilting"]

        processes: Set[CausalProcess] = set()

        # Starting the fan. Endogenous either way: the robot does it.
        # What differs between the two fan envs is only HOW, and so
        # what the process is grounded on.
        robot = Variable("?robot", robot_type)
        fan = Variable("?fan", fan_type)
        if CFG.env == "pybullet_domino_declare":
            # The robot announces it has finished building and the fan
            # starts. The option takes only the robot -- there is
            # nothing to reach for -- so the fan appears in the
            # process's variables and its effects but NOT in the
            # option's arguments. That split is the whole content of
            # this env: an effect with no contact to explain it.
            processes.add(
                EndogenousProcess(
                    "DeclareFinished", [robot, fan],
                    {LiftedAtom(FanOff, [fan])}, set(), set(),
                    {LiftedAtom(FanOn, [fan])}, {LiftedAtom(FanOff, [fan])},
                    DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                          sigma=torch.tensor(0.1)),
                    torch.tensor(1.0), options["DeclareFinished"], [robot],
                    _declare_sampler))
        else:
            processes.add(
                EndogenousProcess(
                    "TurnFanOn", [robot, fan], {LiftedAtom(FanOff, [fan])},
                    set(), set(), {LiftedAtom(FanOn, [fan])},
                    {LiftedAtom(FanOff, [fan])},
                    DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                          sigma=torch.tensor(0.1)),
                    torch.tensor(1.0), options["TurnFanOn"], [robot, fan],
                    _switch_push_sampler))

        # The wind. Exogenous: nobody chooses it, it follows from the
        # fan being on.
        #
        # Deliberately NOT conditioned on which way the fan faces, the
        # way pybullet_fan's MoveToSide is. That needs a direction
        # vocabulary the domino side has no use for, and the tasks are
        # generated with the chain already laid along one fan's axis
        # (domino_fan_aligned_tasks), so a fan press and a topple are
        # one-to-one here. A task set where the planner had to CHOOSE a
        # fan would need it, and this is where it would go.
        #
        # Effects mirror PushStartBlock exactly - Tilting added, Upright
        # deleted - because the two are the same event reached two ways,
        # and a rule that left Upright asserted could fire forever.
        domino = Variable("?domino", domino_type)
        fan2 = Variable("?fan", fan_type)
        conds = {
            LiftedAtom(FanOn, [fan2]),
            LiftedAtom(StartBlock, [domino]),
            LiftedAtom(Upright, [domino]),
        }
        # Delay 0: the block goes the moment the fan comes on. Anything
        # longer opens a window the planner will use - at mu=2 it
        # ordered TurnFanOn FOURTH of six and went on placing dominoes
        # afterwards, believing it could finish the bridge while the
        # start block was mid-topple. It cannot: a Place runs ~19 env
        # steps and the topple is over in a fraction of that. With no
        # window, InFront has to already hold when the fan is switched
        # on, which forces the press to come last - the ordering the
        # task actually has.
        processes.add(
            ExogenousProcess(
                "WindTopplesStartBlock", [domino, fan2], conds, set(), set(),
                {LiftedAtom(Tilting, [domino])},
                {LiftedAtom(Upright, [domino])},
                DiscreteGaussianDelay(mu=torch.tensor(0.0),
                                      sigma=torch.tensor(0.1)),
                torch.tensor(1.0)))
        return processes


# ---------------------------------------------------------------------------
# Grid-free per-skill samplers (NSRTSampler / ParameterizedSampler
# signature) for
# bilevel refinement. The NSRT samplers above read the placement off grid
# ``loc``/``angle`` objects in ``objs``; these instead compute it
# geometrically from the step's ``InFront`` subgoal (passed in the atoms
# slot), so they work in the grid-free agent_bilevel path. Both versions
# coexist intentionally. Refinement clips the returned params to the box.
# ---------------------------------------------------------------------------

_DOMINO_POS_GAP = 0.098  # PyBulletDominoEnv.pos_gap (domino_width * 1.4)
_DOMINO_WIDTH = 0.07  # PyBulletDominoEnv.domino_width
_DOMINO_TARGET_COLOR = (0.85, 0.7, 0.85)
_DOMINO_COLOR_EPS = 1e-3


def _deterministic(sampler: ParameterizedSampler) -> ParameterizedSampler:
    """Flag a sampler as returning constant params (ignores state/rng).

    Backtracking refinement reads this flag to cap such a step's retries at
    1: re-drawing a constant sampler yields the identical option, so spending
    the full per-step budget re-descending through it on every backtrack is
    wasted work (it can never produce a different outcome).
    """
    setattr(sampler, "deterministic", True)
    return sampler


@_deterministic
def _pick_option_sampler(state: State, subgoal_atoms: Set[GroundAtom],
                         rng: np.random.Generator,
                         objects: Sequence[Object]) -> Array:
    """Grid-free Pick sampler: fixed grasp height above the domino origin."""
    del state, subgoal_atoms, rng, objects
    return np.array([_grasp_z_offset()], dtype=np.float32)


@_deterministic
def _push_option_sampler(state: State, subgoal_atoms: Set[GroundAtom],
                         rng: np.random.Generator,
                         objects: Sequence[Object]) -> Array:
    """Grid-free Push sampler: approach distance / contact height."""
    del state, subgoal_atoms, rng, objects
    return np.array([_push_approach_distance(),
                     _push_contact_z_offset()],
                    dtype=np.float32)


def _score_placement(state: State, subgoal_atoms: Set[GroundAtom],
                     held: Object, hx: float, hy: float, hyaw: float) -> int:
    """Count subgoal atoms that hold if ``held`` is placed at (hx, hy,
    hyaw)."""
    s2 = state.copy()
    s2.set(held, "x", hx)
    s2.set(held, "y", hy)
    s2.set(held, "yaw", hyaw)
    s2.set(held, "roll", 0.0)
    s2.set(held, "is_held", 0.0)
    return sum(1 for atom in subgoal_atoms if atom.holds(s2))


def _is_cardinal(angle: float) -> bool:
    """True when ``angle`` is within ~10 deg of a cardinal (axis-aligned) yaw.

    Mirrors the cardinal-facing gate in
    ``DominoComponent._InFront_holds``: a settled reference domino sits
    a degree or two off cardinal, so a hard equality would make chained
    placements onto it unsatisfiable.
    """
    card_thresh = float(np.sin(np.radians(10)))
    return bool(
        abs(np.sin(angle)) < card_thresh or abs(np.cos(angle)) < card_thresh)


def _generator_placements(xr: float, yr: float,
                          ryaw: float) -> List[Tuple[float, float, float]]:
    """Every placement the task generator would lay next to a reference.

    Reproduces ``DominoTaskGenerator._place_straight_domino`` /
    ``_place_turn90_domino`` exactly -- one ``pos_gap`` along a cardinal
    travel direction, with 45-deg turn blocks carrying the generator's
    half-width inward side offset -- expressed relative to a reference domino
    at ``(xr, yr, ryaw)``. Each returned ``(cx, cy, cyaw)`` is a valid
    ``InFront`` placement off the reference.

    A cardinal reference yields, for each of the two chain (forward / backward)
    directions, the straight successor and the two turn-start (``d1``) blocks
    (left / right). A non-cardinal reference -- an already-placed 45-deg
    turn-start block -- yields the turn-completing (``d2``) block that bends
    the chain the rest of the way through the corner.
    """
    gap = _DOMINO_POS_GAP
    s_off = -_DOMINO_WIDTH / 2  # generator's d1_side_offset / side_offset
    out: List[Tuple[float, float, float]] = []
    if _is_cardinal(ryaw):
        for rotation in (ryaw, wrap_angle(ryaw + np.pi)):
            # Straight successor: one gap along travel, same (box) yaw.
            out.append(
                (xr + gap * np.sin(rotation), yr + gap * np.cos(rotation),
                 wrap_angle(ryaw)))
            # Turn-start (d1): one gap ahead, nudged a half width orthogonal
            # to the post-turn travel direction, yaw stepped +-45.
            for turn in (1.0, -1.0):
                d1_dir = wrap_angle(rotation - turn * np.pi / 4)
                cx = xr + gap * np.sin(rotation) + turn * s_off * np.cos(
                    d1_dir)
                cy = yr + gap * np.cos(rotation) - turn * s_off * np.sin(
                    d1_dir)
                out.append((cx, cy, wrap_angle(ryaw + turn * np.pi / 4)))
    else:
        # Turn-completing block (d2) off an already-placed turn-start block.
        # Take whichever turn sign(s) leave the pre-turn travel cardinal.
        for turn in (1.0, -1.0):
            base = wrap_angle(ryaw - turn * np.pi / 4)
            if not _is_cardinal(base):
                continue
            d1_dir = wrap_angle(base - turn * np.pi / 4)
            d2_rot = wrap_angle(base - turn * np.pi / 2)
            cx = xr + gap * np.sin(d1_dir) + turn * s_off * np.cos(d2_rot)
            cy = yr + gap * np.cos(d1_dir) - turn * s_off * np.sin(d2_rot)
            out.append((cx, cy, wrap_angle(base + turn * np.pi / 2)))
    return out


def _is_target_domino(state: State, domino: Object) -> bool:
    """Check whether ``domino`` has the target-block color."""
    return all(
        abs(state.get(domino, feat) - val) < _DOMINO_COLOR_EPS
        for feat, val in zip(("r", "g", "b"), _DOMINO_TARGET_COLOR))


def _future_target_bridge_score(state: State, held: Object, hx: float,
                                hy: float, hyaw: float) -> float:
    """Tie-break score for placements that can be completed to a target.

    The immediate ``InFront(held, ref)`` subgoal underdetermines which
    side of the start domino to place the bridge on. Prefer placements
    for which one additional domino can be placed at the intersection of
    generator-faithful successors from the held domino and from a purple
    target domino. This keeps the sampler from spending most refinement
    attempts on locally valid but globally dead first placements.
    """
    dominoes = [o for o in state if o.type.name == "domino" and o is not held]
    targets = [d for d in dominoes if _is_target_domino(state, d)]
    if not targets:
        return 0.0
    held_next = _generator_placements(hx, hy, hyaw)
    if not held_next:
        return 0.0
    best_resid = float("inf")
    yaw_scale = _DOMINO_POS_GAP / np.pi
    for target in targets:
        tx = state.get(target, "x")
        ty = state.get(target, "y")
        tyaw = state.get(target, "yaw")
        for hx2, hy2, hyaw2 in held_next:
            for tx2, ty2, tyaw2 in _generator_placements(tx, ty, tyaw):
                yaw_resid = abs(wrap_angle(hyaw2 - tyaw2)) * yaw_scale
                resid = float(np.hypot(hx2 - tx2, hy2 - ty2) + yaw_resid)
                best_resid = min(best_resid, resid)
    if best_resid == float("inf"):
        return 0.0
    return -best_resid


def _place_option_sampler(state: State, subgoal_atoms: Set[GroundAtom],
                          rng: np.random.Generator,
                          objects: Sequence[Object]) -> Array:
    """Grid-free Place sampler that draws a generator-faithful placement.

    Builds the discrete set of placements the task generator could lay next
    to each reference domino named in an ``InFront`` subgoal -- straight, or a
    45-deg left / right turn block, in either chain direction (see
    ``_generator_placements``) -- scores each by how many of the step's
    subgoal atoms it satisfies, and draws one uniformly at random from those
    tied for the best score. Randomizing (rather than always returning the
    first / straight placement) is what lets backtracking that re-draws this
    step reach a turn when the lone subgoal (e.g. ``InFront(d1, d0)``) is
    satisfied equally by straight and by a turn and a later step needs the
    bend. No jitter is added -- the generator placements are already the
    exact, cascade-tuned poses. Raises (so refinement falls back to uniform)
    when the held domino or a usable reference can't be found.
    """
    del objects
    dominoes = [o for o in state if o.type.name == "domino"]
    held = [d for d in dominoes if state.get(d, "is_held") > 0.5]
    if len(held) != 1:
        raise ValueError(f"expected one held domino, found {len(held)}")
    held_d = held[0]

    refs = []
    for atom in subgoal_atoms:
        if atom.predicate.name != "InFront":
            continue
        d1, d2 = atom.objects
        if held_d is d1 and held_d is not d2:
            refs.append(d2)
        elif held_d is d2 and held_d is not d1:
            refs.append(d1)
    if not refs:
        raise ValueError("no InFront subgoal references the held domino")

    # Collect every generator-faithful candidate, scored by how many of the
    # step's subgoal atoms it satisfies. The candidates come straight from the
    # task generator's geometry, so each is a valid InFront placement off its
    # reference and the set is exactly what the generator could have laid.
    candidates: List[Tuple[int, float, float, float, float]] = []
    for ref in refs:
        xr = state.get(ref, "x")
        yr = state.get(ref, "y")
        rot = state.get(ref, "yaw")
        for cx, cy, cyaw in _generator_placements(xr, yr, rot):
            score = _score_placement(state, subgoal_atoms, held_d, cx, cy,
                                     cyaw)
            future_score = _future_target_bridge_score(state, held_d, cx, cy,
                                                       cyaw)
            candidates.append((score, future_score, cx, cy, cyaw))
    if not candidates:
        raise ValueError("no usable reference domino for placement")

    # Randomize among the placements tied for the best score, so backtracking
    # that re-draws this step explores a turn instead of always returning the
    # straight pose. Score alone disambiguates: a multi-edge step (a second
    # InFront naming the next block) is satisfied only by the turn block that
    # bends toward it, which no straight placement matches.
    best_score = max(c[0] for c in candidates)
    best_future_score = max(c[1] for c in candidates if c[0] == best_score)
    tied = [
        c for c in candidates
        if c[0] == best_score and abs(c[1] - best_future_score) < 1e-9
    ]
    _, _, cx, cy, cyaw = tied[int(rng.integers(len(tied)))]
    return np.array([cx, cy, _DOMINO_DROP_Z, cyaw], dtype=np.float32)


class PyBulletDominoGroundTruthSamplerFactory(GroundTruthSamplerFactory):
    """Ground-truth grid-free per-skill samplers for the domino env."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "pybullet_domino_grid", "pybullet_domino", "pybullet_domino_real",
            "pybullet_domino_real_geometry", "pybullet_domino_fan",
            "pybullet_domino_declare",
            "pybullet_domino_blow"
        }

    @classmethod
    def get_samplers(cls, env_name: str) -> Dict[str, ParameterizedSampler]:
        del env_name
        return {
            "Pick": _pick_option_sampler,
            "Push": _push_option_sampler,
            "Place": _place_option_sampler,
        }


# ── Blow task: pick, place upwind, declare, and let the wind deliver ──


def _blow_place_sampler(state: State, subgoal_atoms: Set[GroundAtom],
                        rng: np.random.Generator,
                        objects: Sequence[Object]) -> Array:
    """Put the block one slide-length upwind of the goal patch.

    The oracle's whole advantage in this env is this number. A learner
    has to recover it from watching blocks slide; here it is read
    straight off the ground-truth curve.
    """
    del subgoal_atoms, objects
    # pylint: disable-next=import-outside-toplevel
    from predicators.ground_truth_models.domino.predicates import \
        _blow_slide_distance
    regions = [o for o in state if o.type.name == "region"]
    held = [
        o for o in state
        if o.type.name == "domino" and state.get(o, "is_held") > 0.5
    ]
    if not regions or len(held) != 1:
        raise ValueError("blow place sampler: need a region and a held block")
    region = regions[0]
    x = float(state.get(region, "x")) - _blow_slide_distance()
    y = float(state.get(region, "y"))
    # A hair of jitter so backtracking can re-draw rather than retrying
    # an identical pose, kept well inside the patch's own tolerance.
    x += float(rng.uniform(-0.005, 0.005))
    # The canonical release height, NOT the held block's current z: the
    # Place option's release_z is where the GRIPPER opens (its declared
    # range is 0.5-0.6), and the block's carried z is neither that nor
    # inside it, so every refinement was asking for a drop the skill
    # could not make.
    # yaw pi/2 puts the block's WIDE face into the wind. Dropping it at
    # yaw 0 leaves the narrow edge facing the gust, which the wind
    # creeps along without ever tipping: measured 5.0 cm and roll 0.000
    # where the same force on a turned block gives 14.4 cm and flat.
    # The generator stages the block turned; the placement has to keep
    # it that way.
    return np.array([x, y, _DOMINO_DROP_Z, np.pi / 2], dtype=np.float32)


def _get_blow_processes(
        types: Dict[str, Type], predicates: Dict[str, Predicate],
        options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
    """Pick, place upwind, declare, and let the wind carry the block.

    Four processes and no grid. The one that matters is the last: the
    wind is EXOGENOUS - the robot does not carry the block into the
    goal, it arranges the world so that the wind will, and then says it
    is done. That is the shape of the whole task, and it is why the
    placement has to be right rather than merely somewhere.
    """
    robot_type = types["robot"]
    domino_type = types["domino"]
    fan_type = types["fan"]
    region_type = types["region"]

    HandEmpty = predicates["HandEmpty"]
    Holding = predicates["Holding"]
    FanOn = predicates["FanOn"]
    FanOff = predicates["FanOff"]
    ReadyToBlow = predicates["ReadyToBlow"]
    InGoal = predicates["InGoal"]

    robot = Variable("?robot", robot_type)
    block = Variable("?block", domino_type)
    fan = Variable("?fan", fan_type)
    region = Variable("?region", region_type)

    processes: Set[CausalProcess] = set()

    # Predicates a pick or a place disturbs incidentally. Lifting a
    # block off the table changes whether it is Upright and whether it
    # is where the wind would take it; a process that does not declare
    # those as ignorable is rejected in refinement for effects it never
    # claimed, which is what stalled every skeleton at step 0.
    Upright = predicates["Upright"]
    incidental = {Upright, ReadyToBlow, InGoal}

    # Pick the block up.
    processes.add(
        EndogenousProcess(
            "PickBlock", [robot, block], {LiftedAtom(HandEmpty, [robot])},
            set(), set(), {LiftedAtom(Holding, [robot, block])},
            {LiftedAtom(HandEmpty, [robot])},
            DiscreteGaussianDelay(mu=torch.tensor(4.0),
                                  sigma=torch.tensor(0.1)),
            torch.tensor(1.0), options["Pick"], [robot, block],
            _pick_sampler, incidental))

    # Put it down one slide-length upwind of the patch.
    processes.add(
        EndogenousProcess(
            "PlaceUpwind", [robot, block, region],
            {LiftedAtom(Holding, [robot, block])}, set(), set(), {
                LiftedAtom(HandEmpty, [robot]),
                LiftedAtom(ReadyToBlow, [block, region])
            }, {LiftedAtom(Holding, [robot, block])},
            DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                  sigma=torch.tensor(0.1)),
            torch.tensor(1.0), options["Place"], [robot],
            _blow_place_sampler, incidental))

    # Press the switch, and the fan starts. HandEmpty is a precondition
    # and not decoration: without it the planner is free to press while
    # still holding the block, and its first skeleton did exactly that -
    # PickBlock, <trigger>, PlaceUpwind - which blows the gust across an
    # empty table while the arm is still carrying the thing it was
    # supposed to move.
    processes.add(
        EndogenousProcess(
            "TurnFanOn", [robot, fan, block, region], {
                LiftedAtom(FanOff, [fan]),
                LiftedAtom(HandEmpty, [robot]),
                # The switch is only worth pressing once the block is
                # where the gust can deliver it. HandEmpty alone is true
                # at t=0, so without this the planner's first skeleton
                # pressed the switch before it had even picked the block
                # up and blew the gust across an empty table.
                LiftedAtom(ReadyToBlow, [block, region])
            }, set(), set(), {LiftedAtom(FanOn, [fan])},
            {LiftedAtom(FanOff, [fan])},
            DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                  sigma=torch.tensor(0.1)),
            torch.tensor(1.0), options["TurnFanOn"], [robot, fan],
            _switch_push_sampler))

    # The gust. Exogenous: the robot never carries the block in.
    # condition_overall as well as condition_at_start: the gust only
    # delivers the block if the fan STAYS on and the block STAYS where
    # it was put for the whole flight, which is what the cascade's own
    # exogenous processes assert too.
    wind_conditions = {
        LiftedAtom(FanOn, [fan]),
        LiftedAtom(ReadyToBlow, [block, region])
    }
    processes.add(
        ExogenousProcess(
            "WindCarriesToGoal", [fan, block, region], wind_conditions,
            wind_conditions.copy(), set(),
            {LiftedAtom(InGoal, [block, region])}, set(),
            # Delay is in PROCESS steps, not simulator steps. Handing it
            # the gust's 60 simulator steps put the effect beyond the
            # planner's lookahead and every skeleton was exhausted
            # without the goal ever becoming true. The cascade's own
            # exogenous processes use 1-4 for the same reason.
            DiscreteGaussianDelay(mu=torch.tensor(12.0),
                                  sigma=torch.tensor(0.5)),
            torch.tensor(1.0)))

    # Wait. No preconditions, no effects: it exists so the planner can
    # let TIME pass. The gust needs about sixty simulator steps to carry
    # the block, and without a Wait in the skeleton the episode ends the
    # instant the robot finishes speaking.
    processes.add(
        EndogenousProcess("Wait", [robot], set(), set(), set(), set(), set(),
                          ConstantDelay(1), torch.tensor(1.0),
                          options["Wait"], [robot], null_sampler))

    return processes
