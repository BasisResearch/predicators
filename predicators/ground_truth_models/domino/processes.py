"""Ground-truth processes for the domino environment."""

from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from predicators.ground_truth_models import GroundTruthProcessFactory, \
    GroundTruthSamplerFactory
from predicators.settings import CFG
from predicators.structs import Array, CausalProcess, EndogenousProcess, \
    ExogenousProcess, GroundAtom, LiftedAtom, Object, OptionSampler, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import ConstantDelay, DiscreteGaussianDelay, \
    null_sampler, wrap_angle

# Fixed parameter values for domino environment.
_DOMINO_GRASP_Z_OFFSET = 0.0825  # domino_height * 0.55
_DOMINO_DROP_Z = 0.5695  # table_height + domino_height * 1.13
_DOMINO_OFFSET_X = 0.045  # domino_depth * 3
_DOMINO_OFFSET_Z = 0.0825  # domino_height * 0.55


def _pick_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    """Return fixed grasp_z_offset for domino pick."""
    del state, goal, rng, objs
    return np.array([_DOMINO_GRASP_Z_OFFSET], dtype=np.float32)


def _push_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    """Return fixed push params for domino push."""
    if not CFG.domino_use_skill_factories:
        return np.array([], dtype=np.float32)
    del state, goal, rng, objs
    return np.array([_DOMINO_OFFSET_X, _DOMINO_OFFSET_Z], dtype=np.float32)


def _place_sampler(state: State, goal: Set[GroundAtom],
                   rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    """Return placement params from process objects."""
    if not CFG.domino_use_skill_factories:
        return np.array([], dtype=np.float32)
    del state, goal, rng
    # objs = [robot, domino1, domino2, target_pos, rotation]
    target_pos = objs[3]
    rotation = objs[4]
    x = float(target_pos.name.split("_")[1])
    y = float(target_pos.name.split("_")[2])
    angle_deg = float(rotation.name.split("_")[-1])
    yaw = np.radians(angle_deg)
    return np.array([x, y, _DOMINO_DROP_Z, yaw], dtype=np.float32)


class PyBulletDominoGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the domino grid environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_domino_grid", "pybullet_domino"}

    @classmethod
    def get_processes(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        del env_name  # unused

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

        # Options
        Push = options["Push"]
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
        option = Push
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
        push_start_block_process = EndogenousProcess(
            "PushStartBlock", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _push_sampler,
            ignore_effects)
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

        return processes


# ---------------------------------------------------------------------------
# Grid-free per-skill samplers (NSRTSampler / OptionSampler signature) for
# bilevel refinement. The NSRT samplers above read the placement off grid
# ``loc``/``angle`` objects in ``objs``; these instead compute it
# geometrically from the step's ``InFront`` subgoal (passed in the atoms
# slot), so they work in the grid-free agent_bilevel path. Both versions
# coexist intentionally. Refinement clips the returned params to the box.
# ---------------------------------------------------------------------------

_DOMINO_POS_GAP = 0.098  # PyBulletDominoEnv.pos_gap (domino_width * 1.4)


def _pick_option_sampler(state: State, subgoal_atoms: Set[GroundAtom],
                         rng: np.random.Generator,
                         objects: Sequence[Object]) -> Array:
    """Grid-free Pick sampler: fixed grasp height above the domino origin."""
    del state, subgoal_atoms, rng, objects
    return np.array([_DOMINO_GRASP_Z_OFFSET], dtype=np.float32)


def _push_option_sampler(state: State, subgoal_atoms: Set[GroundAtom],
                         rng: np.random.Generator,
                         objects: Sequence[Object]) -> Array:
    """Grid-free Push sampler: fixed approach distance / contact height."""
    del state, subgoal_atoms, rng, objects
    return np.array([_DOMINO_OFFSET_X, _DOMINO_OFFSET_Z], dtype=np.float32)


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


def _place_option_sampler(state: State, subgoal_atoms: Set[GroundAtom],
                          rng: np.random.Generator,
                          objects: Sequence[Object]) -> Array:
    """Grid-free Place sampler aimed at the step's ``InFront`` subgoal.

    Places the held domino one ``pos_gap`` from a reference domino named
    in an ``InFront`` subgoal atom, along the reference's cardinal
    facing, at the turn offset (straight / +-45 deg) and direction that
    satisfy the most subgoal atoms. Raises (so refinement falls back to
    uniform) when the held domino or a usable reference can't be found.
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

    turn_offsets = (0.0, np.pi / 4, -np.pi / 4)
    best: Optional[Tuple[float, float, float]] = None
    best_score = -1
    for ref in refs:
        xr = state.get(ref, "x")
        yr = state.get(ref, "y")
        rot = state.get(ref, "yaw")
        # _InFront's "ahead" relation only holds for cardinal back-facings.
        if not (abs(np.sin(rot)) < 1e-3 or abs(np.cos(rot)) < 1e-3):
            continue
        for direction in (1.0, -1.0):
            cx = xr + direction * _DOMINO_POS_GAP * np.sin(rot)
            cy = yr + direction * _DOMINO_POS_GAP * np.cos(rot)
            for off in turn_offsets:
                cyaw = wrap_angle(rot + off)
                score = _score_placement(state, subgoal_atoms, held_d, cx, cy,
                                         cyaw)
                if score > best_score:
                    best_score = score
                    best = (cx, cy, cyaw)
    if best is None:
        raise ValueError("no cardinal-facing reference domino for placement")

    cx, cy, cyaw = best
    # Small jitter (well within InFront's position tolerance) so backtracking
    # retries explore slightly different placements. Refinement clips the
    # result to the option's params box.
    jitter = _DOMINO_POS_GAP * 0.05
    cx += float(rng.uniform(-jitter, jitter))
    cy += float(rng.uniform(-jitter, jitter))
    return np.array([cx, cy, _DOMINO_DROP_Z, cyaw], dtype=np.float32)


class PyBulletDominoGroundTruthSamplerFactory(GroundTruthSamplerFactory):
    """Ground-truth grid-free per-skill samplers for the domino env."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_domino_grid", "pybullet_domino"}

    @classmethod
    def get_samplers(cls, env_name: str) -> Dict[str, OptionSampler]:
        del env_name
        return {
            "Pick": _pick_option_sampler,
            "Push": _push_option_sampler,
            "Place": _place_option_sampler,
        }
