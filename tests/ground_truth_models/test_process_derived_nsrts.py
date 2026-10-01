"""Ground-truth NSRTs derived from the oracle endogenous processes, for the
process-planning envs whose NSRT factories are missing or stale."""
from typing import Sequence, Set

import numpy as np
import pytest
import torch
from gym.spaces import Box

from predicators import utils
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options, \
    get_gt_processes, nsrts_from_processes
from predicators.structs import Action, Array, CausalProcess, \
    EndogenousProcess, ExogenousProcess, GroundAtom, LiftedAtom, Object, \
    Predicate, State, Type, Variable
from predicators.utils import ConstantDelay


def _sampler(state: State, goal: Set[GroundAtom], rng: np.random.Generator,
             objs: Sequence[Object]) -> Array:
    del state, goal, objs
    return rng.uniform(size=(1, )).astype(np.float32)


def test_nsrts_from_processes_keeps_the_endogenous_view() -> None:
    """An endogenous process becomes an NSRT with its start condition as
    preconditions and the same effects, option and sampler; exogenous processes
    are dropped."""
    cup_type = Type("cup", ["x"])
    Full = Predicate("Full", [cup_type], lambda s, o: True)
    Empty = Predicate("Empty", [cup_type], lambda s, o: False)
    option = utils.SingletonParameterizedOption("Fill",
                                                lambda s, m, o, p: Action(p),
                                                types=[cup_type],
                                                params_space=Box(0, 1, (1, )))
    cup = Variable("?cup", cup_type)
    pre = {LiftedAtom(Empty, [cup])}
    add = {LiftedAtom(Full, [cup])}
    endo = EndogenousProcess("FillCup", [cup], set(pre), set(), set(),
                             set(add), set(pre), ConstantDelay(1),
                             torch.tensor(1.0), option, [cup], _sampler)
    exo = ExogenousProcess("Evaporate", [cup], set(pre), set(), set(),
                           set(add), set(pre), ConstantDelay(1),
                           torch.tensor(1.0))
    processes: Set[CausalProcess] = {endo, exo}
    nsrts = nsrts_from_processes(processes)
    assert len(nsrts) == 1
    nsrt = next(iter(nsrts))
    assert nsrt.name == "FillCup"
    assert nsrt.preconditions == pre
    assert nsrt.add_effects == add
    assert nsrt.option is option and list(nsrt.option_vars) == [cup]
    params = nsrt.sampler(State({}), set(), np.random.default_rng(0), [])
    assert params.shape == (1, )


def test_fan_gt_nsrts_come_from_the_processes() -> None:
    """pybullet_fan's NSRT factory predates its current types; get_gt_nsrts
    falls back to the process-derived NSRTs instead of raising."""
    utils.reset_config({
        "env": "pybullet_fan",
        "num_train_tasks": 1,
        "num_test_tasks": 1
    })
    env = create_new_env("pybullet_fan", do_cache=False, use_gui=False)
    options = get_gt_options("pybullet_fan")
    nsrts = get_gt_nsrts("pybullet_fan", env.predicates, options)
    names = {n.name for n in nsrts}
    assert {"TurnFanOn", "TurnFanOff"}.issubset(names)
    for nsrt in nsrts:
        assert nsrt.option in options
        for atom in nsrt.preconditions | nsrt.add_effects:
            assert atom.predicate in env.predicates


@pytest.mark.parametrize("domino_targets", [False, True])
def test_domino_gt_processes_build_in_both_target_modes(
        domino_targets: bool) -> None:
    """The domino processes are well typed with the default hinged targets as
    well as with domino targets, and get_gt_nsrts derives the NSRTs from them.

    Toppled ranges over the targets, so a domino that falls flat adds it
    only when dominoes are the targets.
    """
    utils.reset_config({
        "env": "pybullet_domino",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "domino_use_domino_blocks_as_target": domino_targets,
    })
    env = create_new_env("pybullet_domino", do_cache=False, use_gui=False)
    options = get_gt_options("pybullet_domino")
    processes = get_gt_processes("pybullet_domino", env.predicates, options)
    fall_flat = next(p for p in processes if p.name == "DominoTiltingDelete")
    expected = {"Toppled"} if domino_targets else set()
    assert {a.predicate.name for a in fall_flat.add_effects} == expected
    nsrts = get_gt_nsrts("pybullet_domino", env.predicates, options)
    nsrt_names = {n.name for n in nsrts}
    assert {"PushStartBlock", "PickDomino", "PlaceDomino"} <= nsrt_names
