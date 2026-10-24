"""EMPIRIC from assets with the agent's own parameter belief.

The agent builds its scene as in EMPIRIC from assets, the harness fits
nothing, and the agent supplies its parameter uncertainty as
``PARAM_DRAWS`` in ``simulator.py``; the joint draws run on them.
"""
# pylint: disable=protected-access
import hashlib
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from predicators.agent_sdk.belief_probe import BeliefProbe
from predicators.agent_sdk.tools import ToolContext
from predicators.approaches import create_approach
from predicators.approaches.agent_continual_from_assets_approach import \
    AgentContinualFromAssetsApproach
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.parameter_belief import MAX_AGENT_DRAWS, \
    agent_draws_belief, agent_param_draws
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.run.level_players import create_level_player
from predicators.structs import Dataset
from tests.approaches.test_agent_continual_approach import _call, _config, \
    _result
from tests.approaches.test_agent_continual_real_to_sim_approach import \
    SCENE_SOURCE, _benchmark_flags

_NAME = "agent_continual_from_assets_agent_belief"
_SPECS = [
    ParamSpec("mass", 0.25, lo=0.1, hi=1.0, scale="log"),
    ParamSpec("lift", 1.4, lo=0.5, hi=3.0),
]


def _arm_flags() -> Dict[str, Any]:
    flags = _benchmark_flags("pybullet_boil")
    flags.update(agent_sim_learn_declared_params_only=True,
                 continual_rehearsal_gate=True,
                 continual_predictive_check=True,
                 approach=_NAME,
                 env="pybullet_boil",
                 continual_render=False,
                 continual_make_video=False)
    return flags


def _make(tmp_path: Any, **overrides: Any) -> Any:
    flags = _arm_flags()
    flags.update(overrides)
    _config(tmp_path, **flags)
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    return create_approach(_NAME, env.predicates,
                           get_gt_options(env.get_name()), env.types,
                           env.action_space,
                           [t.task for t in env.get_train_tasks()])


def test_param_draws_are_checked_against_the_declarations():
    """Draws name declared parameters inside their ranges; one that leaves a
    parameter out takes its init_value; anything else is refused with the
    reason."""
    rows, why = agent_param_draws([{"mass": 0.3}, {"lift": 2.0}], _SPECS)
    assert why == ""
    assert rows == [{"mass": 0.3, "lift": 1.4}, {"mass": 0.25, "lift": 2.0}]
    rows, why = agent_param_draws(lambda: [{"mass": 0.5}], _SPECS)
    assert rows == [{"mass": 0.5, "lift": 1.4}] and why == ""
    for value, reason in (
        ([], "non-empty list"),
        ({
            "mass": 0.3
        }, "non-empty list"),
        ([{
            "drag": 0.1
        }], "undeclared parameters ['drag']"),
        ([{
            "mass": 2.0
        }], "outside its declared range"),
        ([{
            "mass": float("nan")
        }], "outside its declared range"),
        ([{
            "mass": "heavy"
        }], "is not a number"),
        ([0.3], "is not a dict"),
        ([{}] * (MAX_AGENT_DRAWS + 1), "the most it may hold"),
    ):
        rows, why = agent_param_draws(value, _SPECS)
        assert rows == [] and reason in why, (value, why)

    def broken() -> Any:
        raise RuntimeError("no data")

    assert agent_param_draws(
        broken, _SPECS) == ([], "PARAM_DRAWS() raised RuntimeError: no data")


def test_the_belief_is_the_draws_and_nothing_tempers_it():
    """Few draws are all stored; many give the rehearsal a random subset of
    them; fresh draws resample the whole list; tempering moves none."""
    rows, _ = agent_param_draws([{
        "mass": 0.2 + 0.01 * i,
        "lift": 1.0 + 0.1 * i
    } for i in range(5)], _SPECS)
    belief = agent_draws_belief(_SPECS, rows, 16, seed=0)
    assert belief.map_estimate == {"mass": 0.25, "lift": 1.4}
    assert belief.draw_dicts() == rows
    assert np.array_equal(belief.inflated(9.0).draws, belief.draws)
    fresh = belief.sample(np.random.default_rng(1), 50)
    assert {tuple(sorted(r.items())) for r in fresh} <= \
        {tuple(sorted(r.items())) for r in rows}
    assert belief.describe()
    many, _ = agent_param_draws([{
        "lift": 0.5 + 0.01 * i
    } for i in range(40)], _SPECS)
    subset = agent_draws_belief(_SPECS, many, 16, seed=3).draw_dicts()
    assert len(subset) == 16
    assert len({r["lift"] for r in subset}) == 16
    assert all(r in many for r in subset)


def test_a_method_the_arm_leaves_out_is_absent():
    """A probe method listed in probe_absent does not exist on ``sim``; the
    others stay."""
    sim = BeliefProbe(ToolContext(probe_absent=frozenset({"fit"})))
    assert not hasattr(sim, "fit") and "fit" not in dir(sim)
    assert hasattr(sim, "run") and "run" in dir(sim)
    with pytest.raises(AttributeError, match="no attribute 'fit'"):
        sim.fit()  # pylint: disable=no-member
    assert hasattr(BeliefProbe(ToolContext()), "fit")


def test_arm_needs_the_joint_belief(tmp_path: Any) -> None:
    """Without uncertainty decisions or joint draws there is nothing to run the
    agent's draws on."""
    with pytest.raises(ValueError, match="joint_draws"):
        _make(tmp_path, belief_joint_draws=0)


def test_arm_serves_the_agents_draws(tmp_path: Any, monkeypatch: Any) -> None:
    """The draws the current simulator.py exported are the parameter belief,
    and the model status says what the rehearsals run; a refused or missing
    list leaves the declared values, and the status says why."""
    approach = _make(tmp_path)
    assert approach._probe_surface().fit is False
    assert approach._probe_surface().uncertainty is True
    assert approach._tool_context.probe_absent == frozenset({"fit"})
    approach._physical_param_specs = list(_SPECS)
    digest = {"now": "a"}
    monkeypatch.setattr(approach, "_current_simulator_digest",
                        lambda: digest["now"])
    approach._exported_draws.update({
        "a": [{
            "mass": 0.2,
            "lift": 1.2
        }, {
            "mass": 0.4
        }],
        "b": [{
            "mass": 5.0
        }],
        "c":
        None,
    })
    assert approach.parameter_belief().draw_dicts() == [{
        "mass": 0.2,
        "lift": 1.2
    }, {
        "mass": 0.4,
        "lift": 1.4
    }]
    status = approach._fit_status_text()
    assert "2 PARAM_DRAWS" in status and "varying lift, mass" in status
    declared = [{"mass": 0.25, "lift": 1.4}]
    for now, reason in (("b", "outside its declared range"),
                        ("c", "exports no PARAM_DRAWS"), ("d",
                                                          "has not loaded")):
        digest["now"] = now
        assert approach.parameter_belief().draw_dicts() == declared
        assert reason in approach._fit_status_text()


def test_loading_keeps_the_exported_draws(tmp_path: Any,
                                          monkeypatch: Any) -> None:
    """A load keeps PARAM_DRAWS by the file's content, calling a function then;
    one that raises is reported."""
    approach = _make(tmp_path)
    path = tmp_path / "simulator.py"
    path.write_text("# a model\n", encoding="utf-8")
    namespace: Dict[str, Any] = {}

    def parent_load(self: Any, *_args: Any, **_kwargs: Any) -> Any:
        del self
        return None, None, None, namespace

    monkeypatch.setattr(AgentContinualFromAssetsApproach,
                        "_load_simulator_from_module_file", parent_load)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    namespace["PARAM_DRAWS"] = lambda: [{"mass": 0.3}]
    approach._load_simulator_from_module_file(str(path))
    assert approach._exported_draws[digest] == [{"mass": 0.3}]

    def broken() -> Any:
        raise RuntimeError("no data")

    namespace["PARAM_DRAWS"] = broken
    approach._load_simulator_from_module_file(str(path))
    approach._physical_param_specs = list(_SPECS)
    monkeypatch.setattr(approach, "_current_simulator_digest", lambda: digest)
    assert "PARAM_DRAWS() raised RuntimeError: no data" in \
        approach._fit_status_text()


def test_arm_prompt_describes_the_agents_draws(tmp_path: Any) -> None:
    """The prompt keeps EMPIRIC from assets' modelling rules, says the harness
    fits nothing, gives the PARAM_DRAWS contract, and says the predictive check
    leaves the draws as written."""
    approach = _make(tmp_path)
    prompt = approach._get_agent_system_prompt()
    assert "EMPIRIC from assets, with your own parameter belief" in prompt
    assert "PARAM_DRAWS = [" in prompt
    assert "Each rehearsal runs 16 joint draws" in prompt
    assert "The harness leaves your draws as you wrote them" in prompt
    assert "sim.fit()" not in prompt
    assert "the fit weighs them" not in prompt
    assert "whose state a force depends on" in prompt
    assert "make your first step on a new level" in prompt
    assert "Test levels require a loaded model" in prompt


@pytest.mark.slow
def test_agent_rehearses_on_its_own_draws(tmp_path: Any,
                                          monkeypatch: Any) -> None:
    """End to end on Boil: the agent's scene loads with its PARAM_DRAWS, a
    rehearsal's draws are the agent's (an undeclared material stays at the
    scene's value), sim.fit refuses, and a loadable model opens the test- level
    gate."""
    approach = _make(tmp_path)
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    draws = [{
        "jug_spinning_friction": 0.001
    }, {
        "jug_spinning_friction": 0.002
    }, {
        "jug_spinning_friction": 0.003
    }]

    def query(message: str, *_args: Any, **_kwargs: Any) -> Any:
        assert "PARAM_DRAWS" in message
        sandbox = Path(approach._tool_context.sandbox_dir)
        (sandbox / "simulator.py").write_text(SCENE_SOURCE +
                                              f"\nPARAM_DRAWS = {draws!r}\n",
                                              encoding="utf-8")
        loaded = _call(approach,
                       "run_python",
                       code="sim.reset(current=True)\nprint('loaded')")
        assert "loaded" in loaded and not loaded.startswith("ERROR"), loaded
        assert "3 PARAM_DRAWS" in approach._fit_status_text()
        code = (
            "r = sim.reset(current=True).run('Wait(robot:robot)[2]')\n"
            "spin = sorted({d['params']['jug_spinning_friction'] "
            "for d in r.draws})\n"
            "roll = {d['params'].get('jug_rolling_friction') for d in r.draws}\n"
            "print('SPIN', spin, 'ROLL', roll)")
        rehearsed = _call(approach, "run_python", code=code)
        assert not rehearsed.startswith("ERROR"), rehearsed
        assert "SPIN [0.001, 0.002, 0.003]" in rehearsed, rehearsed
        assert "ROLL {None}" in rehearsed, rehearsed
        absent = _call(approach,
                       "run_python",
                       code="print(hasattr(sim, 'fit'), 'fit' in dir(sim))")
        assert "False False" in absent, absent
        assert approach._model_readiness(str(sandbox / "simulator.py"),
                                         []) is None
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", query)
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert card.end_note == "done"
