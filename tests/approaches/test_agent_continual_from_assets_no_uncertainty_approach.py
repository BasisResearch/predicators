"""EMPIRIC from assets without explicit uncertainty."""
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pybullet as p
import pytest

from predicators import utils
from predicators.approaches import create_approach
from predicators.approaches.base_approach import BaseApproach
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.orchestrator import run_rollout_sysid
from predicators.code_sim_learning.scene_base import SceneBase
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.run.level_players import create_level_player
from predicators.structs import Action, Dataset, Object, State, Type
from scripts.cluster_utils import generate_run_configs
from tests.agent_sdk.test_prompt_goldens import _check_golden
from tests.approaches.test_agent_continual_approach import _call, _config, \
    _result
# pylint: disable-next=unused-import
from tests.approaches.test_agent_continual_real_to_sim_approach import \
    _dispose_test_physics_clients

# pylint: disable=protected-access

CONFIG = "empiric/benchmark.yaml"
ARM = "from_assets_no_uncertainty_opus"
NAME = "agent_continual_from_assets_no_uncertainty"
# The switches No uncertainty turns off (common.yaml turns most of them on).
NO_UNCERTAINTY_OFF = (
    "continual_uncertainty_decisions",
    "agent_sim_learn_param_uncertainty",
    "agent_explorer_info_seeking",
    "agent_explorer_info_seeking_adaptive",
    "agent_explorer_info_seeking_noise_aware",
    "code_sim_learning_interval_belief",
    "code_sim_learning_carry_posterior",
    "code_sim_learning_rollout_noise_filter",
    "continual_belief_frame",
    "continual_obs_noise_declared",
)
# The switches of EMPIRIC from assets the arm refuses: all but the last read
# a belief, its joint draws or a rehearsal on them, and the reach check
# reads a gap in multiples of the declared noise.
REFUSED_SWITCHES = (
    "continual_rehearsal_gate",
    "continual_rehearsal_notes_gate",
    "continual_rehearsal_safe_plan",
    "continual_rehearsal_still_start",
    "continual_rehearsal_win_first",
    "continual_rehearsal_game_over_gate",
    "continual_practise_after_reset",
    "continual_claim_check",
    "continual_skill_preflight",
    "belief_undeclared_material_screen",
    "continual_rehearsal_mean_verdict",
    "continual_rehearsal_game_over",
    "continual_rehearsal_support_note",
    "continual_rehearsal_speed_note",
    "continual_predictive_check",
    "belief_material_lines",
    "belief_keep_undeclared_materials",
    "belief_segment_temperature",
    "continual_fit_bound_note",
    "continual_pooled_level_init",
    "continual_fit_reach_check",
)
# What this arm keeps of EMPIRIC from assets: the switches that act on the
# fit, the scene and the model's own rollouts.
FROM_ASSETS_KEPT = (
    "continual_require_model_on_test",
    "code_sim_learning_prior_spans_bounds",
    "scene_observed_sizes",
    "scene_manifest_loader",
    "continual_restage_velocities",
    "continual_belief_velocities",
)
# A model for the Boil scene the harness loads: two parameters, a scored
# reading, a mechanism and model memory, and no body of its own.
MODEL = '''
class BoilModel(SceneBase):
    AGENT_PARAM_SPECS = [ParamSpec("heat_rate", 0.01, lo=0.0, hi=0.1),
                         ParamSpec("jug_spinning_friction", 0.005, lo=0.0,
                                   hi=0.05)]
    RESIDUAL_FEATURES = {"jug": ["water_volume"]}
    MODEL_STATE_INIT = {"ticks": 0}

    @classmethod
    def update_model_state(cls, observation, model_state, params, action):
        model_state["ticks"] += 1

    def _domain_specific_step(self):
        for obj in self._objects:
            if obj.type.name == "jug":
                key = (obj.name, "water_volume")
                self._feature_store[key] += self.agent_param("heat_rate")


RESIDUAL_ENV = BoilModel
'''

_SLIDER = Object("slider0", Type("slider", ["x"]))


class _GainWorld:
    """A rollout world for the fit: ``x`` advances by ``0.01 * gain`` a
    step."""

    def __init__(self) -> None:
        self._physics_client_id = p.connect(p.DIRECT)
        self._gain = 1.0
        self._x = 0.0

    def get_physical_param_info(self) -> Dict[str, Dict[str, Any]]:
        """One parameter, as the model declares it."""
        return {
            "gain": {
                "default": 1.0,
                "lo": 0.1,
                "hi": 10.0,
                "scale": "log",
                "description": "",
            }
        }

    def apply_physical_param_overrides(self, params: Dict[str, float]) -> None:
        """Set the gain a fit asks for."""
        self._gain = float(params.get("gain", self._gain))

    def _set_state(self, state: State) -> None:
        self._x = float(state.get(_SLIDER, "x"))

    def step(self, action: Action) -> State:
        """Advance ``x`` by one step at the current gain."""
        del action
        self._x += 0.01 * self._gain
        return State({_SLIDER: np.array([self._x], dtype=float)})


def _arm_class() -> Any:
    """The class the arm's name is registered under."""
    return utils.get_registered_subclass(BaseApproach, NAME)


def _arm_flags(env_name: str = "pybullet_boil") -> Dict[str, Any]:
    cfg = next(c for c in generate_run_configs(CONFIG, False, approaches=[ARM])
               if c.env == env_name)
    # The launcher pins machine-specific output paths; tests keep their own.
    flags = {
        k: v
        for k, v in cfg.flags.items() if k not in ("log", "continual_runs_dir")
    }
    flags.update(approach=cfg.approach,
                 env=cfg.env,
                 continual_render=False,
                 continual_make_video=False)
    return flags


def _make(tmp_path: Any, **overrides: Any) -> Any:
    flags = {**_arm_flags(), **overrides}
    _config(tmp_path, **flags)
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    approach = create_approach(flags["approach"], env.predicates,
                               get_gt_options(env.get_name()), env.types,
                               env.action_space,
                               [t.task for t in env.get_train_tasks()])
    return env, approach


def test_config_is_from_assets_at_point_estimates() -> None:
    """Five settings, five seeds, parked, under its own class: No uncertainty's
    switches, the fit and scene switches of EMPIRIC from assets, and none of
    its belief or rehearsal switches."""
    cls = _arm_class()
    assert cls.__name__ == "AgentContinualFromAssetsNoUncertaintyApproach"
    assert set(cls.refused_switches) == set(REFUSED_SWITCHES)
    assert all(run.approach != NAME
               for run in generate_run_configs(CONFIG, False))
    runs = list(generate_run_configs(CONFIG, False, approaches=[ARM]))
    assert len(runs) == 25
    for run in runs:
        assert run.approach == NAME
        assert run.flags["belief_joint_draws"] == 0
        for name in NO_UNCERTAINTY_OFF:
            assert run.flags[name] is False, name
        for name in FROM_ASSETS_KEPT:
            assert run.flags[name] is True, name
        for name in REFUSED_SWITCHES + ("continual_belief_window", ):
            assert name not in run.flags, name


@pytest.mark.parametrize(
    "flag, value",
    [(name, True) for name in NO_UNCERTAINTY_OFF + REFUSED_SWITCHES] + [
        ("belief_joint_draws", 16),
        ("code_sim_learning_prior_spans_bounds", False),
    ])
def test_arm_rejects_an_uncertainty_switch(flag: str, value: Any) -> None:
    """A config that turns on a switch of the belief, the rehearsal or the
    noise declaration, or that drops the fit's range prior, fails before a run
    starts."""
    cfg = next(iter(generate_run_configs(CONFIG, False, approaches=[ARM])))
    utils.reset_config({
        **{k: v
           for k, v in cfg.flags.items() if k != "log"}, flag: value
    })
    # Contract validation must fail before constructing an environment.
    with pytest.raises(ValueError, match=flag):
        _arm_class()()


def test_fit_gives_point_estimates_and_no_belief() -> None:
    """Under the arm's flags a fit of raw noisy readings searches the declared
    range from the agent's guess, applies its point estimate and builds no
    belief."""
    utils.reset_config(_arm_flags())
    rng = np.random.default_rng(0)
    # Raw noisy readings of a slider that moves at twice the agent's guess.
    states = [
        State({
            _SLIDER:
            np.array([0.02 * step + rng.normal(0.0, 0.002)], dtype=float)
        }) for step in range(21)
    ]
    actions = [Action(np.zeros(1, dtype=np.float32)) for _ in range(20)]
    spec = ParamSpec("gain", 1.0, lo=0.1, hi=10.0, scale="log")
    outcome = run_rollout_sysid(_GainWorld(), [(states, actions)], [spec],
                                {"slider": ["x"]},
                                anchors={"gain": 1.0})
    assert outcome.num_survivors == 1
    assert outcome.fitted["gain"] == pytest.approx(2.0, rel=0.1)
    assert outcome.applied == outcome.fitted
    # No belief: no draws, no measured misfit and no basins to mix.
    assert outcome.belief is None
    assert outcome.fit_result.misfit is None
    assert len(outcome.fit_result.basins or []) <= 1


def test_prompt_offers_the_fit_and_no_noise_belief_or_rehearsal_gate(
        tmp_path: Any) -> None:
    """The system prompt is the from-assets model contract with the harness fit
    and raw observations: no noise section, no belief or draw wording beyond
    the lines that say those calls are disabled, and no rehearsal gate."""
    _, approach = _make(tmp_path)
    assert isinstance(approach, _arm_class())
    assert isinstance(approach._base_env, SceneBase)
    surface = approach._probe_surface()
    assert surface.fit and not surface.uncertainty
    assert approach.parameter_belief() is None
    prompt = approach._get_agent_system_prompt()
    _check_golden("continual_system_from_assets_no_uncertainty", prompt)
    assert "__" not in prompt.replace("__init__", "")
    # The from-assets contract, for a scene the harness loads.
    assert "## EMPIRIC from assets" in prompt
    assert "SceneBase" in prompt and "scene_manifest.json" in prompt
    assert "Do not build bodies and do not override `initialize_pybullet`" \
        in prompt
    assert "force, torque or constraint it applies" in prompt
    assert "Give each parameter to the part that owns it" in prompt
    assert "BaseSimulator" not in prompt
    assert "visible base physics" not in prompt
    assert "model or not" not in prompt
    # The harness fit is offered, and a test-level step waits for it.
    assert "sim.fit()" in prompt
    assert "the harness fits them" in prompt
    assert "### Test levels require a fitted model" in prompt
    assert "the fit and every rollout keep the value your scene gives it" \
        in prompt
    assert "A declared material's range covers at least its plausible " \
        "range" in prompt
    # Raw observations and point estimates.
    assert "## No explicit uncertainty handling" in prompt
    assert "latest observation as the current state" in prompt
    assert "one fitted value per parameter" in prompt
    assert "starts every body at rest" in prompt
    # Nothing about the noise.
    for word in ("noise", "noisy", "sigma", "denoise"):
        assert word not in prompt.lower(), word
    # A belief or a draw is named only where its call is said to be off.
    for line in prompt.splitlines():
        if "belief" in line or "draw" in line:
            assert "disabled" in line or \
                "is a belief, not a measured fact" in line, line
    # No harness-run rehearsal, and no gate, note or check built on one.
    for absent in ("P-hat", "joint rehearsal", "joint draws",
                   "### Test-level steps wait", "has the highest P-hat",
                   "A rehearsed request is refused",
                   "Start a new plan only from rest",
                   "### A reset training level asks for a rehearsal",
                   "### Claim checks",
                   "### Each rehearsed step is checked against its rehearsal",
                   "### Every skill request is rehearsed first",
                   "### State estimates, timing, and execution discrepancies",
                   "uncertainty-aware rehearsal", "belief handling",
                   "Use margins", "margin across models",
                   "what uncertainty could change"):
        assert absent not in prompt, absent
    assert "Loading a model does not itself enable automatic rehearsal" \
        in prompt


@pytest.mark.slow
def test_agent_fits_and_acts_on_a_test_level_with_no_rehearsal_gate(
        tmp_path: Any, monkeypatch: Any) -> None:
    """On a test level a skill waits for a model, then for its fit, and for
    nothing else: the model loads in the scene the harness built, sim.fit runs,
    sim rolls a plan out at the point estimate, and the skill runs."""
    env, approach = _make(tmp_path, continual_levels="test_only")
    seen: List[str] = []
    pick = "PickJug(robot:robot, jug0:jug)[0.05]"
    wait = "Wait(robot:robot)[2]"

    def query(message: str, *_args: Any, **_kwargs: Any) -> Any:
        seen.append(message)
        ctx = approach._tool_context
        sandbox = Path(ctx.sandbox_dir)
        model_file = sandbox / "simulator.py"
        assert "sim` has no world until" in message
        assert "call `sim.fit()` before you act on a test level" in message
        assert "noise" not in message.lower()
        # The model gate is installed; nothing that reads a rehearsal is.
        assert ctx.skill_gate is not None
        for name in ("rehearsal_gate", "practise_gate", "claim_check",
                     "skill_preflight", "material_screen",
                     "joint_draws_provider", "scored_features_provider",
                     "reach_note_provider", "bound_note_provider"):
            assert getattr(ctx, name) is None, name
        tools = {t.name: t for t in ctx.extra_mcp_tools}
        assert "claims_record" not in tools
        description = tools["run_python"].description
        assert "sim.fit" in description
        for word in ("P-hat", "joint draws", "belief_draws", "physics_sweep",
                     "suggest_probes", "sim.belief"):
            assert word not in description, word
        # Raw control is not gated: one step that holds the arm where it
        # is gives the fit a recording.
        session = approach._play_session
        hold = [float(q) for q in session.observe().frame.joint_positions]
        assert len(hold) == env.action_space.shape[0]
        assert "step applied" in _call(approach, "env_step", action=hold)
        # An observation carries poses and readings, and no body speed.
        frame = ctx.current_observation_provider()
        assert "body_velocities" not in (frame.simulator_state or {})
        refused = _call(approach, "skills_invoke", skill=pick)
        assert refused.startswith("ERROR"), refused
        assert "Write `./simulator.py`" in refused
        assert "Nothing was charged" in refused
        # The model adds parameters and a mechanism to the loaded scene.
        model_file.write_text(MODEL, encoding="utf-8")
        loaded = _call(approach,
                       "run_python",
                       code="sim.reset(current=True)\n"
                       "print(sorted(sim.state()))")
        assert not loaded.startswith("ERROR"), loaded
        assert "jug0:jug" in loaded
        assert type(approach._base_env).__name__ == "BoilModel"
        assert approach._tool_context.env is approach._base_env
        # Unfitted, the model's values are the agent's guesses: the gate
        # asks for a fit, and for no rehearsal.
        refused = _call(approach, "skills_invoke", skill=pick)
        assert refused.startswith("ERROR"), refused
        assert "has not been fitted" in refused and "sim.fit()" in refused
        assert "Nothing was charged" in refused
        assert "joint" not in refused and "P-hat" not in refused
        # One held step shows no motion the model explains, so this fit
        # keeps the declared values. It is still the canonical fit of the
        # file, which is what the gate asks for.
        fitted = _call(approach, "run_python", code="print(sim.fit())")
        assert not fitted.startswith("ERROR"), fitted
        assert approach._probe_fit_state().get("fit_result") is not None, \
            fitted
        assert approach._probe_fit_state().get("digest") == \
            approach._current_simulator_digest()
        # The fit leaves point estimates and no belief.
        for word in ("Parameter belief", "joint draw", "P-hat", "rehearsal"):
            assert word not in fitted, (word, fitted)
        assert approach._probe_fit_state().get("belief") is None
        assert approach.parameter_belief() is None
        assert not approach._sampled_material_specs()
        assert not approach.belief_free_specs()
        assert approach._model_readiness(str(model_file), []) is None
        # sim rolls a plan out once, at the fitted values, from the latest
        # observation; repeated trials check the controller.
        run = _call(approach,
                    "run_python",
                    code=f"r = sim.reset(current=True).run('{wait}', "
                    "render=False)\n"
                    "print(type(r).__name__, len(r.steps), "
                    "r.steps[0]['failure'])")
        assert "ProbeResult 1 None" in run, run
        trials = _call(approach,
                       "run_python",
                       code=f"r = sim.reset(current=True).run('{wait}', "
                       "trials=2)\nprint(type(r).__name__)")
        assert "ProbeTrialsResult" in trials, trials
        # Nothing draws from a belief.
        for code, why in (
            (f"sim.run('{wait}', draws=0)", "draws= needs the joint belief"),
            (f"sim.run('{wait}', physics_sweep=True)",
             "Explicit uncertainty is disabled"),
            (f"sim.run('{wait}', belief_draws=4)",
             "Explicit uncertainty is disabled"),
            ("sim.belief()", "Explicit uncertainty is disabled"),
            (f"sim.suggest_probes('{wait}')",
             "Disagreement-based probing is disabled"),
        ):
            out = _call(approach, "run_python", code=code)
            assert why in out, (code, out)
        assert not ctx.live_rehearsals
        # Loaded and fitted, the model opens the gate: the skill runs with
        # no rehearsal of it on record.
        before = session.observe().ledger.run_steps
        out = _call(approach, "skills_invoke", skill=pick)
        assert "Nothing was charged" not in out, out
        assert "rehears" not in out.lower(), out
        assert session.observe().ledger.run_steps > before
        # An edit closes the gate until the next fit; the fitted file
        # opens it again.
        with open(model_file, "a", encoding="utf-8") as f:
            f.write("\n# edited\n")
        refused = _call(approach, "skills_invoke", skill=pick)
        assert "has not been fitted" in refused, refused
        model_file.write_text(MODEL, encoding="utf-8")
        assert approach._model_readiness(str(model_file), []) is None
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", query)
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert card.end_note == "done"
    assert len(seen) == 1
    assert card.levels[0].split == "test" and card.levels[0].steps > 1
    # The round's end deploys the fitted file at its point estimates.
    assert approach._last_round_modelled
    assert approach._last_fit_result is not None
    assert "UNFITTED" not in approach._fit_status_text()
    assert approach.parameter_belief() is None
