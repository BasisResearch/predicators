"""Tests for the probe under the joint belief (``belief_joint_draws`` > 0).

``sim.run(plan)`` rehearses the plan on K joint draws of the belief and
reports the success estimate, ``sim.refine`` scores its proposals on the
common draws, and ``sim.suggest_probes`` ranks experiments by the
information their subgoal readings carry about the parameters (paper
Sections 3.3 and 3.4, Appendix B.4).
"""
# pylint: disable=protected-access
import numpy as np
import pytest

from predicators import utils
from predicators.agent_sdk.belief_probe import BeliefProbe, ProbeJointResult, \
    ProbeResult, _separating_ranges
from predicators.structs import GroundAtom, Task, TaskEvaluator
from tests.agent_sdk.test_belief_probe_physics_sweep import _block, \
    _make_ctx, _ReachedHi

_PLAN = "Move(block0:block)[0.95]"
# Two of the four draws fall inside the model's friction hole, where the
# move does nothing and the goal is never reached.
_FRICTIONS = (0.3, 0.5, 0.6, 0.495)


def _joint_ctx(frictions=_FRICTIONS):
    utils.reset_config({
        "belief_joint_draws": len(frictions),
        "continual_uncertainty_decisions": True,
    })
    ctx, model, scope_overrides = _make_ctx([])
    calls = []

    def provider(num, fresh=False):
        calls.append((num, fresh))
        start = ctx.current_observation
        assert start is not None
        return [({"friction": f}, start.copy()) for f in frictions][:num]

    ctx.joint_draws_provider = provider
    ctx.physical_param_names_provider = lambda: {"friction"}
    return ctx, model, scope_overrides, calls


def _on_observation(sim, ctx):
    sim.reset()
    ctx.current_observation = sim._require_state()


def test_rehearsal_reports_p_hat_draws_and_the_mean_rollout():
    """run(plan) rolls the plan on every joint draw, then once from the belief
    mean, which alone advances the state."""
    ctx, _, scope_overrides, calls = _joint_ctx()
    sim = BeliefProbe(ctx)
    _on_observation(sim, ctx)
    res = sim.run(_PLAN, render=False)
    assert isinstance(res, ProbeJointResult)
    assert calls == [(4, False)]
    assert [d["params"]["friction"] for d in res.draws] == list(_FRICTIONS)
    assert [d["success"] for d in res.draws] == [True, False, True, False]
    assert res.p_hat == pytest.approx(0.5)
    assert res.stderr == pytest.approx(np.sqrt(0.25 / 4))
    assert res.from_observation
    # Each draw ran in the fresh scope at its own physical parameters.
    assert scope_overrides[:4] == [{"friction": f} for f in _FRICTIONS]
    assert all(d["solved"] is None for d in res.draws)  # no evaluator
    assert [d["planner_seed"] for d in res.draws] == \
        [utils.CFG.seed + k for k in range(4)]
    assert isinstance(res.mean, ProbeResult)
    assert sim._require_state().get(_block, "x") == pytest.approx(0.95)
    # The mean rollout's fields read through; string methods stay on the
    # rehearsal's report.
    assert res.final_state is res.mean.final_state
    assert res.goal_reached is res.mean.goal_reached
    assert res.splitlines()[0].startswith("Rehearsal on 4 joint draws")
    with pytest.raises(AttributeError, match="no attribute 'nonsense'"):
        _ = res.nonsense
    text = res.text
    assert "P-hat = 0.50 +- 0.25 (2/4 succeeded" in text
    assert "draw 1 [friction=0.5]: goal NOT reached" in text
    assert "no single parameter separates" in text
    assert "Step-by-step rollout from the belief mean" in text
    utils.reset_config({})


def test_rehearsal_modes_and_replaced_flags():
    """draws=0 is one rollout; trials, solved and belief_draws are replaced by
    the rehearsal; without the joint belief draws= is refused."""
    ctx, _, _, calls = _joint_ctx()
    sim = BeliefProbe(ctx)
    _on_observation(sim, ctx)
    single = sim.run(_PLAN, render=False, draws=0)
    assert isinstance(single, ProbeResult) and not calls
    replaced = [{"trials": 2}, {"solved": True, "trials": 2}]
    replaced.append({"belief_draws": 3})
    for flags in replaced:
        with pytest.raises(ValueError, match="replaced by the rehearsal"):
            sim.run(_PLAN, render=False, **flags)
    utils.reset_config({"belief_joint_draws": 0})
    with pytest.raises(ValueError, match="needs the joint belief"):
        sim.run(_PLAN, render=False, draws=0)
    utils.reset_config({})


def test_hypothetical_state_varies_only_the_parameters():
    """Off the current observation, the draws keep the probe's own state."""
    ctx, _, _, _ = _joint_ctx()
    sim = BeliefProbe(ctx)
    _on_observation(sim, ctx)
    sim.reset(mods={"block0": {"x": 0.2}})
    res = sim.run(_PLAN, render=False)
    assert not res.from_observation
    assert "varies the parameters" in res.text
    utils.reset_config({})


class _SettleEnv:
    """Records the states a rollout asks its engine to settle."""

    def __init__(self):
        self.calls = []

    def settle_state(self, state):
        """Return ``state`` unchanged, recording it."""
        self.calls.append(state)
        return state


def test_state_draws_settle_under_a_noisy_channel():
    """Under observation noise each draw starts after its engine settles it;
    exact observations and hypothetical states are not settled."""
    for noise, expected in ((0.01, 4), (0.0, 0)):
        ctx, model, _, _ = _joint_ctx()
        utils.reset_config({
            "belief_joint_draws": 4,
            "continual_uncertainty_decisions": True,
            "continual_obs_noise_position": noise,
        })
        model.sim_env = _SettleEnv()
        sim = BeliefProbe(ctx)
        _on_observation(sim, ctx)
        sim.run(_PLAN, render=False)
        assert len(model.sim_env.calls) == expected
        if expected:
            assert all(
                s.allclose(ctx.current_observation)
                for s in model.sim_env.calls)
            model.sim_env.calls.clear()
            sim.reset(mods={"block0": {"x": 0.2}})
            sim.run(_PLAN, render=False)
            assert not model.sim_env.calls
    utils.reset_config({})


def test_separating_ranges_name_the_failing_side():
    """A parameter whose failing draws all sit on one side is reported."""
    draws = [{
        "params": {
            "mu": mu,
            "m": m
        },
        "success": ok
    } for mu, m, ok in ((0.2, 1.0, False), (0.25, 3.0, False),
                        (0.4, 2.0, True), (0.5, 1.5, True))]
    lines = _separating_ranges(draws)
    assert lines == [
        "mu: every failing draw has mu <= 0.25; passing draws span "
        "[0.4, 0.5]"
    ]
    assert not _separating_ranges([dict(d, success=True) for d in draws])


class _PrefixEvaluator(TaskEvaluator):
    """Solved only when the episode began with a Move and reached the goal:

    a whole-trajectory rule the recorded prefix must satisfy.
    """

    def _certify(self, states, step_options, sim_env=None):
        del sim_env
        first = step_options[0] if step_options else None
        if first is None or first[0] != "Move":
            return False, "the episode did not start with a Move"
        return True, ""


def test_draws_are_scored_on_the_recorded_prefix_and_their_rollout():
    """P-hat applies the task evaluator to the episode so far followed by each
    draw's rollout, so a rule on the prefix decides the verdict."""
    ctx, _, _, _ = _joint_ctx()
    ctx.probe_engine_available = True
    evaluator = _PrefixEvaluator({GroundAtom(_ReachedHi, [_block])})
    assert ctx.current_task is not None
    ctx.current_task = Task(ctx.current_task.init,
                            ctx.current_task.goal,
                            evaluator=evaluator)
    sim = BeliefProbe(ctx)
    _on_observation(sim, ctx)
    start = sim._require_state()
    ctx.episode_prefix_provider = lambda: ([start.copy(
    ), start.copy()], [("Move", ("block0", ), (0.0, ))])
    res = sim.run(_PLAN, render=False)
    assert [d["solved"] for d in res.draws] == [True, False, True, False]
    assert res.p_hat == pytest.approx(0.5)
    # Without the Move in the prefix no draw is solved, even where the
    # rollout reaches the goal. The mean rollout advanced the probe, so
    # start again from the observation.
    sim.reset()
    ctx.episode_prefix_provider = lambda: ([start.copy(),
                                            start.copy()], [None])
    res = sim.run(_PLAN, render=False)
    assert [d["solved"] for d in res.draws] == [False] * 4
    assert [d["goal_reached"] for d in res.draws] == \
        [True, False, True, False]
    assert res.p_hat == 0.0
    utils.reset_config({})


def test_refine_scores_proposals_on_the_joint_draws():
    """refine keeps the best proposal on the common draws and reports its
    estimate on fresh draws."""
    ctx, _, _, calls = _joint_ctx()
    sim = BeliefProbe(ctx)
    _on_observation(sim, ctx)
    res = sim.refine(f"{_PLAN} -> {{ReachedHi(block0:block)}}",
                     timeout=2.0,
                     require_goal=True)
    assert res.success
    assert "Scored " in res.note and "fresh draws" in res.note
    assert calls and calls[-1] == (4, True)
    assert all(num == 4 for num, _ in calls)
    utils.reset_config({})


def test_suggest_probes_ranks_by_information_over_the_joint_draws():
    """Each feasible alternative is rolled out on the joint draws and scored by
    the information its subgoal reading carries; a missing plan gets a usable
    error."""
    ctx, _, _, calls = _joint_ctx()
    sim = BeliefProbe(ctx)
    _on_observation(sim, ctx)
    res = sim.suggest_probes(f"{_PLAN} -> {{ReachedHi(block0:block)}}",
                             max_draws=3)
    assert res.measure == "information"
    [step] = res.suggestions
    # Half the draws read the subgoal true and half false, with exact
    # readings: one bit about the parameters.
    assert step.nominal_score == pytest.approx(1.0)
    assert all(score == pytest.approx(1.0) for _, score, _ in step.candidates)
    assert "information" in res.text
    assert calls and all(fresh is False for _, fresh in calls)
    with pytest.raises(ValueError, match="needs the plan"):
        sim.suggest_probes()
    utils.reset_config({})
