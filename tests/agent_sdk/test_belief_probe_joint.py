"""Tests for the probe under the joint belief (``belief_joint_draws`` > 0).

``sim.run(plan)`` rehearses the plan on K joint draws of the belief and
reports the success estimate, ``sim.refine`` scores its proposals on the
common draws, and ``sim.suggest_probes`` ranks experiments by the
information their subgoal readings carry about the parameters (paper
Sections 3.3 and 3.4, Appendix B.4).
"""
# pylint: disable=protected-access
import re
from types import SimpleNamespace

import numpy as np
import pytest

from predicators import utils
from predicators.agent_sdk.belief_probe import BeliefProbe, ProbeJointResult, \
    ProbeResult, _line_states, _outcome_leads, _separating_ranges
from predicators.agent_sdk.plan_execution import StepOutcome
from predicators.structs import GroundAtom, Object, State, Task, \
    TaskEvaluator, Type
from tests.agent_sdk.test_belief_probe_physics_sweep import _block, \
    _make_ctx, _Move, _ReachedHi

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
    # The rehearsal gate's record: this plan, from this observation.
    assert len(ctx.live_rehearsals) == 1
    rehearsal = ctx.live_rehearsals[0]
    assert rehearsal.observation is ctx.current_observation
    assert rehearsal.plan_text == _PLAN
    assert rehearsal.p_hat == pytest.approx(0.5)
    # Each draw's state after the plan's one line, which the predictive
    # check scores the real outcome against; no fitted program yet.
    xs = [states[0].get(_block, "x") for states in rehearsal.line_states]
    assert xs[0] == pytest.approx(0.95) and xs[2] == pytest.approx(0.95)
    assert xs[1] != pytest.approx(0.95) and xs[3] != pytest.approx(0.95)
    assert rehearsal.basis is None
    # And the rollout at the belief's mean, line by line.
    assert len(rehearsal.mean_states) == 1
    assert rehearsal.mean_states[0] is not None
    assert rehearsal.mean_states[0].get(_block, "x") == pytest.approx(0.95)
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
    assert "P-hat = 0.50 +- 0.25 (2/4 succeeded; each draw pairs" in text
    assert "draw 1 [friction=0.5]: goal NOT reached" in text
    assert "no single parameter separates" in text
    assert "Step-by-step rollout from the belief mean" in text
    utils.reset_config({})


def test_line_states_skip_lines_a_draw_did_not_run():
    """A line a draw ran cleanly leaves its state; one whose controller failed
    there, or that was not initiable, leaves none: the draw did not rehearse
    it."""
    option = _Move.ground([_block], np.array([0.95], dtype=np.float32))
    start = State({_block: np.array([0.2])})
    moved = State({_block: np.array([0.95])})
    states = _line_states([
        StepOutcome(option, start, moved, 12, True, None, None),
        StepOutcome(option, moved, moved, 0, True, "0 actions", None),
        StepOutcome(option, moved, None, 0, False, "not initiable", None),
    ])
    assert len(states) == 3
    assert states[0] is not None
    assert states[0].get(_block, "x") == pytest.approx(0.95)
    assert states[1] is None and states[2] is None


def test_rehearsal_modes_and_replaced_flags():
    """draws=0 is one rollout; trials, solved and belief_draws are replaced by
    the rehearsal; without the joint belief draws= is refused."""
    ctx, _, _, calls = _joint_ctx()
    sim = BeliefProbe(ctx)
    _on_observation(sim, ctx)
    single = sim.run(_PLAN, render=False, draws=0)
    assert isinstance(single, ProbeResult) and not calls
    assert not ctx.live_rehearsals  # the point estimate is not a rehearsal
    replaced = [{"trials": 2}, {"solved": True, "trials": 2}]
    replaced.append({"belief_draws": 3})
    for flags in replaced:
        with pytest.raises(ValueError, match="replaced by the rehearsal"):
            sim.run(_PLAN, render=False, **flags)
    utils.reset_config({"belief_joint_draws": 0})
    with pytest.raises(ValueError, match="needs the joint belief"):
        sim.run(_PLAN, render=False, draws=0)
    utils.reset_config({})


def test_rehearsal_names_draws_that_did_not_start_from_their_state():
    """A draw whose world reads a feature back differently from the state it
    was set to ran from another state; the rehearsal says so and names the
    feature, leaving aside the robot's joints, which inverse kinematics
    restores only approximately."""
    ctx, model, _, _ = _joint_ctx()
    robot = Object("robot", Type("robot", ["x"]))
    model.sim_env = SimpleNamespace(
        _last_unreconstructible_features=[(_block, "x"), (robot, "x")])
    sim = BeliefProbe(ctx)
    _on_observation(sim, ctx)
    res = sim.run(_PLAN, render=False)
    assert [d["unreconstructed"] for d in res.draws] == [["block0.x"]] * 4
    assert "4 of 4 draws did not start from the state they were given" in \
        res.text
    assert "block0.x" in res.text and "robot.x" not in res.text
    del model.sim_env
    _on_observation(sim, ctx)
    assert "did not start from the state" not in sim.run(_PLAN,
                                                         render=False).text
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
    assert not ctx.live_rehearsals  # a staged state is not the observation
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


def test_outcome_leads_name_what_the_failing_draws_lean_on():
    """When no parameter separates the failing draws outright, the report names
    the ones they lean on, marks the undeclared ones, and says how often
    shuffled outcomes lean as far."""
    # Eight failing then eight passing draws. mu sits lower on the failing
    # draws, mat higher, each with an overlap, and noise follows neither.
    mu = [0.1, 0.15, 0.2, 0.22, 0.25, 0.3, 0.33, 0.45
          ] + [0.35, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    mat = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.5
           ] + [0.2, 0.3, 0.35, 0.4, 0.45, 0.55, 0.65, 0.3]
    noise = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6
             ] + [0.15, 0.85, 0.25, 0.75, 0.35, 0.65, 0.45, 0.55]
    draws = [{
        "params": {
            "mu": mu[k],
            "mat": mat[k],
            "noise": noise[k]
        },
        "success": k >= 8
    } for k in range(16)]
    assert not _separating_ranges(draws)
    lines = _outcome_leads(draws, {"mat"})
    assert lines[0].startswith("The failing draws lean on these parameters")
    assert lines[1] == ("  mu: AUC 0.03, failing draws lower (median 0.235 "
                        "against 0.575)")
    assert lines[2] == ("  mat*: AUC 0.95, failing draws higher (median "
                        "0.775 against 0.375)")
    assert "not declared in simulator.py" in lines[3]
    shuffled = re.match(
        r"  Shuffling which draws failed leans some parameter this far in "
        r"(\d+)% of 400 shuffles", lines[4])
    assert shuffled is not None and int(shuffled.group(1)) <= 5
    assert len(lines) == 5
    # Declared parameters get no mark and no footnote.
    assert not any("*" in line for line in _outcome_leads(draws, set()))
    # Too few failing draws, or no parameter leaning far enough, give none.
    assert not _outcome_leads(draws[6:], {"mat"})
    only_noise = [{
        "params": {
            "noise": d["params"]["noise"]
        },
        "success": d["success"]
    } for d in draws]
    assert not _outcome_leads(only_noise, set())
    # The report prints them in place of the no-separator line.
    report = repr(
        ProbeJointResult([
            dict(d,
                 solved=d["success"],
                 goal_reached=d["success"],
                 failure=None,
                 num_actions=5) for d in draws
        ],
                         0.5,
                         0.12,
                         True,
                         leads=lines))
    assert lines[1] in report
    assert "no single parameter separates" not in report


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
    # The headline says the goal atoms held on two of the failing draws.
    assert res.splitlines()[0].startswith(
        "Rehearsal on 4 joint draws of the belief: P-hat = 0.00 +- 0.00 "
        "(0/4 succeeded; the goal atoms held on 2 of the failing draws, "
        "which the task's evaluator did not count as solved;")
    # The evaluator ends the episode unsolved on those two, as a real
    # episode ends in GAME_OVER; under continual_rehearsal_game_over the
    # report counts them and marks each.
    assert [d["lost"] for d in res.draws] == [True, False, True, False]
    assert "GAME_OVER" not in repr(res)
    utils.update_config({"continual_rehearsal_game_over": True})
    assert "; 2 of the 4 failing draws the task's evaluator scored end the " \
        "episode in GAME_OVER" in res.splitlines()[0]
    assert sum("NOT solved, GAME_OVER" in line
               for line in res.splitlines()) == 2
    # A step that ends no draw's episode says so.
    safe = ProbeJointResult([
        dict(d, lost=False, goal_reached=False, failure=None)
        for d in res.draws
    ], 0.0, 0.0, True)
    assert "; none of the 4 failing draws the task's evaluator scored " \
        "ends the episode in GAME_OVER" in safe.splitlines()[0]
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
