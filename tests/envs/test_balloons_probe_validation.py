"""Regression for a turning point misclassified as a balloons jam."""
# pylint: disable=protected-access
from predicators import utils
from predicators.envs.pybullet_balloons import BalloonsProbeOutcome, \
    PyBulletBalloonsEnv


def test_saved_noisy_seed_is_ambiguous_and_recoverable():
    """Both decoy release orders win on the exact diagnosed task geometry."""
    utils.reset_config({
        "env": "pybullet_balloons",
        "seed": 1,
        "skill_phase_use_motion_planning": False
    })
    env = PyBulletBalloonsEnv(use_gui=False)
    try:
        state = env.level_state(1, [0, 1, 2, 3],
                                (.5051171428571427, .5551171428571428))
        outcome = env.assess_subset(state, (0, 2))
        assert outcome.won
        assert outcome.steps > 22  # the old helper stopped at step 22
        assert not outcome.jammed
        assert env.subset_outcome(state, (3, )) == (True, False)
        candidates = env.candidate_outcomes(state)
        assert sum(
            any(o.won for o in results)
            for results in candidates.values()) >= 2
        assert env.solution_subset(state) == (3, )
        assert not any(o.burst or o.jammed for results in candidates.values()
                       for o in results)  # rejected as a test task: no decoy
        for order in ((0, 2), (2, 0), (3, )):
            assert env.release_sequence_outcome(state, order).won
    finally:
        env.dispose()


def test_short_probe_is_unresolved_not_a_jam():
    """A finite horizon is not evidence of settled failure."""
    utils.reset_config({
        "env": "pybullet_balloons",
        "seed": 1,
        "balloons_probe_max_steps": 2
    })
    env = PyBulletBalloonsEnv(use_gui=False)
    try:
        state = env.level_state(1, [0, 1, 2, 3], (.505, .555))
        outcome = env.assess_subset(state, (0, 2))
        assert outcome.status == "unresolved"
        assert not outcome.jammed
        assert not outcome.won
    finally:
        env.dispose()


def test_off_target_rest_needs_a_sustained_window():
    """A settled overshoot is a failure, without an unsupported jam label."""
    utils.reset_config({"env": "pybullet_balloons", "seed": 1})
    env = PyBulletBalloonsEnv(use_gui=False)
    try:
        state = env.level_state(1, [0, 1, 2, 3], (.505, .555))
        outcome = env.assess_subset(state, (1, 2))
        assert outcome.status == "resting_outside"
        assert abs(outcome.height - .6) < 1e-3
        assert outcome.steps >= 20
        assert not outcome.jammed
    finally:
        env.dispose()


def test_wall_contact_alone_does_not_establish_a_jam():
    """A contact label needs a successful wall-free counterfactual."""
    contact = BalloonsProbeOutcome("resting_outside", 100, .49, .001, True)
    assert not contact.jammed
    causal = BalloonsProbeOutcome("resting_outside", 100, .49, .001, True,
                                  True)
    assert causal.jammed
