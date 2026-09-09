"""Gate tests for the shared domino domain (PLAN.md section 4.4).

This domain comes from the master branch, so these tests do not check its
design; they check that OUR interface can solve it - a Cartesian ``move_to``
pick-and-place plus a push, under our torque limits and contact-force reflex -
and that our results record the domain's own certificate, which rejects
successes our ``goal_reached`` metric would happily credit.
"""
import pytest

from agent_robot_control.experiments.domino_oracle import run_oracle


@pytest.mark.parametrize("seed", [0, 1])
def test_oracle_bridges_and_certifies(seed):
    r = run_oracle(seed)
    assert r["goal_reached"], r
    assert r["certified"], r["reason"]
    # One blue consumed, so the reward is the success bonus minus one block.
    assert r["reward"] == pytest.approx(0.95), r
    # A solve is cheap in interactions: the difficulty is in the layout.
    assert r["interactions"] < 1000, r


def test_arm_shove_reaches_the_goal_but_does_not_certify():
    """The reason domino results carry the domain's verdict as well as ours:
    shoving the target over with the gripper satisfies the goal atom."""
    r = run_oracle(0, mode="cheat")
    assert r["goal_reached"]
    assert not r["certified"]
    assert r["reward"] <= 0.0
    assert "counterfactual" in r["reason"]


def test_a_gap_too_wide_to_bridge_stays_standing():
    r = run_oracle(0, mode="push_only")
    assert not r["goal_reached"]
    assert r["reward"] == pytest.approx(0.0)
