"""Weighted information scores agree with enumerated observation channels."""

import numpy as np
import pytest

from predicators.code_sim_learning.active_experiment import \
    mean_bernoulli_entropy, noisy_read_information


def _enumerated_information(channel, weights):
    # Sum P(member, read) log(P(member, read) / (P(member) P(read))).
    scores = []
    for probabilities in np.asarray(channel).T:
        conditional = np.column_stack((1. - probabilities, probabilities))
        joint = np.asarray(weights)[:, None] * conditional
        independent = np.asarray(weights)[:, None] * joint.sum(axis=0)
        positive = joint > 0.
        scores.append(
            np.sum(joint[positive] *
                   np.log2(joint[positive] / independent[positive])))
    return float(np.mean(scores))


def test_weighted_noisy_reads_match_enumerated_joint_law():
    """Unequal member mass enters both marginal and conditional entropy."""
    weights = [.1, .3, .6, 0.]
    channel = np.array([[.1, 0., .5], [.8, 1., .5], [.6, 0., .5], [1., 1.,
                                                                   0.]])
    score = noisy_read_information(channel, weights=weights)
    assert score == pytest.approx(_enumerated_information(channel, weights))
    assert score != pytest.approx(noisy_read_information(channel))
    assert score == pytest.approx(
        noisy_read_information(channel[:3], weights=weights[:3]))
    # Splitting a member into identical copies cannot create information.
    split_channel = channel[[0, 1, 2, 2]]
    assert score == pytest.approx(
        noisy_read_information(split_channel, weights=[.1, .3, .2, .4]))
    assert score == pytest.approx(
        noisy_read_information(channel[::-1], weights=weights[::-1]))


def test_weighted_exact_reads_and_uninformative_channel():
    """Exact reads recover entropy and identical channels carry no signal."""
    exact = np.array([[True, False], [False, True], [False, False]])
    weights = [.1, .3, .6]
    reference = _enumerated_information(exact.astype(float), weights)
    assert mean_bernoulli_entropy(exact, weights=weights) == \
        pytest.approx(reference)
    assert noisy_read_information(exact, weights=weights) == \
        pytest.approx(reference)
    assert noisy_read_information(np.tile([.2, .8], (3, 1)),
                                  weights=weights) == pytest.approx(0.,
                                                                    abs=1e-15)
    assert noisy_read_information(exact, weights=[1., 0., 0.]) == 0.
    assert mean_bernoulli_entropy(exact, weights=[1., 0., 0.]) == 0.
    assert noisy_read_information(np.zeros((3, 0)), weights=weights) == 0.
    assert mean_bernoulli_entropy(np.zeros((3, 0)), weights=weights) == 0.


@pytest.mark.parametrize("weights",
                         [[.5], [.2, .2], [-.1, 1.1], [float("nan"), 1.],
                          [float("inf"), 0.], [[.5], [.5]], [0., 0.]])
def test_weighted_scores_reject_invalid_measure(weights):
    """Neither invalid masses nor a wrong row count are silently repaired."""
    for score in (mean_bernoulli_entropy, noisy_read_information):
        with pytest.raises(ValueError, match="weight"):
            score(np.array([[0.], [1.]]), weights=weights)


@pytest.mark.parametrize(
    "channel", [[[float("nan")], [1.]], [[-.1], [1.]], [[0.], [1.1]], [0., 1.],
                np.empty((0, 0))])
def test_weighted_scores_reject_invalid_channel(channel):
    """Probability errors and empty ensembles fail before reporting a score."""
    for score in (mean_bernoulli_entropy, noisy_read_information):
        with pytest.raises(ValueError):
            score(np.asarray(channel), weights=[.5, .5])
    with pytest.raises(ValueError, match="binary"):
        mean_bernoulli_entropy(np.array([[.2], [.8]]), weights=[.5, .5])


def test_equal_weights_recover_incumbent_scores():
    """Explicit uniform weighting has the same statistical interpretation."""
    rng = np.random.default_rng(13)
    for members in (1, 2, 7, 32):
        probabilities = rng.uniform(size=(members, 4))
        exact = probabilities > .5
        weights = np.full(members, 1. / members)
        for matrix, score in ((probabilities, noisy_read_information),
                              (exact, mean_bernoulli_entropy)):
            assert score(matrix,
                         weights=weights) == pytest.approx(score(matrix))
