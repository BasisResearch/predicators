"""Sequential density integration versus exact joint history references."""
import itertools
import math

import numpy as np
import pytest

from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError
from predicators.code_sim_learning.inference_path_integral import \
    summarize_path_integral
from predicators.code_sim_learning.inference_sequential import \
    integrate_sequential_paths


def test_joint_markov_density_and_terminal_measure():
    """Enumeration checks dependence, evidence and terminal weighted paths."""
    observations = (0, 1, 1, 0, 1, 0)
    transition = np.array([[.85, .15], [.3, .7]])
    emission = np.array([[.8, .2], [.1, .9]])
    evidence = 0.
    final_one = 0.
    for path in itertools.product((0, 1), repeat=len(observations)):
        density = .5
        for step, (state, reading) in enumerate(zip(path, observations)):
            density *= emission[state, reading]
            if step:
                density *= transition[path[step - 1], state]
        evidence += density
        final_one += density * path[-1]

    def advance(parent, stage, rng):
        probability = .5 if parent is None else transition[parent[-1], 1]
        state = int(rng.random() < probability)
        history = (() if parent is None else parent) + (state, )
        return history, math.log(emission[state, observations[stage]])

    results = [
        integrate_sequential_paths(advance, 512, len(observations), seed)
        for seed in range(24)
    ]
    summary = summarize_path_integral(tuple(r.log_density for r in results))
    assert math.exp(summary.log_density) == pytest.approx(evidence, rel=.02)
    means = [
        sum(w * history[-1] for history, w in zip(r.histories, r.weights))
        for r in results
    ]
    assert np.mean(means) == pytest.approx(final_one / evidence, abs=.015)
    assert results[0] == integrate_sequential_paths(advance, 512,
                                                    len(observations), 0)
    assert all(len(h) == len(observations) for h in results[0].histories)


def test_proposal_correction_and_zero_paths():
    """A guided proposal retains its normalizer and zero event outcomes."""
    target, proposal = .2, .7

    def advance(parent, stage, rng):
        del parent
        state = int(rng.random() < proposal)
        # Exact event observes state=1 in each of three independent blocks.
        return (stage, state), math.log(target / proposal) if state else \
            -math.inf

    estimates = [
        integrate_sequential_paths(advance, 1000, 3, seed).log_density
        for seed in range(12)
    ]
    estimate = summarize_path_integral(tuple(estimates))
    assert math.exp(estimate.log_density) == pytest.approx(target**3, rel=.03)


def test_ancestry_and_mutable_parent_isolation():
    """Sibling extensions cannot mutate a parent or conceal prior collapse."""
    calls = [0, 0]

    def advance(parent, stage, rng):
        del rng
        index = calls[stage]
        calls[stage] += 1
        if stage == 0:
            return [index], 0. if index == 0 else -math.inf
        assert parent == [0]
        parent.append(index)
        return parent, 10000.

    result = integrate_sequential_paths(advance, 8, 2, 10)
    assert result.log_density == pytest.approx(10000. - math.log(8))
    assert result.effective_terms == (1., 8.)
    assert result.surviving_ancestors == (1, 1)
    assert result.histories == tuple([0, i] for i in range(8))
    assert result.weights == (.125, ) * 8


def test_no_support_and_numerical_errors():
    """Unseen support is distinct from an invalid or interrupted replay."""

    def absent(parent, stage, rng):
        del parent, stage
        return (), 0. if rng.random() < 1e-12 else -math.inf

    result = integrate_sequential_paths(absent, 8, 4, 10)
    assert result.status == "no_sample_support"
    assert not result.histories and not result.weights
    assert result.log_increments == (-math.inf, )
    for invalid in (math.inf, math.nan):
        with pytest.raises(ConditioningNumericalError):
            integrate_sequential_paths(lambda *args, value=invalid:
                                       ((), value),
                                       8,
                                       2,
                                       0)
    calls = []

    def interrupted(parent, stage, rng):
        del parent, stage, rng
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("native replay failed")
        return (), 0.

    with pytest.raises(RuntimeError, match="native replay failed"):
        integrate_sequential_paths(interrupted, 8, 2, 0)
    assert len(calls) == 3
    with pytest.raises(ConditioningNumericalError, match="overflow"):
        integrate_sequential_paths(lambda *args: ((), 1e308), 2, 2, 0)
    for count, stages, seed in ((1, 1, 0), (2, 0, 0), (2, 1, -1), (True, 1,
                                                                   0)):
        with pytest.raises(ValueError):
            integrate_sequential_paths(absent, count, stages, seed)
