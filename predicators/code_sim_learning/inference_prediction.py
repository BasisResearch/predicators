"""Offline future observations from complete, assessed joint posterior rows.

Replay receives the fitted episode and requested actions, never future
readings. One joint particle generates an entire future history. The
output model is fixed across particles; stochastic physical transitions
need separate integration and are not supplied by this deterministic
replay adapter. No production planner or parameter publisher uses it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from predicators.code_sim_learning.inference_assessment import \
    AssessedInference, validated_posterior
from predicators.code_sim_learning.inference_conditioning import \
    ConditioningNumericalError, UnsupportedConditioning
from predicators.code_sim_learning.inference_data import EpisodeData, \
    InferenceData, Observation
from predicators.code_sim_learning.inference_observation import \
    OutputObservationModel
from predicators.code_sim_learning.inference_sampling import BatchPosterior

Actions = Tuple[Tuple[float, ...], ...]
History = Tuple[Observation, ...]
JointReplay = Callable[[Dict[str, float], EpisodeData, Actions], History]


def _forecast_context(
    assessment: AssessedInference, data: InferenceData, episode_id: str,
    model: OutputObservationModel, future_actions: Actions
) -> Tuple[BatchPosterior, EpisodeData, History, Actions]:
    """Validate provenance and construct the complete primitive-step prefix."""
    posterior = validated_posterior(assessment)
    if posterior is None:
        raise ValueError("Joint posterior is " + assessment.availability)
    if data.digest != posterior.identity.data or \
            model.digest != posterior.identity.sensor:
        raise ValueError("Forecast data or output-model identity differs")
    episodes = {episode.episode_id: episode for episode in data.episodes}
    if episode_id not in episodes:
        raise ValueError("Forecast requires a fitted reset episode")
    episode = episodes[episode_id]
    # Reuse the ledger's validation for finite, fixed-dimension actions.
    combined = EpisodeData(episode_id, episode.actions + tuple(future_actions),
                           episode.observations)
    actions = combined.actions[len(episode.actions):]
    observed = {o.step: o for o in episode.observations}
    prefix = tuple(
        observed.get(i, Observation(i, ()))
        for i in range(len(episode.actions) + 1))
    return posterior, episode, prefix, actions


@dataclass(frozen=True)
class JointForecast:
    """A weighted forecast with fitting provenance and retained diagnostics.

    Histories correspond to positive-weight source rows in their
    original order. A failed replay cannot be removed and its mass
    renormalized. Construction validates a supported prefix for every
    retained history; it does not prove that the replay callback
    implemented the stated physics, or that the declared assessment
    protocol was sufficient.
    """
    assessment: AssessedInference
    data: InferenceData
    episode_id: str
    model: OutputObservationModel
    future_actions: Actions
    histories: Tuple[History, ...]

    def __post_init__(self) -> None:
        posterior, _, prefix, actions = _forecast_context(
            self.assessment, self.data, self.episode_id, self.model,
            self.future_actions)
        histories = tuple(tuple(history) for history in self.histories)
        indices = tuple(i for i, w in enumerate(posterior.weights) if w > 0)
        if len(histories) != len(indices):
            raise ValueError(
                "One history per positive-weight joint row needed")
        expected = list(range(len(prefix) + len(actions)))
        for history in histories:
            if [o.step for o in history] != expected:
                raise ValueError("Forecast history must cover every action")
            score = self.model.log_likelihood(history[:len(prefix)], prefix)
            if score == -math.inf:
                raise UnsupportedConditioning(
                    "Positive-weight forecast has a zero-likelihood prefix")
            if not math.isfinite(score):
                raise ConditioningNumericalError("Nonfinite forecast prefix")
        object.__setattr__(self, "future_actions", actions)
        object.__setattr__(self, "histories", histories)

    @classmethod
    def replay(cls, assessment: AssessedInference, data: InferenceData,
               episode_id: str, model: OutputObservationModel,
               future_actions: Actions, replay: JointReplay) -> JointForecast:
        """Replay every positive-mass row, preserving all joint coordinates.

        Each callback receives an owned coordinate dictionary, the
        fitted reset episode, and only requested future actions. It must
        create a fresh candidate world and dispose it, retain inferred
        memory and initial-state dependence, and return initial plus all
        later frames. Callback exceptions abort construction without
        changing weights.
        """
        posterior, episode, _, actions = _forecast_context(
            assessment, data, episode_id, model, future_actions)
        histories = tuple(
            replay(dict(zip(posterior.prior.names, row)), episode, actions)
            for row, weight in zip(posterior.samples, posterior.weights)
            if weight > 0)
        return cls(assessment, data, episode_id, model, actions, histories)

    def log_likelihood(self, future: History) -> float:
        """Mix complete-history densities using the prefix-fitted weights.

        This is log(sum_i w_i p(future | prefix, joint_row_i)), not an
        average log score or independent per-time mixture. The
        observation model's common reference measure must apply to every
        component.
        """
        posterior, _, prefix, _ = _forecast_context(self.assessment, self.data,
                                                    self.episode_id,
                                                    self.model,
                                                    self.future_actions)
        weights = tuple(w for w in posterior.weights if w > 0)
        terms = tuple(
            math.log(weight) +
            self.model.log_future_likelihood(history, prefix, future)
            for weight, history in zip(weights, self.histories))
        peak = max(terms)
        if peak == -math.inf:
            return -math.inf
        if not math.isfinite(peak):
            raise ConditioningNumericalError("Nonfinite predictive density")
        return peak + math.log(math.fsum(math.exp(t - peak) for t in terms)) \
            - math.log(math.fsum(weights))

    def sample(self, count: int, seed: int) -> Tuple[Tuple[int, History], ...]:
        """Draw entire histories with source indices and reproducible noise.

        One particle is selected per history, never separately per step
        or feature. Sampling adds Monte Carlo error; it does not add new
        fitted particles or change their assessment or predictive
        checks.
        """
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("Positive integer forecast count required")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("Nonnegative integer forecast seed required")
        posterior, _, prefix, _ = _forecast_context(self.assessment, self.data,
                                                    self.episode_id,
                                                    self.model,
                                                    self.future_actions)
        indices = tuple(i for i, w in enumerate(posterior.weights) if w > 0)
        weights = np.array([posterior.weights[i] for i in indices])
        weights /= math.fsum(weights)
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(indices), count, p=weights)
        return tuple((indices[i],
                      self.model.sample_future(self.histories[i], prefix, rng))
                     for i in selected)
