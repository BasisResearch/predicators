"""The observation-noise channel of the continual protocol.

Design: ``docs/continual-uncertainty.md``, section 3.1. Every arm sees
the env state through this channel: object positions and orientations
carry additive zero-mean Gaussian noise, discrete features and the
robot's own state (proprioception) stay exact, and everything the
harness judges (the evaluators, the level index) keeps the true state.
One draw per env step, keyed by run seed, level, episode and step, so a
run is reproducible and a resumed run re-observes the same frames.

The same object tells the fit how much of a residual the observation
noise explains (:meth:`ObservationNoise.residual_scale`), which is what
makes the channel principled rather than a nuisance: the likelihood the
fit maximises carries the noise it was declared with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from predicators.run.recording import sanitize_state
from predicators.settings import CFG
from predicators.structs import State, Type

# Feature classes. Positions are metres; orientations are radians, and a
# Type's own ``angular_features`` extend the orientation class.
POSITION_FEATURES = frozenset({"x", "y", "z"})
ORIENTATION_FEATURES = frozenset(
    {"rot", "roll", "pitch", "yaw", "tilt", "wrist"})
# Types observed exactly: the robot reads its own joints.
EXACT_TYPES = frozenset({"robot"})


@dataclass(frozen=True)
class ObservationNoise:
    """Per-feature-class sigmas of the channel; zeros disable it."""
    position: float = 0.0
    orientation: float = 0.0
    # Whether the agent's contract states the sigmas (the base case) or
    # the agent has to find the noise itself (the harder ablation).
    declared: bool = True

    @classmethod
    def from_cfg(cls) -> "ObservationNoise":
        """The channel the current flags describe."""
        return cls(position=float(CFG.continual_obs_noise_position),
                   orientation=float(CFG.continual_obs_noise_orientation),
                   declared=bool(CFG.continual_obs_noise_declared))

    @property
    def enabled(self) -> bool:
        """Whether any feature class carries noise."""
        return self.position > 0.0 or self.orientation > 0.0

    def feature_sigma(self, obj_type: Type, feat: str) -> float:
        """The sigma of ``feat`` on an object of ``obj_type``; 0 if exact."""
        if obj_type.name in EXACT_TYPES:
            return 0.0
        if feat in POSITION_FEATURES:
            return self.position
        if feat in ORIENTATION_FEATURES or feat in getattr(
                obj_type, "angular_features", ()):
            return self.orientation
        return 0.0

    def residual_scale(self, obj_type: Type, feat: str, motion_scale: float,
                       noise_sigma: float) -> float:
        """The residual scale that folds this feature's observation noise into
        the fit's likelihood.

        The rollout objective scores ``(pred - obs) / scale`` under a
        Gaussian of width ``noise_sigma``, so it assumes a residual std
        of ``noise_sigma * motion_scale`` (the model-error floor, a
        fraction of typical motion). Observation noise adds its own
        variance: ``std^2 = (noise_sigma * motion_scale)^2 + sigma_f^2``,
        which the objective reproduces with ``scale = sqrt(motion_scale^2
        + (sigma_f / noise_sigma)^2)`` and ``noise_sigma`` unchanged.
        Every RMS threshold keeps its meaning (a multiple of the total
        noise) and a pure-noise residual never reads as model error.
        """
        sigma = self.feature_sigma(obj_type, feat)
        if sigma <= 0.0:
            return motion_scale
        assert noise_sigma > 0.0, noise_sigma
        return float(np.sqrt(motion_scale**2 + (sigma / noise_sigma)**2))

    def summary(self) -> str:
        """The frame's one-line reminder."""
        parts = []
        if self.position > 0.0:
            parts.append(f"position sigma {self.position:g} m")
        if self.orientation > 0.0:
            parts.append(f"orientation sigma {self.orientation:g} rad")
        return (", ".join(parts) +
                " on object features (robot exact; one draw per step)")

    def describe(self) -> str:
        """One line for prompts and reports."""
        if not self.enabled:
            return "observations are exact"
        parts = []
        if self.position > 0.0:
            parts.append(f"positions (x, y, z) sigma {self.position:g} m")
        if self.orientation > 0.0:
            parts.append(f"orientations (rot, roll, pitch, yaw) sigma "
                         f"{self.orientation:g} rad")
        return ("Gaussian observation noise on every non-robot object: " +
                "; ".join(parts) + "; discrete features and the robot's own "
                "state are exact; one draw per env step, so re-reading an "
                "observation without stepping returns the same values")

    def perturb(self, state: State, rng: np.random.Generator) -> State:
        """A copy of ``state`` as the agent observes it.

        Noisy features get one fresh draw each; the rest of ``data`` and
        the ``latent`` block are copied unchanged. The copy is the
        recording's sanitized form: a PyBullet state keeps its class and
        the robot's own joint data (proprioception is exact, and the
        base simulator reads the fingers from it on every step), while
        the env-only channels (``privileged``, engine handles) are
        dropped, so neither the belief simulator nor the data the agent
        reads can carry the true state under the view.
        """
        view = sanitize_state(state)
        for obj, arr in view.data.items():
            if obj.type.name in EXACT_TYPES:
                continue
            for i, feat in enumerate(obj.type.feature_names):
                sigma = self.feature_sigma(obj.type, feat)
                if sigma > 0.0:
                    arr[i] += rng.normal(0.0, sigma)
        if state.latent is not None:
            view.latent = dict(state.latent)
        return view


def step_rng(seed: int,
             level: int,
             episode: int,
             step: int,
             extra: Iterable[int] = ()) -> np.random.Generator:
    """The generator of one observation's draw.

    Keyed by the run seed and the observation's coordinates so the same
    step of the same run always observes the same frame, whether it is
    replayed, resumed or rendered offline.
    """
    key = [int(seed), int(level), int(episode), int(step), *map(int, extra)]
    return np.random.default_rng(np.random.SeedSequence(key))


def noise_or_none(
        noise: Optional[ObservationNoise]) -> Optional[ObservationNoise]:
    """``noise`` when it carries any sigma, else None (exact observations)."""
    if noise is None or not noise.enabled:
        return None
    return noise
