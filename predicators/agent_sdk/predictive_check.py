"""Check an executed request's outcome against the rehearsal that predicted it.

A rehearsal on the joint draws from the live observation predicts where
each object will be once the request has run. Once the request has run
in the environment, the observed outcome is scored against that
prediction feature by feature, in units of the predictive spread: the
draws' own spread together with the observation noise. A model that fits
its recordings can still mispredict a new step (a mechanism the
recordings never exercised); its belief, built from the in-sample fit,
then understates the error, and this check measures the error out of
sample.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import t as student_t

from predicators.observation_noise import ObservationNoise
from predicators.structs import State

# A prediction the check scores no further than this many predictive
# spreads away: an outcome a deterministic prediction misses by any margin
# is already as surprising as an outcome can be.
_MAX_Z = 1e3
# The largest factor a surprise multiplies the belief's temperature by.
MAX_INFLATION = 1e4
# The chance, over all the scored features together, that a calibrated
# prediction is called surprising.
_FALSE_ALARM = 0.01


@dataclasses.dataclass(frozen=True)
class PredictionCheck:
    """The outcome of one request scored against its rehearsal's draws.

    ``worst`` names the feature (object, feature) whose outcome is the
    least likely under its predictive distribution, with its observed
    value, the draws' mean, the predictive spread and ``z_max``, its
    distance from the mean in predictive spreads. ``critical`` is the
    distance that feature's prediction stays within, for a calibrated
    model, over all ``num_features`` features but for ``_FALSE_ALARM``
    of the time. ``needed`` is the smallest factor on the belief's
    temperature under which every feature would lie within two
    predictive spreads (see :func:`_needed_temperature`).
    """
    num_features: int
    num_draws: int
    z_max: float
    worst: Tuple[str, str]
    observed: float
    mean: float
    spread: float
    critical: float
    needed: float

    @property
    def surprising(self) -> bool:
        """Whether the outcome lies outside what the draws allowed."""
        return self.z_max > self.critical

    @property
    def inflation(self) -> float:
        """The factor a surprise multiplies the belief's temperature by
        (``needed``); 1 for an unsurprising outcome."""
        return self.needed if self.surprising else 1.0


def _needed_temperature(delta: float, draw_var: float,
                        noise_var: float) -> float:
    """The smallest factor on the temperature that brings an outcome ``delta``
    from the draws' mean within two predictive spreads.

    Tempering widens the parameter-driven part of the spread, whose
    variance ``draw_var`` grows in proportion to the temperature, and
    leaves the observation noise's ``noise_var`` as it is: the factor
    solves ``delta^2 = 4 (T draw_var + noise_var)``. An outcome the draws
    do not vary toward is out of reach at any temperature and gets the
    largest factor.
    """
    excess = delta**2 / 4.0 - noise_var
    if excess <= draw_var:
        return 1.0
    if draw_var <= 0.0:
        return MAX_INFLATION
    return float(min(excess / draw_var, MAX_INFLATION))


def check_prediction(observed: State, predicted: Sequence[State],
                     features: Dict[str, List[str]],
                     noise: ObservationNoise) -> Optional[PredictionCheck]:
    """Score ``observed`` against the predicted states ``predicted`` (one per
    draw) on the scored ``features`` (type name -> feature names).

    The robot's own features are left out: its commands drive them. An
    angle is compared on the circle. None when no feature is predicted
    by at least two draws.
    """
    by_name = [{o.name: o for o in state} for state in predicted]
    rows: List[Tuple[float, float, str, str, float, float, float, float]] = []
    needed = 1.0
    for obj in observed:
        type_name = obj.type.name
        if type_name == "robot":
            continue
        angular = set(getattr(obj.type, "angular_features", ()))
        for feat in features.get(type_name, ()):
            if feat not in obj.type.feature_names:
                continue
            values = []
            for state, objects in zip(predicted, by_name):
                twin = objects.get(obj.name)
                if twin is not None and feat in twin.type.feature_names:
                    values.append(float(state.get(twin, feat)))
            if len(values) < 2:
                continue
            seen = float(observed.get(obj, feat))
            draws = np.asarray(values)
            if feat in angular:
                mean = float(
                    np.arctan2(np.mean(np.sin(draws)), np.mean(np.cos(draws))))
                offsets = (draws - mean + np.pi) % (2.0 * np.pi) - np.pi
                delta = (seen - mean + np.pi) % (2.0 * np.pi) - np.pi
                var = float(np.sum(offsets**2) / (draws.size - 1))
            else:
                mean = float(np.mean(draws))
                delta = seen - mean
                var = float(np.var(draws, ddof=1))
            sigma = noise.feature_sigma(obj.type, feat)
            # The draws' variance is estimated from them, with n - 1
            # degrees of freedom, and a new draw spreads by (1 + 1/n) of
            # it; the observation noise's is known. The predictive
            # distribution is then Student's t with the Welch-Satterthwaite
            # degrees of freedom.
            draw_part = var * (1.0 + 1.0 / draws.size)
            total = draw_part + sigma**2 + 1e-12
            dof = (total**2 * (draws.size - 1) /
                   draw_part**2 if draw_part > 0.0 else np.inf)
            spread = float(np.sqrt(total))
            z = float(min(abs(delta) / spread, _MAX_Z))
            tail = float(2.0 * student_t.sf(z, dof))
            rows.append((tail, -z, obj.name, feat, seen, mean, spread, dof))
            needed = max(needed, _needed_temperature(delta, draw_part,
                                                     sigma**2))
    if not rows:
        return None
    _tail, neg_z, name, feat, seen, mean, spread, dof = min(rows)
    z = -neg_z
    critical = float(student_t.isf(_FALSE_ALARM / (2.0 * len(rows)), dof))
    return PredictionCheck(num_features=len(rows),
                           num_draws=len(predicted),
                           z_max=z,
                           worst=(name, feat),
                           observed=seen,
                           mean=mean,
                           spread=spread,
                           critical=critical,
                           needed=needed)
