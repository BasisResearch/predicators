"""Ground-truth simulator program for pybullet_domino process dynamics.

This is an intentionally *empty* (no-op) simulator: it carries no
process dynamics and predicts no state features. It exists so that
``get_gt_simulator("pybullet_domino")`` resolves to a valid module
instead of raising ``NotImplementedError``.

The contract enforced by ``read_simulator_components`` requires a
non-empty ``PROCESS_RULES`` list and a non-empty ``PARAM_SPECS`` list,
so we provide a single identity rule (returns updates unchanged) and a
single placeholder parameter. ``PROCESS_FEATURES`` is empty, signalling
that no features are predicted by the GT process model.
"""

from __future__ import annotations

from typing import Dict, List

from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.utils import Params, ProcessUpdate
from predicators.ground_truth_models import GroundTruthSimulatorFactory
from predicators.structs import State

# ── Process rules ────────────────────────────────────────────────


def _identity(state: State, updates: ProcessUpdate,
              params: Params) -> ProcessUpdate:
    """No-op rule: domino dynamics are not modelled, so pass through."""
    del state, params  # unused
    return updates


# ── Public API: consumed by read_simulator_components ────────────

PROCESS_RULES = [_identity]

# A single placeholder spec keeps PARAM_SPECS non-empty (the loader
# rejects an empty list) while leaving the dynamics a true no-op.
PARAM_SPECS: List[ParamSpec] = [ParamSpec("placeholder", 0.0, lo=0.0)]

PROCESS_FEATURES: Dict[str, List[str]] = {}

# ── Factory binding ──────────────────────────────────────────────


class PyBulletDominoGroundTruthSimulatorFactory(GroundTruthSimulatorFactory):
    """Empty GT process-dynamics simulator for pybullet_domino.

    Only pins the env-name binding so ``get_gt_simulator`` can locate
    this module via the factory registry; the simulator components live
    as module globals above.
    """

    @classmethod
    def get_env_names(cls) -> set:
        return {"pybullet_domino", "pybullet_domino_real"}
