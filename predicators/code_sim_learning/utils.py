"""Utilities for the code sim-learning module.

Core primitives for process-dynamics simulation:

* ``apply_rules`` — run a list of rule functions on a state, return
  feature updates (``ProcessUpdate``).
* ``merge_updates`` — overwrite features in a ``State`` with values
  from a ``ProcessUpdate``.
* ``read_simulator_components`` — pull the ``PROCESS_RULES``,
  ``PARAM_SPECS``, ``PROCESS_FEATURES`` triple out of a namespace
  (oracle module globals or agent-synthesized exec namespace).
* ``sigmoid`` / ``SOFT_EPS`` — building blocks for differentiable
  soft gates in process rules.
"""

from __future__ import annotations

import inspect
import logging
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, \
    Optional, Sequence, Tuple

import numpy as np

from predicators.structs import Action, Object, State

logger = logging.getLogger(__name__)

# ── Type aliases ──────────────────────────────────────────────────

# {Object: {feature_name: new_value}} — the dict that rule functions
# accumulate into.
ProcessUpdate = Dict[Object, Dict[str, float]]

# {param_name: value} — the params dict passed to rule functions.
Params = Dict[str, float]

# ── Soft-gate building blocks ─────────────────────────────────────

# Default smoothing scale for parameter-dependent soft gates. Small
# enough that gates are ~99% saturated when the operand is one
# threshold-width into the active region, large enough to give MCMC a
# usable gradient near the cliff. 0.02 is in the right ballpark for
# both spatial thresholds (~0.05–0.15 m) and water-level thresholds
# (~0.3–1.3). Override per call site as needed.
SOFT_EPS = 0.02


def sigmoid(z: float) -> float:
    """Numerically-stable scalar sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    ez = np.exp(z)
    return ez / (1.0 + ez)


def objs_by_type(state: State) -> Dict[str, List[Object]]:
    """Group state objects by type name."""
    groups: Dict[str, List[Object]] = {}
    for o in state:
        groups.setdefault(o.type.name, []).append(o)
    return groups


# ── Primitives ────────────────────────────────────────────────────


def apply_rules(state: State, rules: List,
                params: Dict[str, float]) -> ProcessUpdate:
    """Apply process rules sequentially and return feature updates.

    Each rule has signature ``rule(state, updates, params) -> updates``.
    Values are normalised to plain floats (rules may return numpy
    scalars).
    """
    updates: ProcessUpdate = {}
    for rule in rules:
        updates = rule(state, updates, params)
    return {
        obj: {feat: float(val)
              for feat, val in feat_dict.items()}
        for obj, feat_dict in updates.items()
    }


# ── Recurrent rule support (latent + history) ─────────────────────

# Read-only history prefix handed to recurrent rules:
# [(state_0, action_0), (state_1, action_1), ..., (state_t, action_t)]
# Most recent last. The first entry's action is ``None``. Typed as
# ``Sequence`` (covariant) so callers can pass a stricter
# ``List[Tuple[State, Action]]`` without an invariance complaint —
# rules treat history as read-only.
History = Sequence[Tuple[State, Optional[Action]]]


@lru_cache(maxsize=None)
def _rule_accepts_latent(rule: Callable) -> bool:
    """Return True iff ``rule`` declares a `latent` parameter or **kwargs.

    Used by :func:`apply_rules_with_latent` to thread the sample's
    `latent` state-feature block / `history` only into rules that opted
    in. Cached because rule callables are reused across many simulator
    invocations and ``inspect.signature`` isn't free.
    """
    try:
        params = inspect.signature(rule).parameters
    except (TypeError, ValueError):
        return False
    if "latent" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD
               for p in params.values())


def has_latent_rules(rules: Iterable[Callable]) -> bool:
    """True iff any rule declares a `latent` param (recurrent 5-arg).

    The dispatch signal that distinguishes a partially-observable
    simulator (carries a latent block) from a fully-observable one: it
    keys off the rule *signatures*, so it is correct on both the oracle
    path (where ``LATENT_INIT`` may not have been loaded) and the agent-
    synthesis path. Empty / all-legacy rule lists return False, so
    fully-observable approaches take their existing non-latent paths
    unchanged.
    """
    return any(_rule_accepts_latent(r) for r in rules)


def apply_rules_with_latent(
    state: State,
    latent: Dict[str, Any],
    history: History,
    rules: List,
    params: Dict[str, float],
) -> ProcessUpdate:
    """Apply rules with a ``latent`` state-feature block and read-only
    ``history``.

    Each rule is either:

    * **Legacy 3-arg**: ``rule(state, updates, params) -> updates``.
      Called without latent/history; latent and history are ignored.
    * **Recurrent 5-arg**: ``rule(state, latent, history, updates,
      params) -> updates``. ``latent`` is mutated in place — the
      same dict object passed in by the caller is threaded across
      steps.

    Signature is inspected once per rule (cached). Values are
    normalised to plain floats. The returned update dict has the
    same shape as ``apply_rules``'s output.
    """
    updates: ProcessUpdate = {}
    for rule in rules:
        if _rule_accepts_latent(rule):
            updates = rule(state, latent, history, updates, params)
        else:
            updates = rule(state, updates, params)
    return {
        obj: {feat: float(val)
              for feat, val in feat_dict.items()}
        for obj, feat_dict in updates.items()
    }


def init_latent(
    latent_init: Optional[Dict[str, Any]],
    params: Dict[str, float],
) -> Dict[str, Any]:
    """Build the initial latent state-feature block for a fresh rollout.

    ``latent_init`` follows the same convention as ``PARAM_SPECS``: it
    may be ``None`` (empty block), a plain ``Dict[str, Any]``, or a
    zero-arg callable returning such a dict. Values may be
    :class:`~predicators.code_sim_learning.training.ParamSpec`
    instances, in which case the corresponding entry from
    ``params[name]`` is used (falling back to ``init_value`` if the
    param hasn't been fit yet) — this lets MCMC fit the initial
    latent value alongside rate parameters.
    """
    if latent_init is None:
        return {}
    if callable(latent_init):
        latent_init = latent_init()
    if not isinstance(latent_init, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in latent_init.items():
        # Late import to avoid a circular dependency.
        # pylint: disable=import-outside-toplevel
        from predicators.code_sim_learning.training import ParamSpec
        if isinstance(v, ParamSpec):
            out[k] = params.get(v.name, v.init_value)
        else:
            out[k] = v
    return out


def merge_updates(
    base_state: State,
    updates: ProcessUpdate,
) -> State:
    """Overwrite features in *base_state* with values from *updates*."""
    if not updates:
        return base_state

    new_data = {}
    for obj in base_state:
        arr = base_state[obj].copy()
        if obj in updates:
            for feat_name, new_val in updates[obj].items():
                idx = obj.type.feature_names.index(feat_name)
                arr[idx] = new_val
        new_data[obj] = arr

    merged = base_state.copy()
    merged.data = new_data
    return merged


def iter_feature_residuals(
    triples: Iterable[Tuple[State, State]],
    feature_scope: Optional[Dict[str, List[str]]] = None,
) -> Iterator[Tuple[int, Object, str, str, float, float]]:
    """Yield ``(step_idx, obj, type_name, feat, pred_val, obs_val)``.

    Walks each ``(s_pred, s_obs)`` pair and emits one tuple per
    ``(object, feature)``. If ``feature_scope`` is provided, only
    features listed under each type name are emitted; otherwise every
    feature in the type's ``feature_names`` is emitted. Used by both the
    residual-based feature-discovery scan and the per-feature residual
    report so the two stay in sync.
    """
    for i, (s_pred, s_obs) in enumerate(triples):
        for obj in s_pred:
            tn = obj.type.name
            feats: Sequence[str] = (feature_scope.get(tn, []) if feature_scope
                                    is not None else obj.type.feature_names)
            for feat in feats:
                yield (
                    i,
                    obj,
                    tn,
                    feat,
                    float(s_pred.get(obj, feat)),
                    float(s_obs.get(obj, feat)),
                )


# ── Module-namespace loader ───────────────────────────────────────


def read_simulator_components(
    ns: Mapping[str, Any],
) -> Tuple[Optional[List], Optional[List], Optional[Dict[str, List[str]]]]:
    """Pull the simulator triple from a namespace (module or exec dict).

    Looks for three names by convention:

    * ``PROCESS_RULES`` — non-empty list of rule functions.
    * ``PARAM_SPECS``   — list of ``ParamSpec``, **or** a zero-arg
      callable returning such a list. The callable form lets oracle
      modules defer CFG-dependent values until consumption time, so the
      module can be imported before CFG is finalized; the agent's
      saved-file form normally just uses a list.
    * ``PROCESS_FEATURES`` — ``{type_name: [feature_names]}`` dict.

    Returns ``(rules, specs, features)`` with ``None`` for any
    missing-or-malformed component; callers decide how to react.

    The optional fourth component ``LATENT_INIT`` (used by the
    recurrent partial-observability approach) is read separately via
    :func:`read_latent_init` so existing callers don't have to grow
    a fourth tuple element.
    """
    rules = ns.get("PROCESS_RULES")
    if not isinstance(rules, list) or not rules:
        rules = None

    specs = ns.get("PARAM_SPECS")
    if callable(specs):
        specs = specs()
    if not isinstance(specs, list) or not specs:
        specs = None

    features = ns.get("PROCESS_FEATURES")
    if features is not None and not isinstance(features, dict):
        features = None

    return rules, specs, features


def read_latent_init(ns: Mapping[str, Any]) -> Optional[Any]:
    """Pull ``LATENT_INIT`` (optional) from a simulator namespace.

    ``LATENT_INIT`` declares the initial values for the latent
    state-feature block used by the partial-observability approach.
    Returns ``None`` if not present or malformed; in that case the
    caller should default to an empty block.

    Accepted shapes:

    * ``Dict[str, Any]`` — literal initial values.
    * ``Callable[[], Dict[str, Any]]`` — zero-arg factory, called at
      consumption time. Mirrors the callable-``PARAM_SPECS`` pattern.
    """
    latent_init = ns.get("LATENT_INIT")
    if latent_init is None:
        return None
    if not (callable(latent_init) or isinstance(latent_init, dict)):
        return None
    return latent_init


# ── LearnedSimulator ──────────────────────────────────────────────


class LearnedSimulator:
    """Wraps a step-level simulator function (handwritten or LLM-synthesized).

    The function predicts process dynamics — features like water_volume,
    heat_level, spilled_level that aren't captured by rigid body
    physics.
    """

    StepFn = Callable[[State], ProcessUpdate]

    def __init__(self,
                 step_fn: StepFn,
                 name: str = "learned_simulator") -> None:
        self._step_fn = step_fn
        self.name = name

    def predict_step(self, state: State) -> ProcessUpdate:
        """Predict process feature updates for a single timestep."""
        try:
            return self._step_fn(state)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Simulator '%s' step raised: %s", self.name, e)
            return {}
