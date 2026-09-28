"""Synthesis-time validation hooks for the agent sim-learning approach.

These helpers run inside an active synthesis-agent session: they need
approach state (base env, train tasks, predicates, options) but never
re-enter the agent — no sketch-prompt query, no new session — so they
can be invoked from a synthesis tool without disturbing the live
session's prompt or tool set. They live in the approaches layer (not
``code_sim_learning``) because they orchestrate approach state and the
planner; the ``SynthesisBackend`` protocol declares exactly the approach
surface they touch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.utils import LearnedSimulator, \
    apply_rules, has_latent_rules

if TYPE_CHECKING:
    from predicators.agent_sdk.synthesis_backend import SynthesisBackend

logger = logging.getLogger(__name__)


def build_candidate_option_model(
    approach: "SynthesisBackend",
    rules: List,
    specs: List[ParamSpec],
    latent_init: Any = None,
) -> Tuple[Any, Dict[str, float]]:
    """Build the candidate's option model at :func:`carry_over_params`.

    The parameters are the last published fit's where a spec still
    exists and the value lies in its box, the declared init value
    otherwise: fitting is the agent's explicit ``sim.fit`` call, never a
    side effect of probing (see
    ``AgentSimLearningApproach._make_candidate_probe_model_provider``).

    The front half of the synthesis-session probe: every rollout must
    exercise the candidate simulator at its *deployed* (fitted)
    parameters, never at init_value. Returns ``(option_model,
    params)``.

    Publishes side effects onto ``approach`` exactly once, here, so the
    two surfaces can never disagree: the candidate ``rules`` /
    ``latent_init`` (the recurrent combined simulator is built from
    instance state) and the fitted params into ``_fitted_params`` *in
    place* (invented predicates hold a ``_ParamsView`` over it - the
    gating rule and the gating predicate must anchor to the same
    values).
    """
    # pylint: disable=protected-access
    latent = has_latent_rules(rules)

    # Publish the candidate rules / latent_init *before* building the
    # combined simulator: the recurrent combined sim reads
    # self._residual_rules / self._latent_init / self._fitted_params, so
    # without this it would validate a stale cycle's rules - or, with
    # _residual_rules still None, mis-dispatch a latent candidate onto
    # the 3-arg path. Per-cycle state; overwritten when synthesis
    # finalises.
    approach._residual_rules = rules
    if latent:
        approach._latent_init = latent_init

    params = carry_over_params(approach._fitted_params, specs)
    approach._fitted_params.clear()
    approach._fitted_params.update(params)
    return _finish_candidate_model(approach, rules, params), params


def carry_over_params(fitted: Dict[str, float],
                      specs: List[ParamSpec]) -> Dict[str, float]:
    """Parameter values for an UNFITTED candidate: the last fit's value where
    the spec still exists and the value lies inside its box, the declared
    ``init_value`` otherwise."""
    out: Dict[str, float] = {}
    for spec in specs:
        val = fitted.get(spec.name)
        lo = spec.lo if spec.lo is not None else -float("inf")
        hi = spec.hi if spec.hi is not None else float("inf")
        if val is not None and lo <= val <= hi:
            out[spec.name] = float(val)
        else:
            out[spec.name] = float(spec.init_value)
    return out


def _finish_candidate_model(approach: "SynthesisBackend", rules: List,
                            params: Dict[str, float]) -> Any:
    """Build the combined simulator + option model over published rules."""
    # pylint: disable=protected-access

    # Fully-observable rules run through this `learned` object; for
    # recurrent rules _build_combined_simulator bypasses it and threads
    # state.latent through the candidate rules published above.
    learned = LearnedSimulator(
        step_fn=lambda s, c, _r=rules, _p=params:  # type: ignore[misc]
        apply_rules(s, _r, _p, cmds=c),
        name="agent_in_session")
    combined_sim = approach._build_combined_simulator(learned)
    return approach._build_option_model(combined_sim)
