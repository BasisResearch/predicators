"""Execution-time tracking of the learned simulator's latent block.

Under partial observability the agent's simulator carries hidden state
in a ``latent`` dict (bond flags, cure counters, fill levels) that its
recurrent rules thread from step to step. Planning and refinement roll
that latent forward inside the belief, so a plan is validated against
"latent evolves by these rules given these states". Real observations
carry no latent at all, so at execution every latent-reading predicate
used to be unconditionally False: Wait targets on such atoms never
fired and the monitor could not evaluate their annotations.

:class:`LatentTracker` closes that gap with prediction only: for each
new real observation it runs the same rules the belief used, on the
observed post-action state with the single-step history window
``combined_simulate`` uses, and attaches a snapshot of the resulting
latent to the state handed to the policy, the termination function,
and the execution monitor. The latent is state-conditioned (a cure
counter advances only while the observed faces are in contact) but not
observation-corrected: a wrong rule stays wrong until the next learn,
exactly as during validation. Likelihood reweighting over the
parameter ensemble (a Rao-Blackwellised particle filter) is the
natural extension and would live here.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from predicators.code_sim_learning.utils import apply_rules_with_latent, \
    has_latent_rules, init_latent, observation_view
from predicators.structs import Action, State

logger = logging.getLogger(__name__)


class LatentTracker:
    """Prediction-only belief over a recurrent simulator's latent block.

    ``params`` is held by reference (not copied) so a fit that updates
    the approach's parameter dict in place is picked up, matching the
    belief simulator's closure. One tracker serves one episode: call
    :meth:`reset` at episode start (``attach`` does so lazily) and
    :meth:`attach` once per observation, in order.
    """

    def __init__(self,
                 rules: Iterable[Any],
                 params: Dict[str, float],
                 latent_init: Any,
                 log_label: str = "latent tracker") -> None:
        self._rules: List[Any] = list(rules)
        self._params = params
        self._latent_init = latent_init
        self._log_label = log_label
        self._latent: Optional[Dict[str, Any]] = None
        self._failed = False
        self._num_observations = 0

    @property
    def num_rules(self) -> int:
        """Number of rules threaded per observation."""
        return len(self._rules)

    @property
    def failed(self) -> bool:
        """True once a rule raised; later observations pass through untracked
        (``latent`` stays ``None``) for the rest of the episode."""
        return self._failed

    @property
    def latent(self) -> Optional[Dict[str, Any]]:
        """A snapshot of the current tracked latent (``None`` before
        :meth:`reset`)."""
        return copy.deepcopy(self._latent)

    def reset(self) -> None:
        """Start an episode from the canonical initial latent, the same
        ``init_latent`` the belief rollout starts from."""
        self._latent = init_latent(self._latent_init, self._params)
        self._failed = False
        self._num_observations = 0

    def attach(self, state: State, prev_action: Optional[Action]) -> State:
        """Advance the latent by ``prev_action``'s observed outcome and return
        a copy of ``state`` carrying a snapshot of it.

        ``prev_action`` is the action whose execution produced
        ``state`` (``None`` for the episode's first observation, which
        carries the initial latent untouched - the belief's task init
        does the same). The rules see ``observation_view(state)`` with
        the one-entry history ``[(obs, prev_action)]``, exactly as
        ``combined_simulate`` feeds them the base sim's post-action
        prediction. The returned state is a shallow copy (same class,
        shared feature arrays) so the raw observation kept for learning
        never carries a belief; its latent is a deep copy so predicate
        classifiers cannot alias the tracker's running dict. After a
        rule raises, the raw state is returned unchanged.
        """
        if self._latent is None:
            self.reset()
        assert self._latent is not None
        if self._failed:
            return state
        if prev_action is not None:
            obs = observation_view(state)
            history: List[Tuple[State,
                                Optional[Action]]] = [(obs, prev_action)]
            try:
                apply_rules_with_latent(obs, self._latent, history,
                                        self._rules, self._params)
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(
                    "%s: a rule raised at observation %d (%s: %s); latent "
                    "tracking is off for the rest of this episode, so "
                    "latent-reading predicates read as on a bare "
                    "observation.", self._log_label,
                    self._num_observations + 1,
                    type(e).__name__, e)
                self._failed = True
                return state
        self._num_observations += 1
        tracked = copy.copy(state)
        tracked.latent = copy.deepcopy(self._latent)
        return tracked


def make_latent_tracker(rules: Optional[Iterable[Any]], params: Dict[str,
                                                                     float],
                        latent_init: Any) -> Optional[LatentTracker]:
    """A tracker for ``rules`` when any of them is recurrent, else None.

    Legacy 3-arg (fully observable) simulators carry no latent, so there
    is nothing to track and every execution path stays as it was.
    """
    rule_list = list(rules or [])
    if not rule_list or not has_latent_rules(rule_list):
        return None
    return LatentTracker(rule_list, params, latent_init)
