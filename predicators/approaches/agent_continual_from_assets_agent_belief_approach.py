"""EMPIRIC from assets with the agent's own parameter belief.

The agent builds its scene from the engine, the manifest and the assets,
as in EMPIRIC from assets, and the harness fits nothing, as in the no-
fitting arm. The agent estimates its parameters and their uncertainty by
any method it chooses (its own fits against the recordings, formulas,
judgment) and exports the uncertainty as ``PARAM_DRAWS``, a list of
parameter draws, in ``simulator.py``. The joint draws, the rehearsals,
execution monitoring and the predictive check run on those draws, each
paired with a draw of the state belief and the memory it implies; the
harness never widens or narrows them. Each declaration's ``init_value``
is the point estimate the deployed model runs at.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from predicators.approaches.agent_continual_ablation_approach import \
    AgentContinualNoFittingApproach
from predicators.approaches.agent_continual_from_assets_approach import \
    AgentContinualFromAssetsApproach
from predicators.code_sim_learning.parameter_belief import ParameterBelief, \
    agent_draws_belief, agent_param_draws, stable_seed
from predicators.settings import CFG
from predicators.structs import LowLevelTrajectory


class AgentContinualFromAssetsAgentBeliefApproach(
        AgentContinualFromAssetsApproach, AgentContinualNoFittingApproach):
    """Agent-built scenes, agent-estimated parameters and draws."""

    _save_suffix = "AgentContinualFromAssetsAgentBelief"
    # The agent's draws are the whole parameter belief: the harness keeps
    # its declared ranges as written and samples no undeclared material.
    _belief_over_guesses = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not CFG.continual_uncertainty_decisions or \
                int(CFG.belief_joint_draws) <= 0:
            raise ValueError(
                "The agent-belief arm plans over the agent's parameter "
                "draws; run it with continual_uncertainty_decisions and "
                "belief_joint_draws > 0")
        # The PARAM_DRAWS each loaded simulator.py exported, by the
        # sha256 of its content, as the load returned them (a callable is
        # called there, with the recordings in scope).
        self._exported_draws: Dict[str, Any] = {}
        super().__init__(*args, **kwargs)
        # The harness never fits here, so the agent's probe has no fit.
        self._tool_context.probe_absent = frozenset({"fit"})

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_from_assets_agent_belief"

    def _load_simulator_from_module_file(
        self,
        path: str,
        trajectories: Optional[List[LowLevelTrajectory]] = None,
    ) -> Any:
        """Load the model as EMPIRIC from assets does, and keep the
        ``PARAM_DRAWS`` it exports."""
        result = super()._load_simulator_from_module_file(path, trajectories)
        namespace = result[3]
        if namespace is not None:
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            value = namespace.get("PARAM_DRAWS")
            if callable(value):
                try:
                    value = value()
                except Exception as e:  # pylint: disable=broad-except
                    value = _Raised(f"{type(e).__name__}: {e}")
            if len(self._exported_draws) > 16:
                self._exported_draws.clear()
            self._exported_draws[digest] = value
        return result

    def _agent_draws(self) -> Tuple[List[Dict[str, float]], str]:
        """The current ``simulator.py``'s draws over its declared parameters,
        and why there are none ("" when there are)."""
        digest = self._current_simulator_digest()
        if digest is None or digest not in self._exported_draws:
            return [], "the current simulator.py has not loaded"
        value = self._exported_draws[digest]
        if value is None:
            return [], "simulator.py exports no PARAM_DRAWS"
        if isinstance(value, _Raised):
            return [], f"PARAM_DRAWS() raised {value.message}"
        return agent_param_draws(value, self._belief_specs())

    def _belief_specs(self) -> List[Any]:
        return list(self._physical_param_specs) + list(
            getattr(self, "_probe_rule_specs", None) or [])

    def _sourced_parameter_belief(
            self) -> Tuple[Optional[ParameterBelief], Optional[str]]:
        """The agent's draws as the parameter belief, or one draw at the
        declared values while it supplies none; never a fitted belief, so
        nothing tempers it."""
        if int(CFG.belief_joint_draws) <= 0:
            return None, None
        specs = self._belief_specs()
        rows, _ = self._agent_draws()
        if not rows:
            rows = [{spec.name: float(spec.init_value) for spec in specs}]
        digest = self._current_simulator_digest()
        key = ("agent", digest, tuple(spec.name for spec in specs),
               tuple(tuple(sorted(row.items())) for row in rows))
        cache = self._belief_cache()
        if key not in cache:
            if len(cache) > 8:
                cache.clear()
            cache[key] = agent_draws_belief(
                specs, rows, int(CFG.belief_joint_draws),
                stable_seed(CFG.seed, "agent draws", digest))
        return cache[key], None

    def _fit_status_text(self) -> str:
        rows, why = self._agent_draws()
        if rows:
            names = sorted({
                name
                for row in rows for name, value in row.items()
                if value != rows[0][name]
            })
            varied = ", ".join(names) if names else "no parameter"
            return (f"your declared values as the point estimate, and "
                    f"{len(rows)} PARAM_DRAWS as the parameter belief "
                    f"(varying {varied}); no harness fit")
        return (f"your declared values only: {why}, so every rehearsal "
                "draw runs your declared values; no harness fit")

    def _no_model_section(self) -> str:
        return "no_model_agent_belief"

    def _play_prompt_options(self) -> Dict[str, Any]:
        return {"scene_built": True, "fit_gate": False, "agent_belief": True}

    def _play_model_contract(self, **options: Any) -> str:
        return super()._play_model_contract(agent_belief=True, **options)


class _Raised:
    """A ``PARAM_DRAWS`` callable that raised when the model loaded."""

    def __init__(self, message: str) -> None:
        self.message = message
