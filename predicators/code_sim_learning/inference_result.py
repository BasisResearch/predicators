"""Versioned inference summaries, initially adapting the incumbent fitter.

This interface does not change an estimator or publish parameters. In
particular, legacy landscape widths are not marginal credible intervals,
and optimizer candidates are not weighted joint posterior samples.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal

if TYPE_CHECKING:
    from predicators.code_sim_learning.orchestrator import SysIdOutcome


@dataclass(frozen=True)
class LegacyInferenceResult:
    """An owned summary of legacy inference after parameter selection.

    Nested dictionaries are independent of the fit and its caches. The
    caller still decides whether this candidate is diagnostic or
    published; selected parameters alone do not certify publication or
    model validity. The original FitResult remains available on
    SysIdOutcome for incumbent numerical consumers and checkpoint
    compatibility.
    """

    point_estimate: Dict[str, float]
    selected_parameters: Dict[str, float]
    parameter_diagnostics: Dict[str, Dict[str, Any]]
    num_segments: int
    num_survivors: int
    schema_version: Literal[1] = 1
    estimator: Literal["legacy_rollout_sysid"] = "legacy_rollout_sysid"
    uncertainty_kind: Literal["legacy_widths"] = "legacy_widths"

    @classmethod
    def from_outcome(cls, outcome: SysIdOutcome) -> LegacyInferenceResult:
        """Adapt without fitting, sampling, changing verdicts, or publishing.

        No-survivor outcomes retain their pinned point summary and empty
        selection. Consumers must preserve the existing refusal
        handling.
        """
        return cls(point_estimate=dict(outcome.fitted),
                   selected_parameters=dict(outcome.applied),
                   parameter_diagnostics=copy.deepcopy(outcome.report),
                   num_segments=outcome.num_segments,
                   num_survivors=outcome.num_survivors)
