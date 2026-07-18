"""Configuration snapshot for the rollout system-identification stack.

:class:`SysIdConfig` freezes the ``code_sim_learning_*`` flags that the
sysID modules (:mod:`trajectory_prep`, :mod:`rollout_objective`,
:mod:`grid_seed`, :mod:`physical_sysid`) consume, so deep helpers take
plain values or a config object instead of reading the global ``CFG``.
Public entry points resolve ``config = config or SysIdConfig.from_cfg()``
at call time - never at import time - because tests reconfigure the
global settings via ``utils.reset_config`` between calls.

``warm_start_with_lm``, ``num_mcmc_steps`` and
``log_hessian_identifiability`` are carried here for completeness of the
sysID knob surface, but :mod:`fitting` and :mod:`lm` keep their direct
``CFG`` reads for them (their use is small and shared with the
non-sysID, per-transition fitting paths).
"""

from __future__ import annotations

from dataclasses import dataclass

from predicators.settings import CFG


@dataclass(frozen=True)
class SysIdConfig:
    """Frozen view of the ``code_sim_learning_*`` flags used by sysID.

    Each field mirrors one CFG flag (see :meth:`from_cfg` for the
    mapping); the flags' semantics are documented in
    ``predicators/settings.py`` next to their defaults.
    """

    warm_start_with_lm: bool
    num_mcmc_steps: int
    rollout_num_mcmc_steps: int
    grid_seed_points: int
    grid_sweep_passes: int
    grid_refine_evals: int
    grid_flat_frac: float
    trim_rms_factor: float
    settle_tol: float
    settle_margin: int
    feature_scale_floor: float
    sensitivity_factor: float
    segment_min_rest_steps: int
    scale_residuals: bool
    consistency_factor: float
    log_hessian_identifiability: bool

    @classmethod
    def from_cfg(cls) -> SysIdConfig:
        """Snapshot the current global ``CFG`` flag values.

        Must be called at entry-point invocation time (never cached at
        import), so ``utils.reset_config`` in tests and experiment
        launchers takes effect.
        """
        return cls(
            warm_start_with_lm=CFG.code_sim_learning_warm_start_with_lm,
            num_mcmc_steps=CFG.code_sim_learning_num_mcmc_steps,
            rollout_num_mcmc_steps=(
                CFG.code_sim_learning_rollout_num_mcmc_steps),
            grid_seed_points=CFG.code_sim_learning_rollout_grid_seed_points,
            grid_sweep_passes=(
                CFG.code_sim_learning_rollout_grid_sweep_passes),
            grid_refine_evals=(
                CFG.code_sim_learning_rollout_grid_refine_evals),
            grid_flat_frac=CFG.code_sim_learning_rollout_grid_flat_frac,
            trim_rms_factor=CFG.code_sim_learning_rollout_trim_rms_factor,
            settle_tol=CFG.code_sim_learning_rollout_settle_tol,
            settle_margin=CFG.code_sim_learning_rollout_settle_margin,
            feature_scale_floor=(
                CFG.code_sim_learning_rollout_feature_scale_floor),
            sensitivity_factor=(
                CFG.code_sim_learning_rollout_sensitivity_factor),
            segment_min_rest_steps=(
                CFG.code_sim_learning_rollout_segment_min_rest_steps),
            scale_residuals=CFG.code_sim_learning_rollout_scale_residuals,
            consistency_factor=(
                CFG.code_sim_learning_rollout_consistency_factor),
            log_hessian_identifiability=(
                CFG.code_sim_learning_log_hessian_identifiability),
        )
