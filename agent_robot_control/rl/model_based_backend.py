"""Model-based backend placeholder (PLAN.md Section 7, last part).

Planned design: an action-conditioned PTv3 particle flow model trained on the
transitions logged by ``SimSession`` (``transitions/shard_*.npz``: particles,
EE state, joint action, next particles), with MPC (CEM or MPPI over EE-delta
sequences) scoring imagined rollouts with the same agent reward function.
Requires CUDA (``uv sync --extra ptv3``) and runs on the cluster.
"""
from __future__ import annotations

from agent_robot_control.rl.backend import RLRequest, RLResult


class ModelBasedBackend:
    """Not implemented yet; exists so the tool surface is stable."""

    def run(self, session, request: RLRequest) -> RLResult:
        raise NotImplementedError(
            "run_model_based_rl_on_particles is not implemented yet. The "
            "world-model + MPC backend is planned (see "
            "agent_robot_control/PLAN.md, Section 7). Use "
            "run_rl_on_particles (model-free) instead.")
