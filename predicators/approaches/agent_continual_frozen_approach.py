"""Continual models whose dynamics are fixed before real interaction."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from predicators.agent_sdk.prompt_templates import render
from predicators.approaches.agent_continual_approach import \
    AgentContinualApproach
from predicators.envs.pybullet_balloons_base import PyBulletBalloonsBaseEnv
from predicators.settings import CFG

if TYPE_CHECKING:
    from predicators.run.continual import ProtocolSession


class AgentContinualZeroShotApproach(AgentContinualApproach):
    """Synthesize once from the initial task, then freeze code and values."""

    _save_suffix = "AgentContinualZeroShot"
    _frozen_prompt_section = "zero_shot"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._frozen_model_source: Optional[str] = None
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_zero_shot"

    def _play_system_prompt(self) -> str:
        return (super()._play_system_prompt() + "\n\n" +
                render("play_frozen", self._frozen_prompt_section))

    def _check_frozen_model(self) -> None:
        if self._frozen_model_source is None:
            return
        path = Path(self._resolve_synthesis_paths().simulator_file)
        if not path.is_file() or path.read_text(
                encoding="utf-8") != self._frozen_model_source:
            raise ValueError("Dynamics are frozen. Restore simulator.py to "
                             "its pre-interaction contents before continuing.")

    def _restore_frozen_model(self) -> None:
        if self._frozen_model_source is not None:
            path = Path(self._resolve_synthesis_paths().simulator_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self._frozen_model_source, encoding="utf-8")

    def _round_extra_tools(self, session: ProtocolSession) -> List[Any]:
        self._restore_frozen_model()
        tools = super()._round_extra_tools(session)
        ctx = self._tool_context
        provider = ctx.probe_option_model_provider
        assert provider is not None

        def frozen_provider() -> Any:
            self._check_frozen_model()
            return provider()

        def no_fit(**_kwargs: Any) -> str:
            return "Numerical fitting is unavailable for the frozen model."

        def before_action() -> None:
            # Load first: invalid or missing code cannot consume a step.
            frozen_provider()
            if self._frozen_model_source is None:
                path = Path(self._resolve_synthesis_paths().simulator_file)
                self._frozen_model_source = path.read_text(encoding="utf-8")
                # Save the seal before the first charge, including when a
                # preemption interrupts the invocation itself.
                self.save(session.level_index)

        ctx.probe_option_model_provider = frozen_provider
        ctx.probe_fit_provider = no_fit
        residuals = ctx.probe_residuals_provider
        assert residuals is not None

        def frozen_residuals(**kwargs: Any) -> str:
            self._check_frozen_model()
            if kwargs.get("path") is not None:
                return ("Alternative dynamics files are unavailable for "
                        "the frozen model.")
            if kwargs.get("phys_params") is not None:
                return ("Alternative parameter values are unavailable for "
                        "the frozen model.")
            if kwargs.get("fit_params") or kwargs.get("sweep_params"):
                return no_fit()
            return residuals(**kwargs)

        ctx.probe_residuals_provider = frozen_residuals
        ctx.before_real_action = before_action
        return tools

    def _after_round(self, session: ProtocolSession, state: Any) -> None:
        # Never deploy an unapproved edit made after the last action of a
        # round. Predicates, plans and journal remain editable.
        self._restore_frozen_model()
        try:
            super()._after_round(session, state)
        finally:
            self._tool_context.before_real_action = None

    def _fit_status_text(self) -> str:
        return "pre-interaction declared parameters; numerical fitting disabled"

    def _extra_save_state(self) -> Dict[str, Any]:
        state = super()._extra_save_state()
        state["frozen_model_source"] = self._frozen_model_source
        return state

    def _load_extra_save_state(self, save_dict: Dict[str, Any]) -> None:
        self._frozen_model_source = save_dict.get("frozen_model_source")
        super()._load_extra_save_state(save_dict)

    def _rehydrate_from_artifacts(self) -> None:
        self._restore_frozen_model()
        super()._rehydrate_from_artifacts()


class AgentContinualOracleSceneApproach(AgentContinualZeroShotApproach):
    """Known scene assets and base calibration, with no added mechanisms."""

    _save_suffix = "AgentContinualOracleScene"
    _frozen_prompt_section = "oracle_scene"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frozen_model_source = self._scene_source()

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_oracle_scene"

    @staticmethod
    def _scene_source() -> str:
        # The supplied base already has exact assets and robot articulation.
        # Only these domains deliberately miscalibrate base body parameters
        # in this sweep. Keep the environment's task-generation flags intact.
        params: Dict[str, float] = {}
        if CFG.env == "pybullet_domino":
            params["lateral_friction"] = float(CFG.domino_true_friction)
        elif CFG.env == "pybullet_balloons":
            for i, (name, _) in enumerate(PyBulletBalloonsBaseEnv.BOX_PALETTE):
                params[f"mass_{name}"] = float(CFG.balloons_box_masses[i])
            params["air_drag"] = float(CFG.balloons_drag)
        elif CFG.env not in {
                "pybullet_bridge", "pybullet_fan", "pybullet_boil"
        }:
            raise ValueError("Oracle scene calibration not audited for " +
                             CFG.env)
        specs = ",\n        ".join(
            f"ParamSpec({name!r}, {value!r}, lo={value!r}, hi={value!r})"
            for name, value in sorted(params.items()))
        return ("# Fixed scene geometry, articulation, and base calibration.\n"
                "# No release, lift, curing, filling, heating or wind code.\n"
                "class OracleScene(BaseSimulator):\n"
                f"    AGENT_PARAM_SPECS = [{specs}]\n"
                "    RESIDUAL_FEATURES = {}\n"
                "    def _domain_specific_step(self):\n"
                "        pass\n"
                "RESIDUAL_ENV = OracleScene\n")

    def _fit_status_text(self) -> str:
        return "fixed oracle base calibration; added mechanisms omitted"


class AgentContinualOracleDynamicsApproach(AgentContinualZeroShotApproach):
    """Fixed correct dynamics; infer execution state from observations."""

    _save_suffix = "AgentContinualOracleDynamics"
    _frozen_prompt_section = "oracle_dynamics"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from predicators.code_sim_learning.continual_oracle import \
            oracle_source  # pylint: disable=import-outside-toplevel

        # Fail before creating an agent or planning world for an unsupported
        # domain. An incomplete oracle must not become an agent-failure seed.
        source = oracle_source()
        super().__init__(*args, **kwargs)
        self._frozen_model_source = source

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_oracle_dynamics"

    def _fit_status_text(self) -> str:
        return "fixed oracle mechanisms and parameters; no oracle controller"
