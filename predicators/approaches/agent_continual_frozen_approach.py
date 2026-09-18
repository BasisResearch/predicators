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
    # Whether the harness supplies the frozen model. The zero-shot arm
    # writes its own before the first action, so it keeps the model
    # API reference; the supplied-model arms drop it.
    _frozen_model_supplied = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._frozen_model_source: Optional[str] = None
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_zero_shot"

    def _play_prompt_options(self) -> Dict[str, Any]:
        # The arm statement opens the prompt and the frozen workflow and
        # workbench replace the learning ones (play_prompts).
        return {
            "frozen_section": render("play_frozen",
                                     self._frozen_prompt_section),
            "frozen_model_supplied": self._frozen_model_supplied,
        }

    def _play_model_contract(self, **options: Any) -> str:
        # Supplied models keep only the predicate contract; the zero-shot
        # arm writes its model once, so it gets the writing contract
        # without the fitting guidance.
        return super()._play_model_contract(
            frozen=True, supplied_model=self._frozen_model_supplied, **options)

    def _fit_available(self) -> bool:
        return False

    def _no_model_section(self) -> str:
        return "no_model_zero_shot"

    def _model_status(self, session: ProtocolSession) -> str:
        # A supplied model is installed before the first round opens but
        # is only versioned once the probe loads it; report it as present
        # from the first query on.
        if (not self._frozen_model_supplied
                or self._current_simulator_version is not None):
            return super()._model_status(session)
        n_eps, n_steps = self._episode_counts(session)
        status = render("play_query",
                        "model_supplied",
                        predicates_version=self._current_predicates_version
                        or "none",
                        fit_status=self._fit_status_text(),
                        n_episodes=str(n_eps),
                        n_steps=str(n_steps))
        return status + (f" {self._probe_ext_status}"
                         if self._probe_ext_status else "")

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
                if not path.is_file():
                    raise ValueError(
                        "No ./simulator.py yet. Write it before your first "
                        "real action or reset; the dynamics are sealed at "
                        "that point.")
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

        validation = ctx.probe_validation_provider
        assert validation is not None

        def frozen_validation(**kwargs: Any) -> str:
            self._check_frozen_model()
            if kwargs.get("params") is not None:
                return ("Alternative parameter values are unavailable for "
                        "the frozen model.")
            return validation(**kwargs)

        ctx.probe_validation_provider = frozen_validation
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


class AgentContinualSceneOnlyApproach(AgentContinualZeroShotApproach):
    """Scene-only: the exact scene twin and base calibration, no mechanisms.

    The frozen model is the supplied base simulator with its miscalibrated
    base constants corrected and nothing else: no mechanism code, no
    fitting, no model edits. The agent still plans, infers state and
    adapts from observations. Logs before Sept 18, 2026 carry the arm's
    former name, agent_continual_oracle_scene.
    """

    _save_suffix = "AgentContinualSceneOnly"
    _frozen_prompt_section = "scene_only"
    _frozen_model_supplied = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frozen_model_source = self._scene_source()

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_scene_only"

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
            raise ValueError("Scene-only calibration not audited for " +
                             CFG.env)
        specs = ",\n        ".join(
            f"ParamSpec({name!r}, {value!r}, lo={value!r}, hi={value!r})"
            for name, value in sorted(params.items()))
        return ("# Fixed scene geometry, articulation, and base calibration.\n"
                "# No release, lift, curing, filling, heating or wind code.\n"
                "class SceneOnly(BaseSimulator):\n"
                f"    AGENT_PARAM_SPECS = [{specs}]\n"
                "    RESIDUAL_FEATURES = {}\n"
                "    def _domain_specific_step(self):\n"
                "        pass\n"
                "RESIDUAL_ENV = SceneOnly\n")

    def _fit_status_text(self) -> str:
        return "fixed scene-only base calibration; mechanisms omitted"


class AgentContinualOracleDynamicsApproach(AgentContinualZeroShotApproach):
    """Fixed correct dynamics; infer execution state from observations."""

    _save_suffix = "AgentContinualOracleDynamics"
    _frozen_prompt_section = "oracle_dynamics"
    _frozen_model_supplied = True

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
