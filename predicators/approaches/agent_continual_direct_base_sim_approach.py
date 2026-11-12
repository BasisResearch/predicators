"""Direct + base simulator: the direct agent with the base simulator as a tool.

The arm is the direct agent with the scene files (the model-free arm
under ``continual_provide_scene_package``): its prompt only asks it to
solve the levels. It adds one tool. The harness builds the domain-general
base simulator by itself, ``SceneBase`` with each level's scene loaded
from its manifest and asset files, as EMPIRIC from assets' base loads it
(``scene_manifest_loader``), and ``run_python`` binds it as ``sim``. The
agent can start it from an observed state, run the robot's skills in it
and read the states it predicts, at the loaded physical values or at
values it names.

The arm has no model file, no harness fit, no belief and no gate, and no
agent file is loaded into the simulator: it has no interface for adding a
mechanism. The agent can fit physical values with its own code.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from predicators import utils
from predicators.agent_sdk.base_simulator_probe import \
    BASE_SIMULATOR_PROBE_ABSENT, BASE_SIMULATOR_PROBE_DISABLED, \
    build_base_simulator_tool, supplied_scene_class
from predicators.agent_sdk.sandbox_setup import GeneratedReference, \
    ReferenceFiles, find_repo_root
from predicators.agent_sdk.tools.continual_tools import CONTINUAL_TOOL_NAMES, \
    play_tool_names
from predicators.approaches.agent_continual_approach import \
    AgentContinualModelFreeApproach
from predicators.option_model import _OptionModelBase, _OracleOptionModel
from predicators.settings import CFG
from predicators.structs import LowLevelTrajectory, State

if TYPE_CHECKING:  # pragma: no cover - the run package imports approaches
    from predicators.run.continual import ProtocolSession

# The sandbox reference paths of the scene files, and of the scene of each
# level played so far, which only the simulator reads.
_SCENE_FILES = ("scene/", "assets/")
_LEVEL_SCENES = "scene/levels/"


class AgentContinualDirectBaseSimApproach(AgentContinualModelFreeApproach):
    """The direct agent with the supplied base simulator in ``run_python``."""

    _save_suffix = "AgentContinualDirectBaseSim"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        missing = [
            name for name in ("continual_provide_scene_package",
                              "scene_manifest_loader")
            if not getattr(CFG, name)
        ]
        if missing:
            raise ValueError(
                "Direct + base simulator loads each level's scene from its "
                "manifest and asset files; run it with " + ", ".join(missing) +
                " (as the direct_base_sim_opus entry "
                "of scripts/configs/empiric/approaches.yaml does)")
        super().__init__(*args, **kwargs)
        # The supplied simulator, built on first use and kept for the run:
        # it holds the scene of the level a state belongs to.
        self._base_simulator: Optional[_OracleOptionModel] = None
        # The recording as ``run_python`` holds it, kept current in place.
        self._recording: List[LowLevelTrajectory] = []
        self._tool_context.probe_absent = BASE_SIMULATOR_PROBE_ABSENT
        self._tool_context.probe_disabled = BASE_SIMULATOR_PROBE_DISABLED

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_direct_base_sim"

    # -- The arm's declarations -------------------------------------------

    def _continual_tool_names(self) -> List[str]:
        # The level's reward model is a helper in run_python here (see
        # build_base_simulator_tool), so the arm has no tool of that name.
        return ["run_python"] + play_tool_names(CONTINUAL_TOOL_NAMES)

    def _play_system_prompt(self) -> str:
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.play_prompts import build_play_system_prompt
        return build_play_system_prompt(
            self._continual_tool_names(),
            base_sim_refs=self._scene_asset_paths(),
            scene_package=True,
            base_simulator=True)

    def _round_was_productive(self, session: ProtocolSession, state: Any,
                              steps_before: int, steps_after: int) -> bool:
        # A round of rollouts in the supplied simulator is work, as a
        # round of model work is for the model arm.
        del session, state
        return (steps_after > steps_before
                or self._tool_context.attempt_rollout_count > 0)

    # -- The scene files ----------------------------------------------------

    def _scene_copy_dir(self) -> str:
        """Where the harness keeps the scene files its simulator loads: its own
        copy beside the sandbox, which no edit in the sandbox reaches."""
        return os.path.join(self._get_log_dir(), "base_simulator", "reference")

    def _get_sandbox_reference_files(self) -> ReferenceFiles:
        """The files the direct agent with the scene files receives.

        The simulator's copy holds them too, with the scene of each
        level played so far (``scene_manifest_loader``).
        """
        files = super()._get_sandbox_reference_files()
        self._write_scene_copy({
            path: source
            for path, source in files.items() if path.startswith(_SCENE_FILES)
        })
        return {
            path: source
            for path, source in files.items()
            if not path.startswith(_LEVEL_SCENES)
        }

    def _write_scene_copy(self, files: ReferenceFiles) -> None:
        """Bring the simulator's copy of the scene files up to ``files``."""
        root = Path(self._scene_copy_dir())
        repo_root = find_repo_root()
        for relative, source in files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(source, GeneratedReference):
                if not target.is_file() or \
                        target.read_text(encoding="utf-8") != source.text:
                    target.write_text(source.text, encoding="utf-8")
            elif not target.is_file():
                shutil.copy2(str(repo_root / source), str(target))

    # -- The supplied simulator --------------------------------------------

    def _base_simulator_model(self) -> _OptionModelBase:
        """The real skill controllers over the supplied world: ``SceneBase``
        with the level's scene loaded and nothing else.

        The world has no agent-written subclass, so it has no mechanism
        and no parameter, and each body keeps what the loader gives it
        (:func:`supplied_scene_class`).
        """
        if self._base_simulator is None:
            copy = self._scene_copy_dir()
            if not os.path.isfile(
                    os.path.join(copy, "scene", "scene_manifest.json")):
                self._get_sandbox_reference_files()
            scene = supplied_scene_class(
                CFG.env, self._types,
                os.path.join(copy, "assets"))(use_gui=CFG.option_model_use_gui)
            model = _OracleOptionModel(self._initial_options, scene.simulate)
            model.sim_env = scene
            if CFG.wait_option_terminate_on_atom_change:
                # A Wait ends on the arm's predicates, as a real one does.
                model._abstract_function = (  # pylint: disable=protected-access
                    lambda s: utils.abstract(s, self._get_all_predicates()))
            self._base_simulator = model
            # sim.render draws the supplied world.
            self._tool_context.env = scene
        return self._base_simulator

    def _sandbox_paths(self) -> Tuple[Optional[str], Optional[str]]:
        """The sandbox directory on the host and as the agent names it; the
        session manager makes it after the round's tools are built."""
        if CFG.agent_sdk_use_local_sandbox:
            return os.path.abspath(os.path.join(self._get_log_dir(),
                                                "sandbox")), "."
        return self._tool_context.sandbox_dir, None

    def _round_extra_tools(self, session: ProtocolSession) -> List[Any]:
        """This round's ``run_python`` over the supplied simulator."""
        ctx = self._tool_context
        self._refresh_arm_data(session)
        ctx.probe_option_model_provider = self._base_simulator_model

        def current_observation() -> State:
            # A skill tool's result does not refresh the context's copy.
            obs = session.observe()
            ctx.current_belief = obs.belief
            return obs.frame

        ctx.current_observation_provider = current_observation
        ctx.skill_record_provider = session.skill_record
        sandbox_dir, sandbox_dir_for_agent = self._sandbox_paths()
        return [
            build_base_simulator_tool(
                ctx,
                self._recording,
                sandbox_dir=sandbox_dir,
                sandbox_dir_for_agent=sandbox_dir_for_agent,
                reward_model=CFG.continual_provide_reward_model)
        ]

    def _refresh_arm_data(self, session: ProtocolSession) -> None:
        del session
        self._recording[:] = self._get_all_trajectories()

    def _after_round(self, session: ProtocolSession, state: Any) -> None:
        del session, state
        ctx = self._tool_context
        ctx.probe_option_model_provider = None
        ctx.current_observation_provider = None
        ctx.skill_record_provider = None
        ctx.probe_param_status = None
