"""No learning: EMPIRIC from assets' planning workflow on a simulator the
harness supplies.

The arm is the ablation of EMPIRIC from assets with all learning of the
physics removed. The harness builds the simulator by itself, as the
Direct + base simulator arm does: the domain-general ``SceneBase`` with
each level's scene loaded from its manifest and asset files
(``supplied_scene_class``, under ``scene_manifest_loader``), from the
harness's own copy of the scene files. The world has no mechanism and no
parameter, each body keeps what the loader gives it, and a feature that
no body pose carries keeps its observed value through the base's feature
store. The agent writes no model file, and no file of the agent's is
loaded.

The agent keeps the planning workflow of EMPIRIC from assets on that
simulator: the state belief from noisy observations, rehearsals on the
joint draws (a draw is a start state with its own planner seed, since
the arm has no parameters), plan refinement, the rehearsal gate and the
rehearsal switches that need no model file, fit or parameter, the claim
check, the predictive check on the bodies' poses, and the level's reward
model.

A withheld thing is absent for the agent, and nothing it reads names
one. Its ``sim`` is a :class:`~predicators.agent_sdk.scene_simulator.
Simulator` that has only the arm's calls, its prompts come from
``scene_simulator_prompts``, its sandbox holds the scene files (and,
under ``continual_provide_simulator_source``, the simulator's source,
listed as that and as nothing to build on), and a traceback it is shown
leaves out the harness's own frames (``agent_sim_learn_fit_invisible``).

The agent reads its run folder, ``<runs>/<approach name>/<experiment
id>/...``, as its working directory, so the approach's registered name
and its menu key are neutral (``agent_continual_from_assets_scene``,
``from_assets_scene_opus``), and the class refuses an experiment id that
says more.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from predicators.agent_sdk.base_simulator_probe import supplied_scene_class
from predicators.agent_sdk.sandbox_setup import GeneratedReference, \
    ReferenceFiles, find_repo_root
from predicators.agent_sdk.scene_simulator import \
    SCENE_SIMULATOR_PROBE_ABSENT, build_scene_simulator_tool
from predicators.agent_sdk.scene_simulator_prompts import \
    build_scene_simulator_claude_md, build_scene_simulator_system_prompt, \
    scene_manifest_legend, scene_simulator_query, scene_simulator_status
from predicators.agent_sdk.tools.context import mark_notes_shown
from predicators.approaches.agent_continual_from_assets_approach import \
    AgentContinualFromAssetsApproach
from predicators.approaches.agent_continual_frozen_approach import \
    AgentContinualZeroShotApproach
from predicators.approaches.scene_package_mixin import SIMULATOR_SOURCE_PATHS
from predicators.settings import CFG
from predicators.structs import Type

if TYPE_CHECKING:  # pragma: no cover - the run package imports approaches
    from predicators.run.continual import ProtocolSession

# The sandbox reference paths of the scene files.
_SCENE_FILES = ("scene/", "assets/")
# The features a body's pose gives an observed object: what the supplied
# world predicts, and what the predictive check scores a step on.
_POSE_FEATURES = ("x", "y", "z", "rot", "roll", "pitch", "yaw")
# What gives the arm away in a path the agent can read: a word for what
# the arm does not have, or for the arm's place among the others.
_GIVEAWAY_IN_PATH = re.compile(
    r"learn|ablat|mechan|param|compar|baseline|frozen|"
    r"(?<![A-Za-z])(?:re|un|mis|no)?_?fit(?:s|ted|ting)?(?![A-Za-z])",
    re.IGNORECASE)
# What the arm needs on, and the switches of EMPIRIC from assets that act
# on a model file, a fit or a parameter, whose text would name one.
_REQUIRED_FLAGS = ("scene_manifest_loader", "continual_uncertainty_decisions",
                   "agent_sim_learn_declared_params_only",
                   "agent_sim_learn_fit_invisible")
_REFUSED_FLAGS = ("continual_require_model_on_test",
                  "continual_skill_preflight",
                  "belief_undeclared_material_screen",
                  "continual_fit_reach_check", "continual_fit_bound_note",
                  "continual_rehearsal_support_note",
                  "continual_rehearsal_speed_note", "belief_material_lines",
                  "code_sim_learning_prior_spans_bounds")
# What the agent is told when the harness cannot build the world.
_LOAD_FAILURE = ("run_python: the simulator could not load this level's "
                 "scene. This is a fault of the harness, not of your code; "
                 "note it in your journal.")


def supplied_scene_source(types: Any) -> str:
    """The file the harness loads as the world: the supplied scene base as it
    is, scored on the pose features of each object type other than the robot.

    It adds no method and no declaration to the base, so the world it
    names has no mechanism and no parameter.
    """
    scored: Dict[str, List[str]] = {}
    for scene_type in sorted(types, key=lambda t: t.name):
        assert isinstance(scene_type, Type)
        pose = [f for f in scene_type.feature_names if f in _POSE_FEATURES]
        if pose and scene_type.name != "robot":
            scored[scene_type.name] = pose
    return ("# The level's scene as the harness loads it, and nothing else.\n"
            "class SuppliedScene(SceneBase):\n"
            f"    RESIDUAL_FEATURES = {scored!r}\n"
            "RESIDUAL_ENV = SuppliedScene\n")


def _with_arm_legend(source: GeneratedReference) -> GeneratedReference:
    """A scene manifest, or a level's record of one, with its legend in this
    arm's wording; its bodies are as they were."""
    record = json.loads(source.text)
    manifest = record.get("manifest", record)
    manifest["legend"] = scene_manifest_legend(manifest["legend"])
    return GeneratedReference(
        json.dumps(record, indent=1, sort_keys=False) + "\n", source.origin)


class AgentContinualNoLearningApproach(AgentContinualFromAssetsApproach,
                                       AgentContinualZeroShotApproach):
    """EMPIRIC from assets' planning workflow on the supplied scene simulator,
    with nothing to write, fit or change."""

    _save_suffix = "AgentContinualNoLearning"
    # No parameter belief: nothing is declared, widened or sampled.
    _belief_over_guesses = False
    # The harness writes the world's file, beside the sandbox.
    _frozen_model_supplied = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        missing = [name for name in _REQUIRED_FLAGS if not getattr(CFG, name)]
        if int(CFG.belief_joint_draws) <= 0:
            missing.append("belief_joint_draws > 0")
        if missing:
            raise ValueError(
                "No learning rehearses on joint draws in a scene the harness "
                "loads, and its agent reads of no harness estimate; run it "
                "with " + ", ".join(missing) + " (as the "
                "from_assets_scene_opus entry of scripts/configs/empiric/"
                "approaches.yaml does)")
        refused = [name for name in _REFUSED_FLAGS if getattr(CFG, name)]
        if float(CFG.continual_rehearsal_game_over_gate) > 0.0:
            refused.append("continual_rehearsal_game_over_gate")
        if refused:
            raise ValueError(
                "No learning has no model file, fit or parameter, and these "
                "switches act on one and say so to the agent: " +
                ", ".join(refused) + "; leave them off")
        for part in (self.get_name(), str(CFG.experiment_id)):
            if _GIVEAWAY_IN_PATH.search(part):
                raise ValueError(
                    f"The run folder's path holds {part!r}, and the agent "
                    "reads that path as its working directory; give this "
                    "arm a name, a menu key and a round that say nothing "
                    "of what the arm has or lacks")
        # The namespace the agent's run_python holds this round.
        self._agent_namespace: Optional[Dict[str, Any]] = None
        super().__init__(*args, **kwargs)
        self._frozen_model_source = supplied_scene_source(self._types)
        ctx = self._tool_context
        ctx.probe_absent = SCENE_SIMULATOR_PROBE_ABSENT
        ctx.supplied_simulator = True

    @classmethod
    def get_name(cls) -> str:
        return "agent_continual_from_assets_scene"

    def _get_sandbox_claude_md(self) -> Optional[str]:
        # This arm's own sandbox rules.
        return build_scene_simulator_claude_md(
            source=CFG.continual_provide_simulator_source)

    # -- The scene files ----------------------------------------------------

    def _scene_copy_dir(self) -> str:
        """Where the harness keeps the scene files its simulator loads: its own
        copy beside the sandbox, which no edit in the sandbox reaches."""
        return os.path.join(self._get_log_dir(), "supplied_model", "reference")

    def _get_sandbox_reference_files(self) -> ReferenceFiles:
        """The scene files EMPIRIC from assets receives; the simulator's copy
        holds the same files.

        Under ``continual_provide_simulator_source`` the agent also gets
        the simulator's source as EMPIRIC from assets reads it: copies
        to read, which the simulator never loads (it is built from the
        harness's own modules, :meth:`_scene_base_class`).
        """
        files = self._scene_asset_files()
        for path, source in files.items():
            if path.startswith("scene/") and path.endswith(".json"):
                assert isinstance(source, GeneratedReference)
                files[path] = _with_arm_legend(source)
        self._write_scene_copy({
            path: source
            for path, source in files.items() if path.startswith(_SCENE_FILES)
        })
        if CFG.continual_provide_simulator_source:
            files.update(self._simulator_source_files())
        return files

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

    def _base_sim_reference_paths(self) -> List[str]:
        return self._scene_asset_paths()

    # -- The supplied simulator --------------------------------------------

    def _scene_base_class(self) -> type:
        """The class the world's file subclasses: ``SceneBase`` bound to the
        deployment and to the harness's copy of the scene files."""
        if self._scene_base is None:
            copy = self._scene_copy_dir()
            if not os.path.isfile(
                    os.path.join(copy, "scene", "scene_manifest.json")):
                self._get_sandbox_reference_files()
            self._scene_base = supplied_scene_class(
                CFG.env, self._types, os.path.join(copy, "assets"))
        return self._scene_base

    def _load_extra_save_state(self, save_dict: Dict[str, Any]) -> None:
        # The world's file is the harness's, whatever a checkpoint holds.
        super()._load_extra_save_state(
            dict(save_dict,
                 frozen_model_source=supplied_scene_source(self._types)))

    def _simulator_load_failure(self) -> str:
        return _LOAD_FAILURE

    def _base_physics_probe_model(self) -> Any:
        # Reached only when the world's file is gone from the harness's
        # side.
        raise RuntimeError(_LOAD_FAILURE)

    # -- No agent-written file is read --------------------------------------

    def _compute_extra_synthesis_paths(self, base: str) -> Dict[str, str]:
        # The arm invents no predicate: the predicate loader looks beside
        # the world's file, where the agent writes nothing.
        del base
        hidden = os.path.join(self._get_log_dir(), "supplied_model")
        return {
            "predicates_file": os.path.join(hidden, "predicates.py"),
            "predicates_versions_dir": os.path.join(hidden,
                                                    "predicates_versions"),
            "predicates_file_for_agent": "(not exposed)",
        }

    def _rehydrate_extra_artifacts(self, base: str) -> None:
        del base

    def _install_extra_synthesis_surfaces(self, exec_ns: Dict[str, Any],
                                          base_pred_triples: List[Any],
                                          inferred_hint: Dict[str, List[str]],
                                          extra_paths: Dict[str, str]) -> None:
        del exec_ns, base_pred_triples, inferred_hint, extra_paths

    def _round_hooks(self, session: ProtocolSession) -> Dict[str, list]:
        # No file of the agent's is a model file, so none is snapshotted.
        del session
        return {}

    def _load_probe_extension(self, exec_ns: Dict[str, Any],
                              sandbox_dir: str) -> None:
        # The shared round setup offers the namespace of its own
        # run_python, which holds calls this arm does not have. The
        # agent's helpers load into the namespace the agent holds.
        if exec_ns is self._agent_namespace:
            super()._load_probe_extension(exec_ns, sandbox_dir)

    # -- The round's tools --------------------------------------------------

    def _round_extra_tools(self, session: ProtocolSession) -> List[Any]:
        """This round's ``run_python`` over the supplied simulator.

        The shared setup installs the workflow's providers and gates for
        the round; its own ``run_python`` describes a model file, so the
        arm attaches the scene simulator's instead.
        """
        self._agent_namespace = None
        super()._round_extra_tools(session)
        ctx = self._tool_context
        paths = self._resolve_synthesis_paths()
        namespace = self._build_synthesis_exec_ns(self._workbench.trajectories)
        # Names for writing a model, and the reward model that sweeps
        # parameters: build_scene_simulator_tool binds the arm's own.
        for name in ("ParamSpec", "evaluate_trajectory"):
            namespace.pop(name, None)
        tool = build_scene_simulator_tool(
            ctx,
            namespace,
            sandbox_dir=paths.base,
            sandbox_dir_for_agent=paths.sandbox_dir_for_agent,
            # The notes gate lets a request through once the notes of its
            # rehearsal have been in an output the agent was given.
            output_observer=((lambda text: mark_notes_shown(
                ctx.live_rehearsals, ctx.shown_notes, text))
                             if CFG.continual_rehearsal_notes_gate else None))
        self._agent_namespace = namespace
        self._load_probe_extension(namespace, paths.base)
        return [tool]

    def _round_was_productive(self, session: ProtocolSession, state: Any,
                              steps_before: int, steps_after: int) -> bool:
        # A round of rollouts in the simulator is work; the world's file
        # loads every round and is none.
        del session, state
        return (steps_after > steps_before
                or self._tool_context.attempt_rollout_count > 0)

    # -- The prompt and the records -----------------------------------------

    def _play_system_prompt(self) -> str:
        return build_scene_simulator_system_prompt(
            self._continual_tool_names(),
            self._scene_asset_paths(),
            source_refs=(SIMULATOR_SOURCE_PATHS
                         if CFG.continual_provide_simulator_source else ()))

    def _model_status(self, session: ProtocolSession) -> str:
        n_eps, n_steps = self._episode_counts(session)
        status = scene_simulator_status(n_eps, n_steps)
        return status + (f" {self._probe_ext_status}"
                         if self._probe_ext_status else "")

    def _build_query(self, session: ProtocolSession, kind: str) -> str:
        return scene_simulator_query(super()._build_query(session, kind))

    def _fit_status_text(self) -> str:
        return "the supplied scene"

    def _round_record_extra(self, state: Any) -> str:
        del state
        return ""

    def _append_model_journal(self, paths: Any, outcome: str) -> None:
        # attempts.md records what the agent did; the world's file is the
        # harness's.
        del paths
        logging.info("[Continual agent] supplied scene after round %d: %s",
                     self._rounds_played + 1, outcome)
