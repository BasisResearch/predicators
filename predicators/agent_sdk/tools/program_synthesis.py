"""The synthesis-session tool surface for the program world model arm.

One ``run_python`` tool whose namespace binds the recorded data, the
``sim`` probe over the CANDIDATE ``world_model.py`` (reset / task / run /
refine / render / predicates), and ``sim.score`` - the Pinductor
particle-filter kernel pseudo-likelihood of the candidate on the
recorded option-level trajectories (see
:mod:`code_sim_learning.program_world_model`). There is no fit: the
program has no numeric parameters to estimate; the agent edits the
program and re-scores.
"""
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from predicators.agent_sdk.tools.python_exec import _make_python_exec_tool
from predicators.agent_sdk.tools.results import _make_coercing_tool, \
    _make_spilling_text_result
from predicators.agent_sdk.tools.sandbox_guard import _scrub_host_paths
from predicators.agent_sdk.tools.snapshots import _ArtifactSnapshotter
from predicators.code_sim_learning.program_world_model import \
    ProgramWorldModel, feature_scales, load_program_world_model, \
    score_program
from predicators.structs import LowLevelTrajectory, ParameterizedOption, \
    Predicate, Type

CandidateLoader = Callable[[Optional[str]],
                           Tuple[Optional[ProgramWorldModel], Optional[str],
                                 Optional[str]]]


@dataclasses.dataclass
class ProgramSynthesisToolkit:
    """What ``create_program_synthesis_tools`` builds for one session.

    ``tools`` are the MCP tools to attach; ``score_runner`` is the
    ``sim.score`` backend (installed as
    ``ToolContext.probe_score_provider``); ``load_candidate`` snapshots
    and loads the current ``world_model.py`` (the candidate probe model
    provider builds on it). All share one snapshotter, so every report
    carries consistent ``[cycle_XXX_vers_YYY]`` tags.
    """
    tools: list
    score_runner: Callable[..., str]
    load_candidate: CandidateLoader


_RUN_PYTHON_DESCRIPTION = (
    "Execute Python code (`code`, or `path` to a .py file you wrote in "
    "the sandbox) for ad-hoc data exploration and model checking. "
    "Available variables: trajectories (List[LowLevelTrajectory]; each "
    "has `is_demo`, `train_task_idx`, `states`, `actions`; each action's "
    "`get_option()` is the skill that produced it), train_tasks "
    "(List[Task]; each has `init`, `goal`, `goal_holds(state)`), "
    "is_goal_state (callable: state, task_idx -> bool), "
    "describe_trajectory(traj_idx, include_states=True, "
    "include_atoms=False, max_timesteps=10), np, and (when the env "
    "defines task evaluators) evaluate_trajectory(states, actions=None, "
    "task_idx=0) -> {reward, solved}. print() output is returned; the "
    "namespace persists across calls; oversize output is saved under "
    "`tool_outputs/run_python/` and previewed. This namespace ALSO binds "
    "`sim`, a BeliefProbe over the CANDIDATE world model: your current "
    "world_model.py, reloaded automatically when the file changes "
    "(errors until a loadable file exists). `sim.score(traj_idxs=None, "
    "num_particles=None)` is the model's score on the recorded "
    "trajectories (particle-filter kernel pseudo-likelihood over the "
    "hidden state; 0 is perfect) with per-feature errors and the worst "
    "transitions - the inner-loop signal; `sim.reset(task_idx, "
    "mods=None)` sets the current state to a train task's init "
    "(task_idx is required in this session); `sim.task(task_idx)` "
    "describes a train task; `sim.run(plan_text, render=True, trials=1)` "
    "executes an option plan FROM THE CURRENT STATE through the "
    "candidate model (subgoal annotations are CHECKED); "
    "`sim.refine(plan, require_goal=False)` is the backtracking "
    "parameter search over a plan sketch; `sim.render(label, "
    "annotations=[...])` renders the current state; `sim.snapshot()` / "
    "`sim.restore(id)` bank and rewind the state; `sim.predicates()` "
    "scores the predicates in predicates.py on the recorded data. Probe "
    "rollouts are CANDIDATE predictions - never confuse them with the "
    "recorded `trajectories`. This tool does NOT define the model: "
    "write `world_model.py` for that.")


def create_program_synthesis_tools(
    exec_ns: Dict[str, Any],
    *,
    trajectories: List[LowLevelTrajectory],
    predicates: Set[Predicate],
    types: Set[Type],
    options: Set[ParameterizedOption],
    world_model_file: str,
    versions_dir: str,
    sandbox_dir: Optional[str] = None,
    sandbox_dir_for_agent: Optional[str] = None,
    cycle_index_provider: Optional[Callable[[], int]] = None,
    budget_check: Optional[Callable[[], None]] = None,
    rng: Optional[np.random.Generator] = None,
) -> ProgramSynthesisToolkit:
    """Build the program-synthesis session's tools and probe backends.

    Args:
        exec_ns: The persistent ``run_python`` namespace (data helpers
            already bound; the caller merges the probe in).
        trajectories: The recorded trajectories the score runs on.
        predicates: The predicates used to segment trajectories into
            option transitions.
        types, options: The env vocabulary the program is exec'd with.
        world_model_file, versions_dir: Host paths of the live artifact
            and its snapshot directory.
        sandbox_dir, sandbox_dir_for_agent: Host / agent-visible
            sandbox roots for output spilling and path display.
        cycle_index_provider: Current online cycle, for snapshot tags.
        budget_check: Called before a score; raises to stop the call
            when the session's budget is spent.
        rng: The scorer's random source (particle draws / resampling).
    """
    # pylint: disable=import-outside-toplevel
    from claude_agent_sdk import tool as _sdk_tool

    from predicators.settings import CFG

    # pylint: enable=import-outside-toplevel
    tool = _make_coercing_tool(_sdk_tool)
    snapshotter = _ArtifactSnapshotter(
        live_file=world_model_file,
        versions_dir=versions_dir,
        artifact_name="world_model",
        cycle_index_provider=cycle_index_provider,
        missing_file_hint=("Use Write to create it with LATENT_FEATURES, "
                           "initial_latent and transition."),
    )
    _text = _make_spilling_text_result(sandbox_dir,
                                       agent_prefix=sandbox_dir_for_agent)
    scales = feature_scales(trajectories)
    score_rng = rng if rng is not None else np.random.default_rng(CFG.seed)

    def load_candidate(
        path: Optional[str] = None
    ) -> Tuple[Optional[ProgramWorldModel], Optional[str], Optional[str]]:
        """Snapshot then load ``path`` (default: the live file).

        Returns ``(program, version_tag, error)``; ``error`` is a ready-
        to-return message when the file is missing or fails to load.
        """
        raw, tag, err = snapshotter.snapshot(path)
        if err is not None:
            return None, tag, err
        assert raw is not None
        program, load_err = load_program_world_model(raw.decode("utf-8"),
                                                     types, predicates,
                                                     options)
        if load_err is not None:
            return None, tag, (f"[{tag}] Error loading "
                               f"{path or world_model_file}:\n"
                               f"{_scrub_host_paths(load_err)}")
        return program, tag, None

    def run_score(path: Optional[str] = None,
                  traj_idxs: Optional[List[int]] = None,
                  num_particles: Optional[int] = None) -> str:
        """The ``sim.score`` backend (see the module docstring)."""
        if budget_check is not None:
            budget_check()
        program, tag, err = load_candidate(path)
        if err is not None:
            return err
        assert program is not None
        if not trajectories:
            return (f"[{tag}] No recorded trajectories to score against "
                    "in this session: validate the model with sim.refine "
                    "/ sim.run rollouts instead.")
        selected: Optional[Set[int]] = None
        if traj_idxs is not None:
            bad = sorted(i for i in traj_idxs
                         if not 0 <= i < len(trajectories))
            if bad:
                return (f"[{tag}] Error: traj_idxs {bad} out of range "
                        f"(0-{len(trajectories) - 1}).")
            selected = set(int(i) for i in traj_idxs)
        k = int(num_particles or CFG.agent_program_belief_particles)
        score = score_program(
            program,
            trajectories,
            predicates,
            num_particles=max(1, k),
            kernel_bandwidth=CFG.agent_program_kernel_bandwidth,
            rng=score_rng,
            scales=scales,
            max_examples=CFG.agent_program_score_max_examples,
            traj_indices=selected)
        return f"[{tag}] " + score.render()

    run_python = _make_python_exec_tool(
        tool,
        name="run_python",
        description=_RUN_PYTHON_DESCRIPTION,
        exec_ns=exec_ns,
        sandbox_dir=sandbox_dir,
        sandbox_dir_for_agent=sandbox_dir_for_agent,
        text_result=_text,
        call_timeout_s=CFG.agent_sdk_synthesis_python_call_timeout,
    )
    return ProgramSynthesisToolkit(tools=[run_python],
                                   score_runner=run_score,
                                   load_candidate=load_candidate)
