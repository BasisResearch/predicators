"""Tests for the natural-language world model approach's harness glue (paper
arm C3)."""
# pylint: disable=protected-access
import os
from types import SimpleNamespace
from typing import Any, List

from predicators import utils
from predicators.agent_sdk import learn_prompts
from predicators.agent_sdk.tools import ToolContext
from predicators.approaches import agent_nl_world_model_approach as anl
from predicators.envs import create_new_env
from predicators.explorers.agent_model_free_explorer import \
    AgentModelFreeExplorer
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG
from predicators.structs import Task

_NOTES = "# Mechanisms\n- the hand moves to the PickPlace parameter\n"


def _cover() -> Any:
    utils.reset_config({
        "env": "cover",
        "num_train_tasks": 2,
        "num_test_tasks": 1,
        "agent_sdk_use_local_sandbox": True,
        "seed": 0,
    })
    env = create_new_env("cover")
    train_tasks = [
        Task(t.task.init, t.task.goal, goal_nl="Cover the target.")
        for t in env.get_train_tasks()
    ]
    options = get_gt_options(env.get_name())
    return env, train_tasks, options


def _bare(env: Any, train_tasks: List[Task], options: Any,
          log_dir: str) -> Any:
    approach = anl.AgentNotesWorldModelApproach.__new__(
        anl.AgentNotesWorldModelApproach)
    approach._types = env.types
    approach._initial_predicates = set(env.predicates)
    approach._initial_options = options
    approach._train_tasks = train_tasks
    approach._tool_context = ToolContext(types=env.types,
                                         predicates=set(env.predicates),
                                         options=options,
                                         train_tasks=train_tasks)
    approach._agent_session = None
    approach._notes = ""
    approach._notes_version = None
    approach._online_learning_cycle = 0
    approach._get_log_dir = lambda: log_dir  # type: ignore[method-assign]
    approach._get_all_options = lambda: options  # type: ignore[method-assign]
    approach._get_all_trajectories = lambda: []  # type: ignore[method-assign]
    approach._offline_dataset = SimpleNamespace(  # type: ignore[assignment]
        trajectories=[])
    approach._online_trajectories = []
    approach._option_model = None
    approach._synthesized_samplers = {}
    return approach


def test_predicate_allowlist_and_paths(tmp_path, monkeypatch) -> None:
    """The kept-predicate allowlist applies; sandbox paths mirror the code
    arms' mapping."""
    env, train_tasks, options = _cover()
    approach = _bare(env, train_tasks, options, str(tmp_path))
    monkeypatch.setattr(CFG, "agent_sim_learn_kept_predicates_names", [])
    assert approach._get_all_predicates() == set(env.predicates)
    monkeypatch.setattr(CFG, "agent_sim_learn_kept_predicates_names",
                        ["Holding"])
    assert {p.name for p in approach._get_all_predicates()} == {"Holding"}
    paths = approach._notes_paths()
    assert paths["notes_file"] == os.path.join(str(tmp_path), "sandbox",
                                               "world_model.md")
    assert paths["notes_file_for_agent"] == "./world_model.md"
    assert approach._get_synthesis_tool_names() == ["run_python"]


def test_learn_notes_gating_and_checkpoint_round_trip(tmp_path,
                                                      monkeypatch) -> None:
    """No data means no session unless zero-shot is on; the document survives a
    checkpoint and is written back into the sandbox."""
    env, train_tasks, options = _cover()
    approach = _bare(env, train_tasks, options, str(tmp_path))
    calls: List[Any] = []
    approach._run_notes_session = calls.append
    monkeypatch.setattr(CFG, "agent_sim_learn_zero_shot", False)
    approach._learn_notes()
    assert not calls
    monkeypatch.setattr(CFG, "agent_sim_learn_zero_shot", True)
    approach._learn_notes()
    assert calls == [[]]
    # A session's document is loaded from the sandbox file.
    paths = approach._notes_paths()
    os.makedirs(paths["base"], exist_ok=True)
    with open(paths["notes_file"], "w", encoding="utf-8") as f:
        f.write(_NOTES)
    approach._load_notes(paths)
    assert approach._notes == _NOTES
    assert approach._notes_version is not None
    assert approach._tool_context.world_model_notes == _NOTES
    assert approach._tool_context.world_model_notes_path == \
        "./world_model.md"
    # Checkpoint round trip into a fresh instance with an empty sandbox.
    saved = approach._extra_save_state()
    assert saved["world_model_notes"] == _NOTES
    other_dir = os.path.join(str(tmp_path), "other")
    other = _bare(env, train_tasks, options, other_dir)
    other._load_extra_save_state(saved)
    assert other._notes == _NOTES
    with open(other._notes_paths()["notes_file"], encoding="utf-8") as f:
        assert f.read() == _NOTES


def test_notes_reach_the_solve_and_explore_prompts(tmp_path) -> None:
    """The document is quoted into the solve prompt, the explore prompt, and
    the system prompt names it; without notes nothing is quoted."""
    env, train_tasks, options = _cover()
    approach = _bare(env, train_tasks, options, str(tmp_path))
    approach._initial_image_section = lambda: ""  # type: ignore[method-assign]
    assert approach._solve_prompt_extra_sections() == ""
    approach._notes = _NOTES
    approach._sync_tool_context()
    extra = approach._solve_prompt_extra_sections()
    assert "World model notes (./world_model.md)" in extra
    assert "hand moves to the PickPlace parameter" in extra
    prompt = approach._build_solve_prompt(train_tasks[0])
    assert "hand moves to the PickPlace parameter" in prompt
    assert prompt.index("World model notes") < prompt.index("## Objects")
    system = approach._get_agent_system_prompt()
    assert "world_model.md" in system and "no simulator" in system
    approach._learning_mode = True
    assert "natural-language document" in approach._get_agent_system_prompt()
    approach._learning_mode = False
    explorer = AgentModelFreeExplorer(set(env.predicates), options, env.types,
                                      env.action_space, train_tasks, 10,
                                      approach._tool_context,
                                      None)  # type: ignore[arg-type]
    explore_prompt = explorer._build_exploration_prompt(0)
    assert "hand moves to the PickPlace parameter" in explore_prompt
    assert explore_prompt.index("World model notes") < \
        explore_prompt.index("## Instructions")


def test_notes_learn_message_and_exec_namespace(tmp_path) -> None:
    """The first message carries the data roster, goals, and the prior-notes
    pointer; the exec namespace exposes the data helpers."""
    env, train_tasks, options = _cover()
    approach = _bare(env, train_tasks, options, str(tmp_path))
    paths = approach._notes_paths()
    message = approach._build_notes_learn_message([], paths)
    assert "0 recorded trajectories" in message
    assert "Cover the target." in message
    assert message.count("Cover the target.") == 1
    assert "Read it first" not in message
    assert "./reference/structs.py" in message
    assert os.path.isfile(
        os.path.join(paths["base"], "reference", "structs.py"))
    os.makedirs(paths["base"], exist_ok=True)
    with open(paths["notes_file"], "w", encoding="utf-8") as f:
        f.write(_NOTES)
    assert "Read it first" in approach._build_notes_learn_message([], paths)
    ns = approach._build_notes_exec_ns([])
    assert set(ns) >= {
        "trajectories", "train_tasks", "is_goal_state", "describe_trajectory",
        "np"
    }
    assert ns["is_goal_state"](train_tasks[0].init, 0) is False


def test_notes_prompt_builders() -> None:
    """Builders render without leftovers; the block is empty without notes."""
    system = learn_prompts.build_notes_learn_system_prompt()
    assert "# Mechanisms" in system and "__" not in system
    assert learn_prompts.render_world_model_notes_block("", "x") == ""
    zero = learn_prompts.render_notes_zero_shot_message()
    assert "No trajectory has been recorded" in zero
