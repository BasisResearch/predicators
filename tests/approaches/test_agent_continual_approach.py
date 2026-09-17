"""Tests for AgentContinualApproach: the play loop over a scripted agent that
drives the real play tools, on pybullet_boil (the sim-learning family needs a
PyBullet env to construct)."""
import asyncio
import json
import os
import pickle
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

from predicators import utils
from predicators.agent_sdk.belief_probe import BeliefProbe
from predicators.agent_sdk.sandbox_setup import trajectories_path
from predicators.approaches import create_approach
from predicators.approaches.agent_continual_approach import \
    AgentContinualApproach
from predicators.code_sim_learning.fit_space import FitResult
from predicators.code_sim_learning.latent_tracker import \
    make_subclass_latent_tracker
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.run.level_players import create_level_player
from predicators.structs import Dataset, Predicate
from tests.code_sim_learning.test_subclass_model_state import _MemoryModel


def _config(tmp_path: Any, **overrides: Any) -> None:
    utils.reset_config({
        "env":
        "pybullet_boil",
        "approach":
        "agent_continual",
        "seed":
        0,
        "num_train_tasks":
        1,
        "num_test_tasks":
        1,
        "boil_goal":
        "simple",
        "boil_num_jugs_train": [1],
        "boil_num_jugs_test": [1],
        "boil_num_burner_train": [1],
        "boil_num_burner_test": [1],
        "option_model_use_gui":
        False,
        "horizon":
        200,
        "experiment_protocol":
        "continual",
        "continual_steps_per_level":
        500,
        "continual_render":
        False,
        "continual_max_idle_rounds":
        3,
        "continual_runs_dir":
        os.path.join(str(tmp_path), "runs"),
        "approach_dir":
        os.path.join(str(tmp_path), "saved"),
        "agent_sdk_use_local_sandbox":
        True,
        "agent_sim_learn_kept_predicates_names": ["Holding"],
        "experiment_id":
        "agenttest",
        **overrides,
    })


def _make_approach() -> Any:
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    options = get_gt_options(env.get_name())
    approach = create_approach("agent_continual", env.predicates, options,
                               env.types, env.action_space,
                               [t.task for t in env.get_train_tasks()])
    assert isinstance(approach, AgentContinualApproach)
    return env, approach


def _call(approach: Any, name: str, **args: Any) -> str:
    tools = approach._tool_context.extra_mcp_tools  # pylint: disable=protected-access
    tool = next(t for t in tools if t.name == name)
    result = asyncio.run(tool.handler(dict(args)))
    text = result["content"][0]["text"]
    return ("ERROR: " + text) if result.get("is_error") else text


def _result(turns: int = 3, cost: float = 0.25) -> List[Dict[str, Any]]:
    return [{
        "type": "assistant",
        "content": [{
            "type": "text",
            "text": "done"
        }]
    }, {
        "type": "result",
        "subtype": "success",
        "num_turns": turns,
        "total_cost_usd": cost,
        "is_error": False,
        "result": "done",
        "session_id": "sess-1",
    }]


def test_fit_status_text_is_a_point_estimate_line() -> None:
    """The prompt's fit status names each fitted parameter's estimate and the
    sample count, not the result's repr."""
    result = FitResult(names=["lateral_friction", "chain_fwd_min"],
                       samples=np.array([[0.48989795, 0.04], [0.51, 0.04]]),
                       log_probs=np.array([0.0, -0.1]),
                       jacobian=np.zeros((3, 2)))
    render = AgentContinualApproach._fit_status_text  # pylint: disable=protected-access
    fitted: Any = SimpleNamespace(_last_fit_result=result,
                                  _probe_fit_state=lambda: {})
    text = render(fitted)
    assert text.startswith("fitted 2 parameter(s) from 2 posterior")
    assert "lateral_friction=" in text and "chain_fwd_min=0.04" in text
    assert "jacobian" not in text and "array(" not in text
    empty: Any = SimpleNamespace(_last_fit_result=None,
                                 _probe_fit_state=lambda: {})
    assert render(empty) == "no fit result"


def test_predicates_install_refreshes_the_session(tmp_path: Any) -> None:
    """A draft installed by ``sim.predicates()`` reaches the run's abstraction
    at once (the observation's atoms, Wait targets, divergence checks), not at
    the next level start or model publish."""
    _config(tmp_path)
    _, approach = _make_approach()
    session: Any = SimpleNamespace(abstract_predicates=set())
    approach._play_session = session  # pylint: disable=protected-access
    hi = Predicate("Hi", [], lambda s, o: True)
    approach._learned_predicates = {hi}  # pylint: disable=protected-access
    approach._on_predicates_installed()  # pylint: disable=protected-access
    assert hi in session.abstract_predicates
    # Between levels there is no session to refresh.
    approach._play_session = None  # pylint: disable=protected-access
    approach._on_predicates_installed()  # pylint: disable=protected-access
    assert hi in session.abstract_predicates


@pytest.mark.slow
def test_current_probe_refreshes_memory_after_parameter_change(
        tmp_path: Any, monkeypatch: Any) -> None:
    """A post-fit current-state rollout uses the revised episode memory."""
    _config(tmp_path)
    env, approach = _make_approach()
    params = {"rate": .25}
    monkeypatch.setattr(approach, "model_state_revision",
                        lambda: tuple(params.items()))
    monkeypatch.setattr(
        approach, "make_latent_tracker",
        lambda: make_subclass_latent_tracker(_MemoryModel, lambda: params))

    def fake_query(*_args: Any, **_kwargs: Any) -> List[Dict[str, Any]]:
        zero = [0.0] * env.action_space.shape[0]
        assert "step applied" in _call(approach, "env_step", action=zero)
        ctx = approach._tool_context  # pylint: disable=protected-access
        assert ctx.current_observation.latent["charge"] == .25
        # Publishing a fit changes parameters without another real action
        # or env_observe call. Reset must reconstruct the current estimate.
        params["rate"] = .5
        probe = BeliefProbe(ctx).reset(current=True)
        state = probe._require_state()  # pylint: disable=protected-access
        assert state.latent is not None and state.latent["charge"] == .5
        assert ctx.current_observation.latent["charge"] == .5
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", fake_query)
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert card.levels[0].steps == 1


@pytest.mark.slow
def test_current_probe_loads_edited_subclass_before_replaying(
        tmp_path: Any, monkeypatch: Any) -> None:
    """Editing a model mid-episode reconstructs memory at carried values."""
    _config(tmp_path)
    env, approach = _make_approach()
    source = '''
class Counter(BaseSimulator):
    AGENT_PARAM_SPECS = [ParamSpec("rate", .25, lo=0.0, hi=1.0)]
    MODEL_STATE_INIT = {"charge": 0.0}
    RESIDUAL_FEATURES = {}

    @classmethod
    def update_model_state(cls, observation, model_state, params, action):
        model_state["charge"] += params["rate"] * 1.0

RESIDUAL_ENV = Counter
'''

    def fake_query(*_args: Any, **_kwargs: Any) -> List[Dict[str, Any]]:
        # pylint: disable=protected-access
        ctx = approach._tool_context
        assert "step applied" in _call(approach,
                                       "env_step",
                                       action=[0.0] *
                                       env.action_space.shape[0])
        assert ctx.current_observation.latent is None
        path = os.path.join(ctx.sandbox_dir, "simulator.py")
        with open(path, "w", encoding="utf-8") as file:
            file.write(source)
        probe = BeliefProbe(ctx).reset(current=True)
        assert probe._require_state().latent == {"charge": .25}
        approach._apply_identified_physical_params({"rate": .5})
        probe.reset(current=True)
        assert probe._require_state().latent == {"charge": .5}
        with open(path, "w", encoding="utf-8") as file:
            file.write(source.replace('* 1.0', '* 2.0'))
        probe.reset(current=True)
        assert probe._require_state().latent == {"charge": 1.0}
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", fake_query)
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert card.levels[0].steps == 1
    assert approach._tool_context.current_observation_provider is None  # pylint: disable=protected-access


@pytest.mark.slow
@pytest.mark.parametrize("fit_then_edit", [False, True])
def test_round_end_deploys_without_fitting(tmp_path: Any, monkeypatch: Any,
                                           caplog: Any,
                                           fit_then_edit: bool) -> None:
    """A real play round can deploy an unfitted model without optimizing it."""
    # pylint: disable=protected-access
    _config(tmp_path)
    env, approach = _make_approach()
    source = '''
class Counter(BaseSimulator):
    AGENT_PARAM_SPECS = [ParamSpec("rate", .25, lo=0.0, hi=1.0)]
    MODEL_STATE_INIT = {"charge": 0.0}
    RESIDUAL_FEATURES = {}

    @classmethod
    def update_model_state(cls, observation, model_state, params, action):
        model_state["charge"] += params["rate"]

RESIDUAL_ENV = Counter
'''

    def fake_query(*_args: Any, **_kwargs: Any) -> List[Dict[str, Any]]:
        assert "step applied" in _call(approach,
                                       "env_step",
                                       action=[0.0] *
                                       env.action_space.shape[0])
        path = os.path.join(approach._tool_context.sandbox_dir, "simulator.py")
        with open(path, "w", encoding="utf-8") as file:
            file.write(source)
        output = _call(approach,
                       "run_python",
                       code="sim.reset(current=True); print('loaded')")
        assert "loaded" in output and not output.startswith("ERROR")
        if fit_then_edit:
            output = _call(approach, "run_python", code="print(sim.fit())")
            assert not output.startswith("ERROR")
            assert approach._probe_fit_state().get("fit_result") is not None
            with open(path, "a", encoding="utf-8") as file:
                file.write("\n# An edit after the explicit fit.\n")
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    monkeypatch.setattr(approach, "_query_agent_sync", fake_query)
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert card.levels[0].steps == 1
    assert approach._last_round_modelled
    assert "FIT FALLBACK" not in caplog.text
    assert approach._last_fit_result is None
    assert approach._identified_physical_params == {"rate": .25}
    assert "UNFITTED" in approach._fit_status_text()
    assert card.levels[0].sandbox.get("fits", 0) == int(fit_then_edit)


@pytest.mark.slow
def test_play_loop_with_a_scripted_agent(tmp_path: Any) -> None:
    """Two rounds of one conversation: act in the first, which also carries the
    model workbench, and stop; give up in the second.

    A round gets ``run_python`` with the ``sim`` probe over the agent's
    model files; when the agent writes no simulator, no model is
    deployed and the query says so. The data follows the recording
    inside the round: after every env call the episode in progress is in
    ``run_python``'s ``trajectories``, in the workbench's base-sim
    predictions and in the sandbox's data file. The second round
    continues the conversation the first opened (by the id the session
    manager recorded) with a short message instead of a fresh context.
    The loop records each round, syncs the data, checkpoints, and ends
    the run as the agent asked.
    """
    _config(tmp_path)
    env, approach = _make_approach()
    queries: List[Dict[str, Any]] = []

    def fake_query(message: str, **kwargs: Any) -> List[Dict[str, Any]]:
        queries.append({"message": message, "kind": kwargs.get("kind")})
        n = len(queries)
        zero = [0.0] * env.action_space.shape[0]
        ctx = approach._tool_context  # pylint: disable=protected-access
        mgr = approach._agent_session  # pylint: disable=protected-access
        names = [t.name for t in ctx.extra_mcp_tools]
        # The model workbench is live: run_python plus the sim probe; no
        # tool ends a round or resets the context.
        assert "run_python" in names and "handoff" not in names
        assert ctx.probe_option_model_provider is not None
        assert ctx.probe_fit_provider is not None
        if n == 1:
            assert "first conversation round of the run" in message
            assert "No model yet" in message
            assert "[context] size not reported yet" in message
            assert mgr.resume_session_id is None
            obs = _call(approach, "env_observe")
            assert "[episode] NOT_FINISHED" in obs and "[render]" not in obs
            assert "[context]" in obs
            assert "PickJug" in _call(approach, "skills_list")
            probe = ("print(len(trajectories), "
                     "[len(t.actions) for t in trajectories])")
            assert _call(approach, "run_python", code=probe).startswith("0 []")
            for i in range(3):
                out = _call(approach, "env_step", action=zero)
                assert "step applied" in out
                # The episode in progress, as it stands, everywhere the
                # agent can look.
                assert _call(approach, "run_python",
                             code=probe).startswith(f"1 [{i + 1}]")
                bench = approach._workbench  # pylint: disable=protected-access
                assert len(bench.obs_triples) == i + 1
                assert len(bench.base_pred_triples) == i + 1
                assert bench.env is not None
                with open(trajectories_path(ctx.sandbox_dir), "rb") as f:
                    on_disk = pickle.load(f)
                assert [len(t["actions"]) for t in on_disk] == [i + 1]
            assert ctx.current_observation is not None
            # Helpers the agent keeps in ./probe_ext.py are loaded into
            # run_python at the next round's start (they are not there
            # yet in this one).
            assert "probe_ext.py" not in message
            with open(os.path.join(ctx.sandbox_dir, "probe_ext.py"),
                      "w",
                      encoding="utf-8") as f:
                f.write("def n_recorded():\n"
                        "    return len(trajectories)\n"
                        "def helper_ok():\n"
                        "    return 'ext-ok'\n")
            # What the session manager records when the CLI opens the
            # conversation; the next round continues it.
            info = os.path.join(approach._get_log_dir(), "session_info.json")  # pylint: disable=protected-access
            with open(info, "w", encoding="utf-8") as f:
                json.dump({"session_id": "conv-1"}, f)
        else:
            assert "you stopped" in message and "not settled" in message
            assert "## Skills" not in message
            assert mgr.resume_session_id == "conv-1"
            assert "`./probe_ext.py` loaded into `run_python` " \
                "(helper_ok, n_recorded)" in message
            assert _call(
                approach,
                "run_python",
                code="print(helper_ok(), n_recorded())").startswith("ext-ok 1")
            assert "Give-up recorded" in _call(approach,
                                               "give_up",
                                               note="enough")
        return _result()

    approach._query_agent_sync = fake_query  # type: ignore[method-assign]  # pylint: disable=protected-access
    approach.prepare_for_continual(Dataset([]))
    run = ContinualRun(env, approach, create_level_player(env, approach))
    card = run.run()

    assert card.end_reason == "agent_ended" and card.end_note == "enough"
    assert [q["kind"] for q in queries] == ["play", "play"]
    # Full and continuation prompts retain one authoritative budget block.
    for query in queries:
        message = query["message"]
        assert message.count("[ledger]") == 1
        assert message.count("[context]") == 1
        assert "[level]" in message and "[episode]" in message
    assert "[goal]" not in queries[0]["message"]
    assert "Goal atoms:" in queries[0]["message"]
    # The workbench is torn down between rounds and at the end; its data
    # (predicted once per transition) stays for the run.
    ctx = approach._tool_context  # pylint: disable=protected-access
    assert ctx.probe_option_model_provider is None
    bench = approach._workbench  # pylint: disable=protected-access
    assert bench.env is None and len(bench.base_pred_triples) == 3
    lv = card.levels[0]
    assert lv.steps == 3 and lv.resets == 0 and not lv.won
    assert lv.sandbox["rounds"] == 2
    # The agent wrote no simulator, so nothing was fit or deployed.
    assert "fits" not in lv.sandbox
    assert approach._current_simulator_version is None  # pylint: disable=protected-access
    trajs = approach._online_trajectories  # pylint: disable=protected-access
    assert len(trajs) == 1 and len(trajs[0].actions) == 3
    assert trajs[0].train_task_idx == 0
    # The agent directory is the run directory's agent/.
    log_dir = approach._get_log_dir()  # pylint: disable=protected-access
    assert log_dir.startswith(os.path.join(str(tmp_path), "runs"))
    assert log_dir.endswith("agent")
    attempts = open(os.path.join(log_dir, "sandbox", "attempts.md"),
                    encoding="utf-8").read()
    assert "### Round 1" in attempts and "the agent stopped" in attempts
    assert "### Round 2" in attempts and "context size not reported" in attempts
    saved = [
        f for f in os.listdir(os.path.join(str(tmp_path), "saved"))
        if f.endswith(".AgentContinual")
    ]
    assert saved, "the approach checkpointed"
    # A broken or escaping extension is reported, never fatal, and
    # leaves the namespace as it was.
    sandbox = ctx.sandbox_dir
    ns: Dict[str, Any] = {"trajectories": []}
    with open(os.path.join(sandbox, "probe_ext.py"), "w",
              encoding="utf-8") as f:
        f.write("def fine():\n    return 1\nraise ValueError('boom')\n")
    approach._load_probe_extension(ns, sandbox)  # pylint: disable=protected-access
    status = approach._probe_ext_status  # pylint: disable=protected-access
    assert "failed to load (ValueError: boom)" in status
    with open(os.path.join(sandbox, "probe_ext.py"), "w",
              encoding="utf-8") as f:
        f.write("open('/etc/passwd').read()\n")
    approach._load_probe_extension(ns, sandbox)  # pylint: disable=protected-access
    status = approach._probe_ext_status  # pylint: disable=protected-access
    assert "NOT loaded: the sandbox guard blocked it" in status
    os.remove(os.path.join(sandbox, "probe_ext.py"))
    approach._load_probe_extension(ns, sandbox)  # pylint: disable=protected-access
    assert approach._probe_ext_status == ""  # pylint: disable=protected-access


def test_play_loop_stops_at_a_lost_test_level(tmp_path: Any) -> None:
    """On a test level (no resets) a GAME_OVER loses the level: the loop ends
    the level's rounds and the run ends as ``level_lost``."""
    _config(tmp_path,
            continual_levels="test_only",
            continual_episode_horizon=2)
    env, approach = _make_approach()
    queries: List[str] = []

    def fake_query(message: str, **kwargs: Any) -> List[Dict[str, Any]]:
        del kwargs
        queries.append(message)
        zero = [0.0] * env.action_space.shape[0]
        assert "(test task 0, no resets)" in message
        assert "step applied" in _call(approach, "env_step", action=zero)
        out = _call(approach, "env_step", action=zero)
        assert "GAME_OVER" in out and "lost" in out
        refused = _call(approach, "env_reset", note="again")
        assert refused.startswith("ERROR") and "lost" in refused
        return _result()

    approach._query_agent_sync = fake_query  # type: ignore[method-assign]  # pylint: disable=protected-access
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert len(queries) == 1
    assert card.end_reason == "level_lost"
    lv = card.levels[0]
    assert lv.split == "test" and lv.lost and not lv.won
    assert lv.steps == 2 and lv.resets == 0 and lv.game_overs == ["horizon"]


@pytest.mark.slow
def test_resume_reads_the_session_id_and_idle_guard(tmp_path: Any) -> None:
    """A checkpointed in-flight round resumes the conversation as a preemption
    resume, later rounds continue the same conversation, and rounds that never
    act trip the idle guard."""
    _config(tmp_path)
    env, approach = _make_approach()
    log_dir = approach._get_log_dir()  # pylint: disable=protected-access
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "session_info.json"),
              "w",
              encoding="utf-8") as f:
        json.dump({"session_id": "old-session"}, f)
    approach._round_in_flight = True  # pylint: disable=protected-access
    resumed: List[Any] = []

    def fake_query(message: str, **kwargs: Any) -> List[Dict[str, Any]]:
        del kwargs
        mgr = approach._agent_session  # pylint: disable=protected-access
        resumed.append((mgr.resume_session_id, "preemption" in message))
        return _result()

    approach._query_agent_sync = fake_query  # type: ignore[method-assign]  # pylint: disable=protected-access
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert card.end_reason == "agent_ended" and "stalled" in card.end_note
    # The first round resumed the interrupted turn; the later ones
    # continued the same conversation as ordinary rounds.
    assert resumed[0] == ("old-session", True)
    assert all(r == ("old-session", False) for r in resumed[1:])
    assert len(resumed) == 3


class _Killed(BaseException):
    """Stands in for the kill a Slurm preemption delivers mid-round."""


@pytest.mark.slow
def test_resume_rebuilds_the_workbench_from_the_recording(
        tmp_path: Any) -> None:
    """A resumed run reads the level's finished episodes back from the
    recording; the workbench predicts them with the base sim exactly as it does
    the live ones, because the recorded states keep the robot's joint data (the
    2026-09-05 busyboard resume crashed here on plain states)."""
    _config(tmp_path, continual_episode_horizon=2)
    env, approach = _make_approach()

    def killed_query(message: str, **kwargs: Any) -> List[Dict[str, Any]]:
        del message, kwargs
        zero = [0.0] * env.action_space.shape[0]
        _call(approach, "env_step", action=zero)
        assert "GAME_OVER" in _call(approach, "env_step", action=zero)
        _call(approach, "env_reset", note="again")
        _call(approach, "env_step", action=zero)
        raise _Killed()

    approach._query_agent_sync = killed_query  # type: ignore[method-assign]  # pylint: disable=protected-access
    approach.prepare_for_continual(Dataset([]))
    with pytest.raises(_Killed):
        ContinualRun(env, approach, create_level_player(env, approach)).run()

    _config(tmp_path, continual_episode_horizon=2, auto_resume=True)
    env2, approach2 = _make_approach()
    seen: Dict[str, Any] = {}

    def resumed_query(message: str, **kwargs: Any) -> List[Dict[str, Any]]:
        del message, kwargs
        bench = approach2._workbench  # pylint: disable=protected-access
        seen["counts"] = (len(bench.trajectories), len(bench.obs_triples),
                          len(bench.base_pred_triples))
        seen["kinds"] = sorted(
            {type(s).__name__
             for s, _, _ in bench.obs_triples})
        seen["out"] = _call(approach2,
                            "run_python",
                            code="print(len(trajectories), "
                            "[len(t.actions) for t in trajectories])")
        _call(approach2, "give_up", note="done")
        return _result()

    approach2._query_agent_sync = resumed_query  # type: ignore[method-assign]  # pylint: disable=protected-access
    approach2.prepare_for_continual(Dataset([]))
    card = ContinualRun(env2, approach2, create_level_player(env2,
                                                             approach2)).run()
    lv = card.levels[0]
    assert lv.resumes == 1 and lv.preemptions == 1 and lv.harness_resets == 0
    assert card.end_reason == "agent_ended"
    # The finished episode (2 steps, from the pickle) and the one in
    # progress (1 step, replayed live), all predicted.
    assert seen["counts"] == (2, 3, 3)
    assert seen["kinds"] == ["PyBulletState"]
    assert seen["out"].startswith("2 [2, 1]")


@pytest.mark.slow
def test_model_gate_on_a_test_level(tmp_path: Any) -> None:
    """Under ``continual_require_model_on_test`` the skill tools refuse on a
    test level, charging nothing, until the sandbox's simulator.py exists and
    declares RESIDUAL_FEATURES; then they run, fitted or not.

    env_step is not gated.
    """
    # pylint: disable=protected-access
    _config(tmp_path,
            continual_levels="test_only",
            continual_require_model_on_test=True)
    env, approach = _make_approach()
    assert "Test levels require a fitted model" in \
        approach._play_system_prompt()
    counter = '''
class Counter(BaseSimulator):
    AGENT_PARAM_SPECS = [ParamSpec("rate", .25, lo=0.0, hi=1.0)]
    MODEL_STATE_INIT = {"charge": 0.0}
%s
    @classmethod
    def update_model_state(cls, observation, model_state, params, action):
        model_state["charge"] += params["rate"]

RESIDUAL_ENV = Counter
'''

    def fake_query(message: str, **kwargs: Any) -> List[Dict[str, Any]]:
        del kwargs
        assert "(test task 0, no resets)" in message
        assert "call `sim.fit()` before you act on a test level" in message
        zero = [0.0] * env.action_space.shape[0]
        assert "step applied" in _call(approach, "env_step", action=zero)
        refused = _call(approach, "skills_invoke", skill="x")
        assert refused.startswith("ERROR")
        assert "Write `./simulator.py`" in refused
        assert "Nothing was charged" in refused
        path = os.path.join(approach._tool_context.sandbox_dir, "simulator.py")
        with open(path, "w", encoding="utf-8") as file:
            file.write(counter % "")
        refused = _call(approach, "skills_execute_plan", plan="x")
        assert refused.startswith("ERROR")
        assert "declares no RESIDUAL_FEATURES" in refused
        with open(path, "w", encoding="utf-8") as file:
            file.write(counter % "    RESIDUAL_FEATURES = {}")
        out = _call(approach, "skills_invoke", skill="x")
        assert "Could not parse" in out and "Nothing was charged" not in out
        with open(path, "w", encoding="utf-8") as file:
            file.write("import nonexistent_module_xyz\n" + counter % "")
        refused = _call(approach, "skills_invoke", skill="x")
        assert refused.startswith("ERROR") and "does not load" in refused
        with open(path, "w", encoding="utf-8") as file:
            file.write(counter % "    RESIDUAL_FEATURES = {}")
        assert "Could not parse" in _call(approach, "skills_invoke", skill="x")
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    approach._query_agent_sync = fake_query  # type: ignore[method-assign]
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert card.levels[0].steps == 1 and card.levels[0].split == "test"
    assert approach._tool_context.skill_gate is None


@pytest.mark.slow
def test_skill_preflight_rehearses_in_the_candidate(tmp_path: Any) -> None:
    """Once a simulator.py exists, every skill request is rehearsed in it
    first: a Place with nothing held is refused, charging nothing, with the
    controller's reason; ``force=true`` runs it anyway; a feasible PickJug
    passes the rehearsal and runs. Before the file exists ``sim`` still rolls
    the real skill controllers on the base physics for the agent, but no
    request is rehearsed there: the same Place runs, and is charged."""
    # pylint: disable=protected-access
    _config(tmp_path, continual_levels="train_only")
    env, approach = _make_approach()
    assert "Every skill request is rehearsed first" in \
        approach._play_system_prompt()
    seen: Dict[str, Any] = {}
    counter = '''
class Counter(BaseSimulator):
    AGENT_PARAM_SPECS = [ParamSpec("rate", .25, lo=0.0, hi=1.0)]
    MODEL_STATE_INIT = {"charge": 0.0}
    RESIDUAL_FEATURES = {}

    @classmethod
    def update_model_state(cls, observation, model_state, params, action):
        model_state["charge"] += params["rate"]

RESIDUAL_ENV = Counter
'''

    def fake_query(message: str, **kwargs: Any) -> List[Dict[str, Any]]:
        del kwargs
        assert "runs the real skill controllers" in message
        out = _call(approach,
                    "run_python",
                    code="r = sim.reset(task_idx=0).run("
                    "'PickJug(robot:robot, jug0:jug)[0.05]', render=False)\n"
                    "print('steps', len(r.steps), 'fail', "
                    "r.steps[0]['failure'])")
        assert "steps 1 fail None" in out, out
        place = "Place(robot:robot)[1.1, 1.6, 0.6, 0.0]"
        unrehearsed = _call(approach, "skills_invoke", skill=place)
        assert "Rehearsed in `sim`" not in unrehearsed, unrehearsed
        assert "Nothing was charged" not in unrehearsed
        path = os.path.join(approach._tool_context.sandbox_dir, "simulator.py")
        with open(path, "w", encoding="utf-8") as file:
            file.write(counter)
        refused = _call(approach, "skills_invoke", skill=place)
        assert refused.startswith("ERROR"), refused
        assert "Rehearsed in `sim` (" in refused
        assert "no model yet" not in refused
        assert "got stuck" in refused
        assert "Nothing was charged" in refused and "force=true" in refused
        seen["refused"] = refused
        forced = _call(approach, "skills_invoke", skill=place, force=True)
        assert "Nothing was charged" not in forced
        out = _call(approach,
                    "skills_invoke",
                    skill="PickJug(robot:robot, jug0:jug)[0.05]")
        assert "Nothing was charged" not in out
        assert "succeeded" in out, out
        assert "Give-up recorded" in _call(approach, "give_up", note="done")
        return _result()

    approach._query_agent_sync = fake_query  # type: ignore[method-assign]
    approach.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, approach, create_level_player(env,
                                                           approach)).run()
    assert "refused" in seen
    assert card.levels[0].steps > 0
    assert approach._tool_context.skill_preflight is None
