"""Continual comparison contracts exercised through real play tools."""
import os
import re
import shlex
import sys
from typing import Any, Callable, Dict, Iterator, List, Set, Tuple

import pybullet as p
import pytest

from predicators import utils
from predicators.agent_sdk.belief_probe import BeliefProbe
from predicators.approaches import create_approach
from predicators.approaches.agent_continual_ablation_approach import \
    AgentContinualNoUncertaintyApproach
from predicators.envs import create_new_env
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.run.level_players import create_level_player
from predicators.settings import CFG
from predicators.structs import Dataset
from scripts.cluster_utils import config_to_cmd_flags, generate_run_configs
from tests.approaches.test_agent_continual_approach import _call, _config, \
    _result

# The benchmark sweep: eight arms on the five benchmark settings, three
# seeds each (Sept 18, 2026).
CONFIG = "predicatorv3/continual_eight_agent_noisy_sweep.yaml"
ARM_COUNT = 8
SEEDS = {0, 1, 2}


@pytest.fixture(autouse=True)
def _dispose_test_physics_clients(monkeypatch: Any) -> Iterator[None]:
    """Release worlds created by a case, including evicted skill simulators."""
    # pylint: disable=import-outside-toplevel,protected-access
    from predicators import envs
    from predicators.ground_truth_models.skill_factories.base import \
        clear_shared_simulator_cache
    owned: Set[int] = set()
    original = p.connect

    def connect(*args: Any, **kwargs: Any) -> int:
        client = original(*args, **kwargs)
        if client >= 0:
            owned.add(client)
        return client

    monkeypatch.setattr(p, "connect", connect)
    try:
        yield
    finally:
        clear_shared_simulator_cache()
        for name, env in list(envs._MOST_RECENT_ENV_INSTANCE.items()):
            if getattr(env, "_physics_client_id", None) in owned:
                del envs._MOST_RECENT_ENV_INSTANCE[name]
        for client in owned:
            if p.isConnected(client):
                p.disconnect(client)
        assert all(not p.isConnected(client) for client in owned)


def test_comparison_config_matches_existing_domains(monkeypatch: Any) -> None:
    """Eight arms share each domain's settings and run paired seeds."""
    new = list(generate_run_configs(CONFIG, False))
    approaches = {c.approach for c in new}
    assert len(approaches) == ARM_COUNT
    assert len(new) == ARM_COUNT * 5 * len(SEEDS)
    seeds: Dict[Tuple[str, str], Set[int]] = {}
    for cfg in new:
        monkeypatch.setattr(
            sys, "argv",
            ["predicators/main.py", *shlex.split(config_to_cmd_flags(cfg))])
        parsed = utils.parse_args()
        assert parsed["env"] == cfg.env
        assert parsed["approach"] == cfg.approach
        no_uncertainty = cfg.approach == "agent_continual_no_uncertainty"
        if no_uncertainty:
            assert not cfg.flags["continual_belief_frame"]
            assert not cfg.flags["code_sim_learning_rollout_noise_filter"]
            # Same noise as every arm, but not declared to this one.
            assert not cfg.flags["continual_obs_noise_declared"]
        seeds.setdefault((cfg.env, cfg.approach), set()).add(parsed["seed"])
        # The EMPIRIC arm is the reference for the domain settings.
        reference = next(
            c for c in new
            if c.env == cfg.env and c.approach == "agent_continual")
        keys = [
            k for k in reference.flags
            if k.startswith(("continual_obs_noise_", "balloons_", "boil_",
                             "bridge_", "domino_", "fan_"))
        ]
        keys += [
            "num_train_tasks", "num_test_tasks", "continual_steps_per_level",
            "continual_levels"
        ]
        for key in keys:
            if no_uncertainty and key == "continual_obs_noise_declared":
                continue
            assert cfg.flags[key] == reference.flags[key], (cfg.env, key)
    assert all(s == SEEDS for s in seeds.values())


@pytest.mark.parametrize("flag", [
    "continual_belief_frame", "code_sim_learning_rollout_noise_filter",
    "continual_obs_noise_declared"
])
def test_no_uncertainty_rejects_smoothing(flag: str) -> None:
    """A config override cannot silently restore denoising, or the noise
    declaration, in this arm."""
    cfg = next(c for c in generate_run_configs(CONFIG, False)
               if c.approach == "agent_continual_no_uncertainty")
    utils.reset_config({
        **{k: v
           for k, v in cfg.flags.items() if k != "log"}, flag: True
    })
    # Contract validation must fail before constructing an environment.
    with pytest.raises(ValueError, match=flag):
        AgentContinualNoUncertaintyApproach()


@pytest.mark.parametrize("domain",
                         ["boil", "bridge", "fan", "domino", "balloons"])
@pytest.mark.parametrize(
    "arm", ["no_fitting", "no_uncertainty", "scene_only", "oracle_dynamics"])
def test_ablation_play_tools(tmp_path: Any, monkeypatch: Any, arm: str,
                             domain: str) -> None:
    """No-uncertainty uses raw observations; tools enforce arm restrictions."""
    cfg = next(c for c in generate_run_configs(CONFIG, False)
               if c.env == f"pybullet_{domain}" and c.approach.endswith(arm))
    _config(
        tmp_path, **{
            **{k: v
               for k, v in cfg.flags.items() if k != "log"}, "continual_render":
            False,
            "continual_make_video": False,
            "continual_runs_dir": str(tmp_path / "runs"),
            "env": cfg.env,
            "approach": cfg.approach
        })
    env = create_new_env(cfg.env, do_cache=False, use_gui=False)
    agent: Any = create_approach(cfg.approach, env.predicates,
                                 get_gt_options(env.get_name()), env.types,
                                 env.action_space,
                                 [t.task for t in env.get_train_tasks()])

    def query(message: str, *_args: Any, **_kwargs: Any) -> Any:
        ctx = agent._tool_context  # pylint: disable=protected-access
        # The first query's model status matches what the arm can do.
        if arm in {"scene_only", "oracle_dynamics"}:
            assert ("the supplied simulator, fixed for the run and not "
                    "exposed as source") in message
            assert "sim.fit()" not in message
        elif arm == "no_fitting":
            assert "the harness fits nothing" in message
            assert "sim.fit()" not in message
        else:
            assert "call `sim.fit()`" in message
        observation = _call(agent, "env_observe")
        # The no-uncertainty arm is not told about the noise at all.
        assert ("[noise]" in observation) is (arm != "no_uncertainty")
        assert "[objects]" in observation
        prompt = agent._play_system_prompt()  # pylint: disable=protected-access
        tool_description = next(t.description for t in ctx.extra_mcp_tools
                                if t.name == "run_python")
        if arm == "no_uncertainty":
            assert "[belief]" not in observation
            assert "[atoms under the belief]" not in observation
            assert "No explicit uncertainty handling" in prompt
            session = agent._play_session  # pylint: disable=protected-access
            obs = session.observe()
            assert obs.belief is None
            assert "do not average, smooth, or filter" in prompt
            for word in ("Observation noise", "sigma", "noisy", "denoise"):
                assert word not in prompt, word
            probe = BeliefProbe(ctx)
            with pytest.raises(ValueError, match="Explicit uncertainty"):
                probe.belief()
            with pytest.raises(ValueError, match="Disagreement"):
                probe.suggest_probes("")
            with pytest.raises(ValueError, match="Explicit uncertainty"):
                probe.run("", physics_sweep=True)
            with pytest.raises(ValueError, match="Explicit uncertainty"):
                probe.run("", belief_draws=2)
            # Numerical fitting remains installed even before any data.
            assert ctx.probe_fit_provider is not None
        elif arm in {"scene_only", "oracle_dynamics"}:
            heading = ("Scene-only comparison" if arm == "scene_only" else
                       "Oracle dynamics comparison")
            assert heading in prompt
            assert "not exposed as source" in prompt
            assert "unavailable" in BeliefProbe(ctx).fit()
            # The supplied model never enters the sandbox, and nothing the
            # agent can read names its calibration constants.
            assert ctx.sandbox_dir is not None
            assert not os.path.exists(
                os.path.join(ctx.sandbox_dir, "simulator.py"))
            paths = agent._resolve_synthesis_paths()  # pylint: disable=protected-access
            assert os.path.isfile(paths.simulator_file)
            assert not paths.simulator_file.startswith(ctx.sandbox_dir)
            # Parameter names appear nowhere the agent reads; the values
            # appear in no probe report and no sandbox file name (the
            # query and observation carry unrelated numbers such as skill
            # ranges, so values are not checked there).
            reports = [
                BeliefProbe(ctx).validate(),
                BeliefProbe(ctx).residuals(),
                _sandbox_listing(ctx)
            ]
            for text in [prompt, message, observation, tool_description
                         ] + reports:
                leaked = _leaked_calibration(domain, text, values=False)
                assert not leaked, (arm, leaked, text[:400])
            for text in reports:
                leaked = _leaked_calibration(domain, text, values=True)
                assert not leaked, (arm, leaked, text[:400])
            with open(paths.simulator_file, encoding="utf-8") as f:
                assert re.search(r"AGENT_PARAM_SPECS(: [^=]+)? = \[\]",
                                 f.read())
        else:
            assert "No harness parameter fitting" in prompt
            assert "Do not implement an optimizer" not in prompt
            assert "parameter estimation is disabled" in BeliefProbe(ctx).fit()
        # The run_python description offers only what the arm's probe
        # accepts.
        if arm in {"scene_only", "oracle_dynamics", "no_fitting"}:
            assert "sim.fit" not in tool_description
            assert "PARAMS UNFITTED" not in tool_description
        else:
            assert "sim.fit(" in tool_description
        if arm in {"scene_only", "oracle_dynamics"}:
            assert "no `simulator.py` to read or write" in tool_description
            assert "phys_params" not in tool_description
        else:
            assert "write `simulator.py`" in tool_description
        if arm == "no_uncertainty":
            assert "belief_draws" not in tool_description
            assert "suggest_probes" not in tool_description
            assert "physics_sweep=True" not in tool_description
        else:
            assert "belief_draws" in tool_description
        assert "step applied" in _call(agent,
                                       "env_step",
                                       action=[0.0] *
                                       env.action_space.shape[0])
        assert "Give-up recorded" in _call(agent, "give_up", note="audit")
        return _result()

    monkeypatch.setattr(agent, "_query_agent_sync", query)
    agent.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, agent, create_level_player(env, agent)).run()
    assert card.total_steps == 1
    assert card.total_resets == 0


def _leaked_calibration(domain: str, text: str, *, values: bool) -> List[str]:
    """Parameter names (and, with ``values``, privileged constants) of the
    supplied model in ``text``: a value counts only as a whole number, so a
    coordinate such as 0.5079 does not match the friction 0.5."""
    names: List[str] = []
    constants: List[float] = []
    if domain == "domino":
        names = ["lateral_friction"]
        constants = [float(CFG.domino_true_friction)]
    elif domain == "balloons":
        names = ["air_drag", "mass_oak", "lift_gold", "fade_height"]
        constants = [float(CFG.balloons_drag)] + [
            float(m) for m in CFG.balloons_box_masses
        ] + [float(l) for l in CFG.balloons_lifts]
    leaked = [n for n in names if n in text]
    for value in (constants if values else []):
        if re.search(rf"(?<![\d.]){re.escape(repr(value))}(?![\d])", text):
            leaked.append(repr(value))
    return leaked


def _sandbox_listing(ctx: Any) -> str:
    """Every file name in the sandbox, so a stray copy of the model shows."""
    names: List[str] = []
    for root, _dirs, files in os.walk(ctx.sandbox_dir):
        names.extend(os.path.join(root, f) for f in files)
    return "\n".join(sorted(names))


def _configure_domain_comparison(tmp_path: Any, domain: str,
                                 approach: str) -> str:
    """Use the benchmark's real domain settings with local test outputs."""
    cfg = next(c for c in generate_run_configs(CONFIG, False)
               if c.env == f"pybullet_{domain}" and c.approach == approach)
    _config(
        tmp_path, **{
            **{k: v
               for k, v in cfg.flags.items() if k != "log"}, "continual_render":
            False,
            "continual_make_video": False,
            "continual_runs_dir": str(tmp_path / "runs"),
            "env": cfg.env,
            "approach": cfg.approach
        })
    return cfg.env


@pytest.mark.parametrize("domain",
                         ["boil", "bridge", "fan", "domino", "balloons"])
@pytest.mark.parametrize("backend", ["program", "pybullet"])
def test_standalone_model_is_live_without_engine(tmp_path: Any,
                                                 monkeypatch: Any, domain: str,
                                                 backend: str) -> None:
    """Agent-owned predictions work without a supplied physical scene."""
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    from predicators.code_sim_learning.program_world_model import \
        ProgramOptionModel  # pylint: disable=import-outside-toplevel
    env_name = _configure_domain_comparison(
        tmp_path, domain, "agent_continual_program_world_model")
    env = create_new_env(env_name, do_cache=False, use_gui=False)
    options = get_gt_options(env.get_name())
    agent: Any = create_approach("agent_continual_program_world_model",
                                 env.predicates, options, env.types,
                                 env.action_space,
                                 [t.task for t in env.get_train_tasks()])
    calls = []

    def query(message: str, *_args: Any, **_kwargs: Any) -> Any:
        assert "No world model yet" in message
        ctx = agent._tool_context  # pylint: disable=protected-access
        refs = agent._get_sandbox_reference_files()  # pylint: disable=protected-access
        assert not any(k.startswith("base_sim/") for k in refs)
        path = Path(agent._resolve_synthesis_paths().base) / "world_model.py"  # pylint: disable=protected-access
        code = ('LATENT_FEATURES = {}\n'
                'def initial_latent(obs, rng):\n    return {}\n'
                'def transition(obs, latent, option, rng):\n'
                '    return obs.copy(), dict(latent), COUNT\n')
        if backend == "pybullet":
            code = '''import pybullet as physics
LATENT_FEATURES = {}
def initial_latent(obs, rng):
    return {}
def transition(obs, latent, option, rng):
    client = physics.connect(physics.DIRECT)
    try:
        physics.setGravity(0, 0, -10, physicsClientId=client)
        physics.setTimeStep(0.1, physicsClientId=client)
        body = physics.createMultiBody(baseMass=1, basePosition=[0, 0, 1],
                                       physicsClientId=client)
        for _ in range(COUNT):
            physics.stepSimulation(physicsClientId=client)
        position, _ = physics.getBasePositionAndOrientation(
            body, physicsClientId=client)
        return obs.copy(), {"fall_height": position[2]}, COUNT
    finally:
        physics.disconnect(client)
'''
        prompt = agent._play_system_prompt()  # pylint: disable=protected-access
        assert "You may use PyBullet" in prompt
        assert "Do not import an environment or a physics engine" not in prompt
        path.parent.mkdir(parents=True, exist_ok=True)
        for count in (2, 3):
            path.write_text(code.replace("COUNT", str(count)),
                            encoding="utf-8")
            model = ctx.probe_option_model_provider()
            assert isinstance(model, ProgramOptionModel)
            assert getattr(model, "sim_env", None) is None
            calls.append(model)
        assert calls[0] is not calls[1]
        probe = BeliefProbe(ctx).reset(current=True)
        wait = next(option for option in options if option.name == "Wait")
        current = ctx.current_observation
        objects = [
            next(obj for obj in current if obj.is_instance(t))
            for t in wait.types
        ]
        import numpy as np  # pylint: disable=import-outside-toplevel
        params = np.zeros(wait.params_space.shape, dtype=np.float32)
        grounded = wait.ground(objects, params)
        assert isinstance(env, PyBulletEnv)
        client = env._physics_client_id  # pylint: disable=protected-access

        def live_bodies() -> Any:
            return [(p.getBasePositionAndOrientation(body,
                                                     physicsClientId=client),
                     p.getBaseVelocity(body, physicsClientId=client))
                    for body in (p.getBodyUniqueId(i, physicsClientId=client)
                                 for i in range(p.getNumBodies(client)))]

        before = live_bodies()
        predicted, steps = calls[-1].get_next_state_and_num_actions(
            current, grounded)
        assert steps == 3
        if backend == "pybullet":
            assert predicted.latent is not None
            assert 0 < predicted.latent["fall_height"] < 1
        plan = ("Wait(" + ", ".join(str(obj) for obj in objects) + ")[" +
                ", ".join(str(float(value)) for value in params) + "]")
        result = probe.run(plan, render=False)
        assert result is not None
        assert live_bodies() == before
        # Close to WorldCoder: score and one rollout at a time; plan
        # search, repeated trials, predicate scoring and engine renders
        # are refused, and the descriptions do not offer them.
        assert probe.run(plan) is not None  # text-only rollout
        refusals: List[Callable[[], Any]] = [
            lambda: probe.refine(plan), lambda: probe.run(plan, trials=2),
            probe.predicates, lambda: probe.render("x"),
            lambda: probe.suggest_probes(plan), probe.belief
        ]
        for refused in refusals:
            with pytest.raises(RuntimeError, match="unavailable"):
                refused()
        tools = {t.name: t for t in ctx.extra_mcp_tools}
        desc = tools["run_python"].description
        for absent in ("sim.refine", "trials=", "sim.predicates", "sim.render",
                       "evaluate_trajectory"):
            assert absent not in desc, absent
        assert "sim.score" in desc and "sim.run(plan_text)" in desc
        assert "sim.refine" not in prompt
        assert "first rehearses" not in prompt
        for kwargs in ({
                "solved": True
        }, {
                "contacts": True
        }, {
                "physics_sweep": True
        }):
            with pytest.raises(ValueError, match="Engine diagnostics"):
                probe.run(plan, **kwargs)

        assert ctx.probe_fit_provider is None
        assert ctx.probe_score_provider is not None
        assert "step applied" in _call(agent,
                                       "env_step",
                                       action=[0.0] *
                                       env.action_space.shape[0])
        assert len(agent._program_trajectories[-1].actions) == 1  # pylint: disable=protected-access
        assert "Give-up recorded" in _call(agent, "give_up", note="audit")
        return _result()

    monkeypatch.setattr(agent, "_query_agent_sync", query)
    agent.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, agent, create_level_player(env, agent)).run()
    assert card.total_steps == 1
    assert isinstance(agent._option_model, ProgramOptionModel)  # pylint: disable=protected-access


@pytest.mark.parametrize("domain",
                         ["boil", "bridge", "fan", "domino", "balloons"])
def test_zero_shot_seals_before_first_charge(tmp_path: Any, monkeypatch: Any,
                                             domain: str) -> None:
    """Missing models and later edits cannot take steps, even after resume."""
    from pathlib import Path  # pylint: disable=import-outside-toplevel
    env_name = _configure_domain_comparison(tmp_path, domain,
                                            "agent_continual_zero_shot")
    env = create_new_env(env_name, do_cache=False, use_gui=False)
    agent: Any = create_approach("agent_continual_zero_shot", env.predicates,
                                 get_gt_options(env.get_name()), env.types,
                                 env.action_space,
                                 [t.task for t in env.get_train_tasks()])
    code = ('class Model(BaseSimulator):\n'
            '    AGENT_PARAM_SPECS = []\n'
            '    RESIDUAL_FEATURES = {}\n'
            '    def _domain_specific_step(self):\n        pass\n'
            'RESIDUAL_ENV = Model\n')

    def query(message: str, *_args: Any, **_kwargs: Any) -> Any:
        assert "the dynamics are sealed at that point" in message
        ctx = agent._tool_context  # pylint: disable=protected-access
        description = next(t.description for t in ctx.extra_mcp_tools
                           if t.name == "run_python")
        assert "seals it" in description
        assert "sim.fit" not in description
        action = [0.0] * env.action_space.shape[0]
        result = _call(agent, "env_step", action=action)
        assert "step applied" not in result
        assert "No ./simulator.py yet" in result
        path = Path(agent._resolve_synthesis_paths().simulator_file)  # pylint: disable=protected-access
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        assert "step applied" in _call(agent, "env_step", action=action)
        saved = agent._extra_save_state()  # pylint: disable=protected-access
        assert saved["frozen_model_source"] == code
        ctx = agent._tool_context  # pylint: disable=protected-access
        assert "unavailable" in BeliefProbe(ctx).validate(params={"test": 2.0})
        path.write_text(code + "# edited after action\n", encoding="utf-8")
        assert "Dynamics are frozen" in _call(agent, "env_step", action=action)
        ctx = agent._tool_context  # pylint: disable=protected-access
        assert "unavailable" in BeliefProbe(ctx).fit()
        agent._load_extra_save_state(saved)  # pylint: disable=protected-access
        assert path.read_text(encoding="utf-8") == code
        assert "step applied" in _call(agent, "env_step", action=action)
        assert "Give-up recorded" in _call(agent, "give_up", note="audit")
        return _result()

    monkeypatch.setattr(agent, "_query_agent_sync", query)
    agent.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, agent, create_level_player(env, agent)).run()
    assert card.total_steps == 2
    assert card.total_resets == 0


def test_program_current_memory_replays_after_edits_and_reset(
        tmp_path: Any, monkeypatch: Any) -> None:
    """Current-state rehearsals carry skill history using the latest model."""
    from pathlib import Path  # pylint: disable=import-outside-toplevel
    _config(tmp_path,
            approach="agent_continual_program_world_model",
            max_num_steps_option_rollout=2)
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    agent: Any = create_approach("agent_continual_program_world_model",
                                 env.predicates,
                                 get_gt_options(env.get_name()), env.types,
                                 env.action_space,
                                 [t.task for t in env.get_train_tasks()])

    def query(*_args: Any, **_kwargs: Any) -> Any:
        # pylint: disable=protected-access
        ctx = agent._tool_context
        path = Path(agent._resolve_synthesis_paths().base) / "world_model.py"
        code = ('LATENT_FEATURES = {}\n'
                'def initial_latent(obs, rng):\n    return {"count": 0}\n'
                'def transition(obs, latent, option, rng):\n'
                '    option.memory["audit_touched"] = True\n'
                '    return obs.copy(), {"count": latent["count"] + INC}, 2\n')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code.replace("INC", "1"), encoding="utf-8")
        robot = next(o for o in ctx.current_observation
                     if o.type.name == "robot")

        def check_memory(expected: int) -> None:
            probe = BeliefProbe(ctx).reset(current=True)
            assert probe._state is not None
            assert probe._state.latent == {"count": expected}
            episodes = agent._play_session.level_episodes()
            assert all("audit_touched" not in action.get_option().memory
                       for ep in episodes for action in ep["actions"]
                       if action.has_option())

        for expected in (1, 2):
            _call(agent, "skills_invoke", skill=f"Wait({robot})[]")
            check_memory(expected)
        path.write_text(code.replace("INC", "3"), encoding="utf-8")
        check_memory(6)
        saved = agent._extra_save_state()
        agent._load_extra_save_state(saved)
        check_memory(6)
        _call(agent, "env_reset", note="memory audit")
        check_memory(0)
        _call(agent, "env_step", action=[0.0] * env.action_space.shape[0])
        with pytest.raises(ValueError, match="primitive actions"):
            BeliefProbe(ctx).reset(current=True)
        _call(agent, "give_up", note="audit")
        return _result()

    monkeypatch.setattr(agent, "_query_agent_sync", query)
    agent.prepare_for_continual(Dataset([]))
    card = ContinualRun(env, agent, create_level_player(env, agent)).run()
    # Two two-step skills, one charged reset, and one primitive action.
    assert card.total_steps == 6
    assert card.total_resets == 1


def _program_resume_stage(directory: str, stage: int) -> None:
    """Drive one actual process of the standalone preemption audit."""
    # pylint: disable=import-outside-toplevel,protected-access
    from pathlib import Path

    from predicators.run.checkpoints import maybe_auto_resume
    from predicators.run.continual import run_continual
    from tests.approaches.test_agent_continual_approach import _Killed
    root = Path(directory)
    _config(root,
            approach="agent_continual_program_world_model",
            max_num_steps_option_rollout=2,
            auto_resume=stage == 2,
            continual_make_video=False)
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    agent: Any = create_approach("agent_continual_program_world_model",
                                 env.predicates,
                                 get_gt_options(env.get_name()), env.types,
                                 env.action_space,
                                 [t.task for t in env.get_train_tasks()])
    if stage == 2:
        maybe_auto_resume(agent)

    def query(*_args: Any, **_kwargs: Any) -> Any:
        ctx = agent._tool_context
        path = Path(agent._resolve_synthesis_paths().base) / "world_model.py"
        robot = next(o for o in ctx.current_observation
                     if o.type.name == "robot")
        if stage == 1:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                'LATENT_FEATURES = {}\n'
                'def initial_latent(obs, rng):\n'
                '    return {"count": 0}\n'
                'def transition(obs, latent, option, rng):\n'
                '    return obs.copy(), '
                '{"count": latent["count"] + 1}, 2\n',
                encoding="utf-8")
            for _ in range(2):
                _call(agent, "skills_invoke", skill=f"Wait({robot})[]")
            _call(agent, "env_reset", note="new episode before preemption")
            _call(agent, "skills_invoke", skill=f"Wait({robot})[]")
            agent.save(0)
            raise _Killed()
        assert path.is_file()
        probe = BeliefProbe(ctx).reset(current=True)
        assert probe._state is not None
        assert probe._state.latent == {"count": 1}
        episodes = agent._play_session.level_episodes()
        assert [len(ep["actions"]) for ep in episodes] == [4, 2]
        _call(agent, "skills_invoke", skill=f"Wait({robot})[]")
        probe = BeliefProbe(ctx).reset(current=True)
        assert probe._state is not None
        assert probe._state.latent == {"count": 2}
        path.write_text(path.read_text().replace('+ 1', '+ 3'),
                        encoding="utf-8")
        probe = BeliefProbe(ctx).reset(current=True)
        assert probe._state is not None
        assert probe._state.latent == {"count": 6}
        _call(agent, "give_up", note="resume audit complete")
        return _result()

    agent._query_agent_sync = query
    if stage == 1:
        with pytest.raises(_Killed):
            run_continual(env, agent)
    else:
        card = run_continual(env, agent)
        assert card.total_steps == 9 and card.total_resets == 1
        assert card.levels[0].resumes == 1
        assert card.levels[0].harness_resets == 0
        assert card.end_reason == "agent_ended"


def test_program_fresh_process_resume(tmp_path: Any) -> None:
    """Resume skill identity and memory in an independent interpreter."""
    import subprocess  # pylint: disable=import-outside-toplevel
    for stage in (1, 2):
        code = ('from tests.approaches.test_continual_comparison_approach '
                'import _program_resume_stage\n'
                f'_program_resume_stage({str(tmp_path)!r}, {stage})\n')
        result = subprocess.run([sys.executable, "-c", code],
                                check=False,
                                capture_output=True,
                                text=True,
                                timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
