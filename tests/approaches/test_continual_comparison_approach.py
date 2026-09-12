"""Continual comparison contracts exercised through real play tools."""
import shlex
import sys
from typing import Any

import pytest

from predicators import utils
from predicators.agent_sdk.belief_probe import BeliefProbe
from predicators.approaches import create_approach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.run.continual import ContinualRun
from predicators.run.level_players import create_level_player
from predicators.structs import Dataset
from scripts.cluster_utils import config_to_cmd_flags, generate_run_configs
from tests.approaches.test_agent_continual_approach import _call, _config, \
    _result

CONFIG = "predicatorv3/protocol_continual_comparisons_noisy_r1.yaml"


def test_comparison_config_matches_existing_domains(monkeypatch: Any) -> None:
    """Only six requested arms, same domain settings and paired seeds."""
    old = list(
        generate_run_configs(
            "predicatorv3/protocol_continual_noisy_sweep_r1.yaml", False))
    new = list(generate_run_configs(CONFIG, False))
    assert len(new) == 90
    assert len({c.approach for c in new}) == 6
    for cfg in new:
        monkeypatch.setattr(
            sys, "argv",
            ["predicators/main.py", *shlex.split(config_to_cmd_flags(cfg))])
        parsed = utils.parse_args()
        assert parsed["env"] == cfg.env
        assert parsed["approach"] == cfg.approach
        assert cfg.approach not in ("agent_continual",
                                    "agent_continual_model_free")
        reference = next(c for c in old if c.env == cfg.env)
        keys = [
            k for k in reference.flags
            if k.startswith(("continual_obs_noise_", "balloons_", "boil_",
                             "domino_"))
        ]
        keys += [
            "num_train_tasks", "num_test_tasks", "continual_steps_per_level",
            "continual_levels"
        ]
        for key in keys:
            assert cfg.flags[key] == reference.flags[key], (cfg.env, key)


@pytest.mark.parametrize(
    "arm", ["no_fitting", "no_uncertainty", "oracle_scene", "oracle_dynamics"])
def test_ablation_play_tools(tmp_path: Any, monkeypatch: Any,
                             arm: str) -> None:
    """Real noisy observations retain means; tools enforce arm restrictions."""
    cfg = next(c for c in generate_run_configs(CONFIG, False)
               if c.env == "pybullet_boil" and c.approach.endswith(arm))
    _config(
        tmp_path, **{
            **{k: v
               for k, v in cfg.flags.items() if k != "log"}, "continual_render":
            False,
            "continual_make_video": False,
            "continual_runs_dir": str(tmp_path / "runs"),
            "approach": cfg.approach
        })
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    agent: Any = create_approach(cfg.approach, env.predicates,
                                 get_gt_options(env.get_name()), env.types,
                                 env.action_space,
                                 [t.task for t in env.get_train_tasks()])

    def query(*_args: Any, **_kwargs: Any) -> Any:
        ctx = agent._tool_context  # pylint: disable=protected-access
        observation = _call(agent, "env_observe")
        assert "[noise]" in observation
        assert "[objects]" in observation
        prompt = agent._play_system_prompt()  # pylint: disable=protected-access
        if arm == "no_uncertainty":
            assert "[belief]" not in observation
            assert "[atoms under the belief]" not in observation
            assert "Point-estimate comparison" in prompt
            session = agent._play_session  # pylint: disable=protected-access
            obs = session.observe()
            assert obs.belief is not None
            assert obs.frame.allclose(obs.belief.frame)
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
        elif arm in {"oracle_scene", "oracle_dynamics"}:
            heading = ("Oracle scene reconstruction comparison" if arm
                       == "oracle_scene" else "Oracle dynamics comparison")
            assert heading in prompt
            assert "unavailable" in BeliefProbe(ctx).fit()
        else:
            assert "No numerical parameter fitting" in prompt
            assert "parameter estimation is disabled" in BeliefProbe(ctx).fit()
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


def test_standalone_model_is_live_without_engine(tmp_path: Any,
                                                 monkeypatch: Any) -> None:
    """A program edit changes predictions inside the acting conversation."""
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    from predicators.code_sim_learning.program_world_model import \
        ProgramOptionModel  # pylint: disable=import-outside-toplevel
    _config(tmp_path, approach="agent_continual_program_world_model")
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    options = get_gt_options(env.get_name())
    agent: Any = create_approach("agent_continual_program_world_model",
                                 env.predicates, options, env.types,
                                 env.action_space,
                                 [t.task for t in env.get_train_tasks()])
    calls = []

    def query(*_args: Any, **_kwargs: Any) -> Any:
        ctx = agent._tool_context  # pylint: disable=protected-access
        refs = agent._get_sandbox_reference_files()  # pylint: disable=protected-access
        assert not any(k.startswith("base_sim/") for k in refs)
        path = Path(agent._resolve_synthesis_paths().base) / "world_model.py"  # pylint: disable=protected-access
        code = ('LATENT_FEATURES = {}\n'
                'def initial_latent(obs, rng):\n    return {}\n'
                'def transition(obs, latent, option, rng):\n'
                '    return obs.copy(), dict(latent), COUNT\n')
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
        _, steps = calls[-1].get_next_state_and_num_actions(current, grounded)
        assert steps == 3
        plan = ("Wait(" + ", ".join(str(obj) for obj in objects) + ")[" +
                ", ".join(str(float(value)) for value in params) + "]")
        result = probe.run(plan, render=False)
        assert result is not None
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


def test_zero_shot_seals_before_first_charge(tmp_path: Any,
                                             monkeypatch: Any) -> None:
    """Missing models and later edits cannot take steps, even after resume."""
    from pathlib import Path  # pylint: disable=import-outside-toplevel
    _config(tmp_path, approach="agent_continual_zero_shot")
    env = create_new_env("pybullet_boil", do_cache=False, use_gui=False)
    agent: Any = create_approach("agent_continual_zero_shot", env.predicates,
                                 get_gt_options(env.get_name()), env.types,
                                 env.action_space,
                                 [t.task for t in env.get_train_tasks()])
    code = ('class Model(BaseSimulator):\n'
            '    AGENT_PARAM_SPECS = []\n'
            '    RESIDUAL_FEATURES = {}\n'
            '    def _domain_specific_step(self):\n        pass\n'
            'RESIDUAL_ENV = Model\n')

    def query(*_args: Any, **_kwargs: Any) -> Any:
        action = [0.0] * env.action_space.shape[0]
        result = _call(agent, "env_step", action=action)
        assert "step applied" not in result
        path = Path(agent._resolve_synthesis_paths().simulator_file)  # pylint: disable=protected-access
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        assert "step applied" in _call(agent, "env_step", action=action)
        saved = agent._extra_save_state()  # pylint: disable=protected-access
        assert saved["frozen_model_source"] == code
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
