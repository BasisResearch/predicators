"""The experiment config loader and the EMPIRIC benchmark config: includes,
parked entries, rounds and launch subsets."""
import os
from typing import Any, Dict, List

import pytest
import yaml

from scripts.cluster_utils import SingleSeedRunConfig, generate_run_configs, \
    parse_seed_range

BENCHMARK = "empiric/benchmark.yaml"


def _benchmark_runs(**kwargs: Any) -> List[SingleSeedRunConfig]:
    runs = []
    for config in generate_run_configs(BENCHMARK, False, **kwargs):
        assert isinstance(config, SingleSeedRunConfig)
        runs.append(config)
    return runs


def test_benchmark_is_seven_arms_on_five_settings() -> None:
    """The default config runs the seven arms on the five settings, five seeds
    each, with each arm's capability contract."""
    runs = _benchmark_runs(round_name="r9")
    assert len(runs) == 7 * 5 * 5
    assert {r.seed for r in runs} == set(range(5))
    assert len({(r.experiment_id, r.seed) for r in runs}) == len(runs)
    assert {r.env
            for r in runs} == {
                "pybullet_balloons", "pybullet_bridge", "pybullet_boil",
                "pybullet_fan", "pybullet_domino"
            }
    assert {r.experiment_id.split("-", 1)[1]
            for r in runs} == {
                f"{arm}_opus_r9"
                for arm in ("mb", "mf", "mf_scene_package", "standalone",
                            "oracle_dynamics", "no_fitting", "no_uncertainty")
            }
    # The scene-package arm shares the model-free class, not its flags.
    assert len({r.approach for r in runs}) == 6
    for run in runs:
        flags = run.flags
        assert flags["agent_sdk_model_name"] == "claude-opus-5"
        assert flags["experiment_protocol"] == "continual"
        assert flags["partially_observable"]
        assert not flags.get("continual_skill_preflight", False)
        assert not flags.get("continual_validation_audit", False)
        assert flags["continual_wall_clock_hours"] == 48.0
        assert "auto_resume" in run.args
        # The principled joint belief is the default; only No uncertainty
        # turns it off.
        assert not flags["code_sim_learning_carry_posterior"]
        assert flags["belief_joint_draws"] == (
            0 if run.approach == "agent_continual_no_uncertainty" else 16)
        if run.approach == "agent_continual_model_free":
            assert not flags["agent_planner_use_simulator"]
            assert not flags["continual_uncertainty_decisions"]
            assert bool(flags.get("continual_provide_scene_package")) == (
                "mf_scene_package" in run.experiment_id)
        elif run.approach == "agent_continual_no_fitting":
            assert flags["agent_sim_learn_declared_params_only"]
            assert flags["continual_uncertainty_decisions"]
            assert flags["continual_require_model_on_test"]
        elif run.approach == "agent_continual_no_uncertainty":
            for name in ("continual_obs_noise_declared",
                         "continual_uncertainty_decisions",
                         "agent_sim_learn_param_uncertainty",
                         "code_sim_learning_interval_belief",
                         "code_sim_learning_rollout_noise_filter",
                         "continual_belief_frame"):
                assert not flags[name]
            assert flags["continual_require_model_on_test"]
        elif run.approach == "agent_continual_oracle_dynamics":
            # The repaired Oracle policy: no automatic execution gate or
            # shadow audit.
            assert flags["continual_skill_preflight"] is False
            assert flags["continual_validation_audit"] is False
        if run.env == "pybullet_bridge":
            assert flags["bridge_train_span_blocks"] == 3
            assert flags["bridge_test_span_blocks"] == 4
            assert flags["continual_steps_per_level"] == 10000
        else:
            assert flags["continual_steps_per_level"] == 5000
        if run.env == "pybullet_balloons":
            assert flags["num_train_tasks"] == 2
            assert flags["balloons_goal_dwell_steps"] == 25
        if run.env == "pybullet_fan":
            assert flags["fan_ramp_transfer"]
            assert flags["fan_ramp_rise"] == 0.003
        if run.env == "pybullet_boil":
            assert flags["boil_num_jugs_test"] == [2]


def test_benchmark_subsets_and_rounds() -> None:
    """--envs, --approaches and --seeds pick a subset; --round names every
    experiment id."""
    runs = _benchmark_runs(round_name="fan_fix_r1",
                           envs=["fan"],
                           approaches=["mb_opus", "mf_opus"],
                           seeds=parse_seed_range("2-4"))
    assert sorted({r.experiment_id
                   for r in runs
                   }) == ["fan-mb_opus_fan_fix_r1", "fan-mf_opus_fan_fix_r1"]
    assert sorted(r.seed for r in runs) == [2, 2, 3, 3, 4, 4]
    with pytest.raises(ValueError, match="unknown envs"):
        _benchmark_runs(round_name="r1", envs=["fan_maze"])
    with pytest.raises(ValueError, match="unknown approaches"):
        _benchmark_runs(round_name="r1", approaches=["mb_sonnet"])
    with pytest.raises(ValueError, match="letters, digits"):
        _benchmark_runs(round_name="r 1")


def test_continual_launch_requires_a_round() -> None:
    """A launch that could resume an earlier one's run folders fails until it
    names a round."""
    with pytest.raises(ValueError, match="name a round"):
        _benchmark_runs(require_round=True)
    assert _benchmark_runs(round_name="r2", require_round=True)


def test_parse_seed_range() -> None:
    """Seed ranges are N or N-M, inclusive."""
    assert parse_seed_range("3") == (3, 1)
    assert parse_seed_range("0-4") == (0, 5)
    for bad in ("4-2", "a", "1-", "-3"):
        with pytest.raises(ValueError):
            parse_seed_range(bad)


def _write(configs_dir: str, name: str, content: Dict[str, Any]) -> None:
    with open(os.path.join(configs_dir, name), "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f)


def test_includes_skip_round_and_precedence(monkeypatch: Any,
                                            tmp_path: Any) -> None:
    """A launcher includes menus and a common file; parked entries stay out
    unless selected, the ROUND key names the ids, and env flags override arm
    flags, which override the common ones."""
    configs_dir = tmp_path / "configs"
    os.makedirs(configs_dir / "menu")
    _write(
        str(configs_dir), "common.yaml", {
            "START_SEED": 0,
            "NUM_SEEDS": 1,
            "ARGS": ["debug"],
            "FLAGS": {
                "shared": 1,
                "over": "common"
            }
        })
    _write(
        str(configs_dir / "menu"), "envs.yaml", {
            "ENVS": {
                "balloons": {
                    "NAME": "pybullet_balloons",
                    "FLAGS": {
                        "over": "env"
                    }
                },
                "bridge": {
                    "NAME": "pybullet_bridge",
                    "SKIP": True
                }
            }
        })
    _write(
        str(configs_dir / "menu"), "arms.yaml", {
            "APPROACHES": {
                "mb": {
                    "NAME": "agent_continual",
                    "FLAGS": {
                        "gate": True,
                        "over": "arm"
                    }
                }
            }
        })
    _write(
        str(configs_dir), "launch.yaml", {
            "includes": ["common.yaml", "menu/envs.yaml", "menu/arms.yaml"],
            "ROUND": "r2",
            "APPROACHES": {
                "mb": {
                    "FLAGS": {
                        "round": 2
                    }
                }
            }
        })
    monkeypatch.setattr("scripts.cluster_utils.os.path.realpath",
                        lambda _: str(tmp_path / "cluster_utils.py"))
    runs = list(generate_run_configs("launch.yaml", batch_seeds=True))
    assert [r.experiment_id for r in runs] == ["balloons-mb_r2"]
    run = runs[0]
    assert run.approach == "agent_continual"
    assert run.env == "pybullet_balloons"
    assert run.flags["gate"] is True
    assert run.flags["round"] == 2
    assert run.flags["shared"] == 1
    assert run.flags["over"] == "env"
    # A command-line round overrides the config's; naming a parked entry
    # launches it.
    runs = list(
        generate_run_configs("launch.yaml",
                             batch_seeds=True,
                             round_name="r3",
                             envs=["bridge"]))
    assert [r.experiment_id for r in runs] == ["bridge-mb_r3"]
