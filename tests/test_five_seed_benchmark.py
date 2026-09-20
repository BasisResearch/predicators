"""The additional paper seeds preserve the intended capability contracts."""
from scripts.cluster_utils import SingleSeedRunConfig, generate_run_configs


def test_five_seed_launch_contracts() -> None:
    """Exactly six arms, five domains, and two fresh seeds with matched
    costs."""
    runs = []
    for config in generate_run_configs(
            "predicatorv3/continual_benchmark_five_seeds.yaml", False):
        assert isinstance(config, SingleSeedRunConfig)
        runs.append(config)
    assert len(runs) == 60
    assert {r.seed for r in runs} == {3, 4}
    assert len({(r.experiment_id, r.seed) for r in runs}) == 60
    assert len({r.env for r in runs}) == 5
    # Direct + scene shares the direct-agent class, not its capability flags.
    assert len({r.approach for r in runs}) == 5
    assert len({r.experiment_id.split("-", 1)[1] for r in runs}) == 6
    for run in runs:
        flags = run.flags
        assert flags["agent_sdk_model_name"] == "claude-opus-5"
        assert flags["partially_observable"]
        assert not flags["continual_skill_preflight"]
        assert not flags["continual_validation_audit"]
        assert flags["continual_wall_clock_hours"] == 48.0
        assert "auto_resume" in run.args
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
                         "code_sim_learning_carry_posterior",
                         "code_sim_learning_rollout_noise_filter",
                         "continual_belief_frame"):
                assert not flags[name]
            assert flags["continual_require_model_on_test"]
        if run.env == "pybullet_bridge":
            assert flags["bridge_test_span_blocks"] == 4
            assert flags["continual_steps_per_level"] == 10000
        else:
            assert flags["continual_steps_per_level"] == 5000
        if run.env == "pybullet_balloons":
            assert flags["num_train_tasks"] == 2
            assert flags["balloons_goal_dwell_steps"] == 25
