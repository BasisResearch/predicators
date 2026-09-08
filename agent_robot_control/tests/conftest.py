"""Shared fixtures: every test builds envs headless with a fresh CFG."""
import pytest

from predicators import utils


def make_env(env_name: str, env_cls, **cfg_overrides):
    """Construct ``env_cls`` headless with a clean predicators config."""
    cfg = {
        "env": env_name,
        "seed": 0,
        "num_train_tasks": 5,
        "num_test_tasks": 1,
        "pybullet_control_mode": "position",
    }
    cfg.update(cfg_overrides)
    utils.reset_config(cfg)
    return env_cls(use_gui=False)


@pytest.fixture
def scratch_dir(tmp_path):
    """Per-test scratch directory."""
    return tmp_path
