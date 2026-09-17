"""The experiment config loader: includes, parked menu entries and EXTENDS."""
import os
from typing import Any, Dict

import pytest
import yaml

from scripts.cluster_utils import _resolve_extends, generate_run_configs


def _write(configs_dir: str, name: str, content: Dict[str, Any]) -> None:
    with open(os.path.join(configs_dir, name), "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f)


def test_extends_gives_a_menu_entry_a_new_id(monkeypatch: Any,
                                             tmp_path: Any) -> None:
    """A launcher un-parks a menu env and derives round-specific arms from
    parked menu arms with EXTENDS; ids, names and merged flags follow."""
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
                    "SKIP": True,
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
                    "SKIP": True,
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
            "ENVS": {
                "balloons": {
                    "SKIP": False
                }
            },
            "APPROACHES": {
                "mb_r2": {
                    "EXTENDS": "mb",
                    "FLAGS": {
                        "round": 2
                    }
                },
                "mb_parked": {
                    "EXTENDS": "mb",
                    "SKIP": True
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
    # Env flags override arm flags, which override the common ones.
    assert run.flags["over"] == "env"


def test_extends_rejects_unknown_and_chained_bases() -> None:
    """EXTENDS names a menu entry of the same section, one level deep."""
    with pytest.raises(ValueError, match="unknown"):
        _resolve_extends({"a": {"EXTENDS": "missing"}})
    with pytest.raises(ValueError, match="itself EXTENDS"):
        _resolve_extends({
            "base": {
                "NAME": "x"
            },
            "mid": {
                "EXTENDS": "base"
            },
            "top": {
                "EXTENDS": "mid"
            }
        })
    resolved = _resolve_extends({
        "base": {
            "NAME": "x",
            "SKIP": True
        },
        "top": {
            "EXTENDS": "base"
        }
    })
    assert resolved["base"]["SKIP"] is True
    assert resolved["top"] == {"NAME": "x", "SKIP": False}
