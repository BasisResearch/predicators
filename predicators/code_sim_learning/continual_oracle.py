"""Frozen oracle artifacts for the continual comparison.

These artifacts contain mechanism code, not controllers or task
generators. Unsupported domains fail explicitly until their observation-
memory contract has been audited against the current environment.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Dict

from predicators.settings import CFG


def _fixed_specs(params: Dict[str, float]) -> str:
    return "[" + ", ".join(
        f"ParamSpec({key!r}, {value!r}, lo={value!r}, hi={value!r})"
        for key, value in sorted(params.items())) + "]"


def oracle_source() -> str:
    """Return a self-contained artifact with frozen parameter values."""
    # Imports here avoid loading all PyBullet environments before registry
    # discovery has completed.
    # pylint: disable=import-outside-toplevel
    if CFG.env == "pybullet_boil":
        boil_path = Path(__file__).with_name("boil_oracle.py")
        tree = ast.parse(boil_path.read_text(encoding="utf-8"))
        tree.body = [
            node for node in tree.body
            if not (isinstance(node, ast.ImportFrom)
                    and node.module == "predicators.envs.pybullet_boil")
        ]
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "BoilOracle":
                node.bases = [ast.Name(id="BaseSimulator", ctx=ast.Load())]
        tree.body.extend(ast.parse("RESIDUAL_ENV = BoilOracle").body)
        return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    if CFG.env == "pybullet_domino":
        return ("class OracleDynamics(BaseSimulator):\n"
                "    AGENT_PARAM_SPECS = " + _fixed_specs(
                    {"lateral_friction": float(CFG.domino_true_friction)}) +
                "\n"
                "    RESIDUAL_FEATURES = {}\n"
                "    def _domain_specific_step(self):\n        pass\n"
                "RESIDUAL_ENV = OracleDynamics\n")
    if CFG.env == "pybullet_fan":
        if CFG.fan_use_kinematic:
            raise ValueError("Continual oracle supports dynamic Fan only")
        # The native helper controls fan rotors as well as queuing the wind.
        # It is part of the concrete visible-base class, whose top-level
        # residual hook is normally disabled.
        from predicators.envs.pybullet_fan import PyBulletFanEnv
        wind = float(PyBulletFanEnv.wind_force_magnitude)
        return ("class OracleDynamics(BaseSimulator):\n"
                "    AGENT_PARAM_SPECS = []\n"
                "    RESIDUAL_FEATURES = {'ball': ['x', 'y']}\n"
                f"    wind_force_magnitude = {wind!r}\n"
                "    def _domain_specific_step(self):\n"
                "        self._simulate_fans_dynamic()\n"
                "RESIDUAL_ENV = OracleDynamics\n")
    if CFG.env == "pybullet_balloons":
        from predicators.envs.pybullet_balloons_base import \
            PyBulletBalloonsBaseEnv
        from predicators.ground_truth_models.balloons import gt_simulator_env
        params = {
            f"lift_{name}": float(CFG.balloons_lifts[i])
            for i, (name,
                    _) in enumerate(PyBulletBalloonsBaseEnv.BALLOON_PALETTE)
        }
        params.update({
            f"mass_{name}": float(CFG.balloons_box_masses[i])
            for i, (name, _) in enumerate(PyBulletBalloonsBaseEnv.BOX_PALETTE)
        })
        params.update(fade_height=float(CFG.balloons_fade_height),
                      air_drag=float(CFG.balloons_drag))
        source_file = inspect.getsourcefile(gt_simulator_env)
        assert source_file is not None
        tree = ast.parse(Path(source_file).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node,
                          ast.ClassDef) and node.name == "BalloonsResidualEnv":
                node.body.extend(
                    ast.parse(
                        "def _box_mass_for(self, color_index):\n"
                        "    return self.agent_param("
                        "_mass_param_name(color_index))\n"
                        "def _drag(self):\n"
                        "    return self.agent_param('air_drag')\n").body)
                for index, stmt in enumerate(node.body):
                    if (isinstance(stmt, ast.AnnAssign)
                            and isinstance(stmt.target, ast.Name)
                            and stmt.target.id == "AGENT_PARAM_SPECS"):
                        node.body[index] = ast.parse(
                            "AGENT_PARAM_SPECS = " +
                            _fixed_specs(params)).body[0]
        return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    raise NotImplementedError(
        "Current continual oracle memory is not yet audited for " + CFG.env)
