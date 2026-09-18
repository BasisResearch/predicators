"""Frozen oracle artifacts for the continual comparison.

These artifacts contain mechanism code, not controllers or task
generators. Unsupported domains fail explicitly until their observation-
memory contract has been audited against the current environment.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

from predicators.settings import CFG


def oracle_source() -> str:
    """Return a self-contained artifact with frozen parameter values."""
    # Imports here avoid loading all PyBullet environments before registry
    # discovery has completed.
    # pylint: disable=import-outside-toplevel,protected-access
    if CFG.env == "pybullet_bridge":
        from predicators.envs.pybullet_bridge import ATTACH_SLOTS, \
            GLUE_FACES, PyBulletBridgeEnv
        bridge_path = Path(__file__).with_name("bridge_oracle.py")
        tree = ast.parse(bridge_path.read_text(encoding="utf-8"))
        tree.body = [
            node for node in tree.body
            if not (isinstance(node, ast.ImportFrom)
                    and node.module == "predicators.envs.pybullet_bridge")
        ]
        native = ast.parse(
            textwrap.dedent(
                inspect.getsource(
                    PyBulletBridgeEnv._domain_specific_step))).body[0]
        assert isinstance(native, ast.FunctionDef)
        native.name = "_native_step"
        for index, node in enumerate(tree.body):
            if (isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "_native_step"
                    for t in node.targets)):
                tree.body[index:index + 1] = [
                    *ast.parse("from typing import Tuple\n"
                               "import numpy as np\n"
                               f"GLUE_FACES = {GLUE_FACES!r}\n"
                               f"ATTACH_SLOTS = {ATTACH_SLOTS!r}").body, native
                ]
                break
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "BridgeOracle":
                node.bases = [ast.Name(id="BaseSimulator", ctx=ast.Load())]
        tree.body.extend(ast.parse("RESIDUAL_ENV = BridgeOracle").body)
        return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
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
        friction = float(CFG.domino_true_friction)
        return (f"_FIXED_PARAMS = {{'lateral_friction': {friction!r}}}\n"
                "class OracleDynamics(BaseSimulator):\n"
                "    AGENT_PARAM_SPECS = []\n"
                "    RESIDUAL_FEATURES = {}\n"
                "    def __init__(self, *args, **kwargs):\n"
                "        super().__init__(*args, **kwargs)\n"
                "        self.apply_physical_param_overrides("
                "dict(_FIXED_PARAMS))\n"
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
                # The true values are module constants read through
                # ``agent_param``, not declared parameters: nothing the
                # probe reports or the predicate loader exposes names them.
                node.body.extend(
                    ast.parse(
                        "def agent_param(self, name):\n"
                        "    return _FIXED_PARAMS[name]\n"
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
                            "AGENT_PARAM_SPECS = []").body[0]
        # After the docstring and any ``from __future__`` import, which
        # must lead the file.
        def _leads(node: ast.AST) -> bool:
            return (isinstance(node, ast.ImportFrom) and node.module
                    == "__future__") or (isinstance(node, ast.Expr) and
                                         isinstance(node.value, ast.Constant))

        first = next(
            (i for i, node in enumerate(tree.body) if not _leads(node)), 0)
        tree.body.insert(
            first,
            ast.parse(
                f"_FIXED_PARAMS = {dict(sorted(params.items()))!r}").body[0])
        return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    raise NotImplementedError(
        "Current continual oracle memory is not yet audited for " + CFG.env)
