"""Every PyBullet call that reaches a physics server names its client.

PyBullet sends a call without ``physicsClientId`` to client 0, the first
world the process connected. Each env owns its own client, so such a
call reads or edits whichever env was built first: Circuit, Laser and
Switch failed to build after any other env, and Laser's planning world
drew its laser beams into the executing env.
"""

import ast
from pathlib import Path
from typing import List

import predicators

# PyBullet functions that are pure math and never reach a physics server.
_PURE_FUNCTIONS = frozenset({
    "computeProjectionMatrix",
    "computeProjectionMatrixFOV",
    "computeViewMatrix",
    "computeViewMatrixFromYawPitchRoll",
    "connect",
    "getAxisAngleFromQuaternion",
    "getDifferenceQuaternion",
    "getEulerFromQuaternion",
    "getMatrixFromQuaternion",
    "getQuaternionFromAxisAngle",
    "getQuaternionFromEuler",
    "getQuaternionSlerp",
    "invertTransform",
    "multiplyTransforms",
    "rotateVector",
})


def _calls_without_client_id(path: Path) -> List[str]:
    """The PyBullet calls in one file that omit ``physicsClientId``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names if alias.name == "pybullet"
    }
    missing = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in aliases):
            continue
        if node.func.attr in _PURE_FUNCTIONS:
            continue
        # A **kwargs splat may carry the client; trust it.
        keywords = {kw.arg for kw in node.keywords}
        if "physicsClientId" in keywords or None in keywords:
            continue
        missing.append(f"{path}:{node.lineno} pybullet.{node.func.attr}")
    return missing


def test_pybullet_calls_name_their_client():
    """No call in the predicators package falls back to client 0."""
    package_dir = Path(predicators.__file__).parent
    missing = []
    for path in sorted(package_dir.rglob("*.py")):
        missing.extend(_calls_without_client_id(path))
    assert not missing, ("PyBullet calls without physicsClientId=:\n" +
                         "\n".join(missing))
