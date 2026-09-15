"""Offline workers must not inherit mutable sidecars or ambient settings."""
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from predicators.code_sim_learning.inference_runtime import RuntimeFile, \
    RuntimeInputs


def test_isolated_worker_input_identity_and_source_edits(
        tmp_path: Path) -> None:
    """Actual child processes get frozen inputs and a complete environment."""
    program = b"""import json, os
from pathlib import Path
p = Path('model_params.json')
data = json.loads(p.read_text()) if p.exists() else {'lift': 1.}
print(json.dumps([data['lift'], os.environ.get('INFERENCE_AMBIENT_TEST'),
                  os.environ.get('DECLARED_MODE')]))
"""
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"lift": 2.}))
    present = RuntimeInputs((RuntimeFile(
        "worker.py", program), RuntimeFile.read("model_params.json", source)),
                            (("DECLARED_MODE", "offline"), ))
    absent = replace(present,
                     files=(RuntimeFile("worker.py", program),
                            RuntimeFile("model_params.json", None)))
    assert absent.digest != present.digest
    assert absent.digest != replace(absent, environment=()).digest
    source.write_text(json.dumps({"lift": 999.}))
    old = os.environ.get("INFERENCE_AMBIENT_TEST")
    os.environ["INFERENCE_AMBIENT_TEST"] = "must not be inherited"
    try:
        for label, inputs, expected in (("present", present, 2.),
                                        ("absent", absent, 1.)):
            work = tmp_path / label
            inputs.materialize(work)
            result = subprocess.run([sys.executable, "worker.py"],
                                    cwd=work,
                                    env=inputs.environment_dict,
                                    check=True,
                                    capture_output=True,
                                    text=True)
            assert json.loads(result.stdout) == [expected, None, "offline"]
            inputs.verify(work)
            manifest = inputs.artifacts.save(tmp_path / "artifacts")
            assert manifest.read_bytes() == inputs.artifacts.manifest
    finally:
        if old is None:
            os.environ.pop("INFERENCE_AMBIENT_TEST", None)
        else:
            os.environ["INFERENCE_AMBIENT_TEST"] = old


@pytest.mark.parametrize(
    "mutation", ["change", "remove", "add", "absent", "link", "directory"])
def test_runtime_mutation_is_detected(tmp_path: Path, mutation: str) -> None:
    """Changed or newly created inputs invalidate a candidate result."""
    inputs = RuntimeInputs(
        (RuntimeFile("nested/data.json",
                     b"{}"), RuntimeFile("optional.json", None)))
    work = tmp_path / "worker"
    inputs.materialize(work)
    target = work / "nested/data.json"
    if mutation == "change":
        target.write_bytes(b"changed")
    elif mutation == "remove":
        target.unlink()
    elif mutation == "add":
        (work / "new.json").write_bytes(b"new")
    elif mutation == "absent":
        (work / "optional.json").write_bytes(b"{}")
    elif mutation == "link":
        target.unlink()
        target.symlink_to(tmp_path / "outside")
    else:
        (work / "new-directory").mkdir()
    with pytest.raises(ValueError, match="runtime input"):
        inputs.verify(work)


def test_runtime_contract_validation(tmp_path: Path) -> None:
    """Missing source is not a silently recorded absent dependency."""
    with pytest.raises(ValueError, match="present regular"):
        RuntimeFile.read("optional", tmp_path / "missing")
    for name in ("", ".", "../escape", "a/../b", "/absolute", "a//b", "a\\b"):
        with pytest.raises(ValueError, match="relative paths"):
            RuntimeFile(name, b"")
    with pytest.raises(ValueError, match="Duplicate runtime file"):
        RuntimeInputs((RuntimeFile("a", None), RuntimeFile("a", b"")))
    with pytest.raises(ValueError, match="contain one another"):
        RuntimeInputs((RuntimeFile("a", None), RuntimeFile("a/b", b"")))
    with pytest.raises(ValueError, match="Duplicate runtime environment"):
        RuntimeInputs((), (("A", "x"), ("A", "y")))
    with pytest.raises(ValueError, match="environment entry"):
        RuntimeInputs((), (("A=B", "x"), ))
    inputs = RuntimeInputs(())
    work = tmp_path / "nonempty"
    work.mkdir()
    (work / "ambient.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="must be empty"):
        inputs.materialize(work)
    link = tmp_path / "link"
    link.symlink_to(work, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        inputs.materialize(link)
