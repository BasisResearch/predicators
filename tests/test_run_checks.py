"""The README check command must fail when any constituent check fails."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "failure", ["yapf", "docformatter", "isort", "mypy", "lint", "tests", ""])
def test_run_checks_exit_status(tmp_path: Path, failure: str) -> None:
    """Run the real shell scripts with deterministic check command stubs."""
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(root / "run_autoformat.sh", tmp_path / "run_autoformat.sh")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for command in ("yapf", "docformatter", "isort", "mypy", "pytest"):
        stub = bindir / command
        stub.write_text("#!/bin/sh\n"
                        "step=${0##*/}\n"
                        "if [ \"$step\" = pytest ]; then\n"
                        "  if [ \"$1\" = . ]; then step=lint; "
                        "else step=tests; fi\n"
                        "fi\n"
                        "if [ \"$step\" = \"$CHECK_FAILURE\" ]; then "
                        "exit 13; fi\n"
                        "exit 0\n")
        stub.chmod(0o755)
    env = dict(os.environ,
               PATH=str(bindir) + os.pathsep + os.environ["PATH"],
               CHECK_FAILURE=failure)
    result = subprocess.run(
        ["bash", str(root / "scripts/run_checks.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False)
    assert result.returncode == (13 if failure else 0), result.stdout
    assert ("All checks passed!" in result.stdout) == (not failure)
