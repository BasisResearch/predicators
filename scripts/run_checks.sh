#!/bin/bash
set -euo pipefail

# Everything runs through `uv run` (this branch uses uv, not a conda env).
echo "Running autoformatting."
./run_autoformat.sh
echo "Autoformatting complete."

echo "Running type checking."
uv run mypy . --config-file mypy.ini
echo "Type checking passed."

echo "Running linting."
uv run pytest . --pylint -m pylint --pylint-rcfile=.predicators_pylintrc
echo "Linting passed."

echo "Running unit tests."
uv run pytest -s tests/ --cov-config=.coveragerc --cov=predicators/ --cov=tests/ --cov-report=term-missing:skip-covered --durations=0
echo "Unit tests passed."

echo "All checks passed!"
