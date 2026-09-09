#!/bin/bash
set -euo pipefail

echo "Running autoformatting."
./run_autoformat.sh
echo "Autoformatting complete."

echo "Running type checking."
mypy . --config-file mypy.ini
echo "Type checking passed."

echo "Running linting."
pytest . --pylint -m pylint --pylint-rcfile=.predicators_pylintrc
echo "Linting passed."

echo "Running unit tests."
pytest -s tests/ --cov-config=.coveragerc --cov=predicators/ --cov=tests/ --cov-report=term-missing:skip-covered --durations=0
echo "Unit tests passed."

echo "All checks passed!"
