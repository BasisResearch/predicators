#!/bin/bash

echo "Running autoformatting."
uv run yapf -i -r --style .style.yapf --exclude '**/third_party' predicators
uv run yapf -i -r --style .style.yapf scripts
uv run yapf -i -r --style .style.yapf tests
uv run docformatter -i -r . --exclude venv predicators/third_party
uv run isort .
echo "Autoformatting complete."

echo "Running type checking."
uv run mypy . --config-file mypy.ini
if [ $? -eq 0 ]; then
    echo "Type checking passed."
else
    echo "Type checking failed! Terminating check script early."
    exit
fi

echo "Running linting."
uv run pytest . --pylint -m pylint --pylint-rcfile=.predicators_pylintrc
if [ $? -eq 0 ]; then
    echo "Linting passed."
else
    echo "Linting failed! Terminating check script early."
    exit
fi

echo "Running unit tests."
uv run pytest -s tests/ --cov-config=.coveragerc --cov=predicators/ --cov=tests/ --cov-report=term-missing:skip-covered --durations=0
if [ $? -eq 0 ]; then
    echo "Unit tests passed."
else
    echo "Unit tests failed!"
    exit
fi

echo "All checks passed!"
