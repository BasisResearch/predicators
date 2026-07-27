#!/bin/bash
yapf -i -r --style .style.yapf --exclude '**/third_party' predicators
yapf -i -r --style .style.yapf scripts
yapf -i -r --style .style.yapf tests
yapf -i -r --style .style.yapf setup.py
# third_party/ holds git submodules: formatting them would dirty another repo's
# working tree with changes this repo's style config, not theirs, asked for.
docformatter -i -r . --exclude venv predicators/third_party third_party
isort . --skip third_party
