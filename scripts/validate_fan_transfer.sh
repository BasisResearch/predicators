#!/bin/bash -l
# Short physical validation only, submitted on mit_quicktest.
set -euo pipefail
module load miniforge/25.11.0-0
conda activate pred
cd /home/ycliang/predicators
export PYTHONPATH=/home/ycliang/predicators
export PYTHONHASHSEED=0
pytest -s tests/envs/test_pybullet_fan_transfer.py tests/envs/test_pybullet_fan_boundary.py tests/envs/test_pybullet_fan_maze.py -q
