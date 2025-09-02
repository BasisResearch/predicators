# Predicators Installation Instructions for Python 3.13

This guide provides step-by-step instructions for installing the `predicators` package on Python 3.13, addressing compatibility issues with the original setup.py dependencies.

## Prerequisites

- Python 3.13.x
- A virtual environment (recommended)
- Git (for Git-based dependencies)

## Installation Steps

### 1. Set up and activate your virtual environment

```bash
# If you don't have a virtual environment yet:
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Install build dependencies

The original setup.py has strict version pins that aren't compatible with Python 3.13. First, install the required build tools:

```bash
pip install --upgrade setuptools wheel
```

### 3. Install predicators without dependencies

This avoids the dependency resolution conflicts:

```bash
pip install --no-deps -e .
```

### 4. Install compatible dependencies

Install dependencies with more flexible version constraints:

```bash
pip install "numpy>=1.24.0" pytest mypy "gym>=0.26.0" matplotlib imageio imageio-ffmpeg pandas "torch>=2.0.0" torchvision scipy tabulate dill pyperplan pathos pillow requests slack_bolt "pybullet>=3.2.0" scikit-learn graphlib-backport "openai>=1.19.0" pyyaml pylint types-PyYAML lisdf seaborn ImageHash google-generativeai tenacity "httpx>=0.27.0" "opencv-python>=4.5.0" colorlog wandb
```

### 5. Install Git-based dependencies

Install the remaining dependencies from Git repositories:

```bash
pip install "git+https://github.com/sebdumancic/structure_mapping.git" "git+https://github.com/tomsilver/pg3.git" "git+https://github.com/Learning-and-Intelligent-Systems/gym-sokoban.git"
```

### 6. Install additional missing dependencies

```bash
pip install psutil
```

## Running Predicators

After installation, you need to set the `PYTHONHASHSEED` environment variable before running predicators:

```bash
# Set environment variable (required)
export PYTHONHASHSEED=0

# Example command
python predicators/main.py --env burger --approach vlm_open_loop --seed 0 --num_train_tasks 1 --num_test_tasks 1 --bilevel_plan_without_sim True --make_failure_videos --sesame_task_planner fdopt --debug --vlm_model_name gemini-1.5-pro-latest --vlm_open_loop_use_training_demos True
```

### Persistent Environment Variable Setup

To avoid setting `PYTHONHASHSEED=0` every time, add it to your shell profile:

```bash
# For bash
echo "export PYTHONHASHSEED=0" >> ~/.bashrc

# For zsh
echo "export PYTHONHASHSEED=0" >> ~/.zshrc
```

Then reload your shell or run:
```bash
source ~/.bashrc  # or ~/.zshrc
```

## Install Fast Downward

```bash
bash install_fast_downward.sh
export FD_EXEC_PATH=$(pwd)/external/downward/fast-downward.py
```

## Verification

To verify the installation worked:

```bash
source .venv/bin/activate
export PYTHONHASHSEED=0
python predicators/main.py --help
```

You should see the help message without any import errors.

## Known Issues and Warnings

1. **Gym deprecation warning**: You may see warnings about gym being unmaintained. This is expected and doesn't affect functionality.

2. **Package version conflicts**: The dependency resolver will show warnings about version conflicts between the strict pins in setup.py and the installed versions. These are expected and generally don't cause issues.

3. **pkg_resources deprecation**: You may see warnings about pkg_resources being deprecated. This comes from some dependencies and is not critical.

## Troubleshooting

### If you encounter "ModuleNotFoundError"

Make sure:
1. Your virtual environment is activated
2. You've installed all dependencies as listed above
3. The predicators package was installed with `pip install --no-deps -e .`

### If you encounter build errors

Ensure you have the latest setuptools and wheel:
```bash
pip install --upgrade setuptools wheel pip
```

### If specific dependencies fail to install

Some dependencies may need system-level packages. On macOS with Homebrew:
```bash
# For opencv-python issues
brew install opencv

# For other compilation issues
xcode-select --install
```

## Key Differences from Original setup.py

The main changes made for Python 3.13 compatibility:

- **numpy**: Updated from `==1.23.5` to `>=1.24.0` (numpy 1.23.5 doesn't support Python 3.13)
- **torch/torchvision**: Updated to compatible versions
- **Other packages**: Used more flexible version constraints instead of strict pins
- **Missing dependencies**: Added `psutil` which was required but not listed

This approach maintains compatibility while working around the Python 3.13 restrictions in the original dependency specifications.
