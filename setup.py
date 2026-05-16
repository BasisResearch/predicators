"""Setup script."""
from setuptools import find_packages, setup

# Heavy ML / LLM / planning deps. Required for `python predicators/main.py`
# (learning approaches, NSRT pipeline, LLM-backed tools) but NOT for using
# the PyBullet envs directly (via the gymnasium_wrapper or env modules).
# `pip install predicators` is the slim env-runtime install; learners
# should `pip install predicators[ml]` (or `[develop]`, which is a
# superset including formatters/test tools).
ML_DEPS = [
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "scipy==1.9.3",
    "scikit-learn>=1.1.3",
    "pandas==1.5.1",
    "seaborn==0.12.1",
    "imageio==2.22.2",
    "imageio-ffmpeg",
    "openai==1.19.0",
    "google-generativeai",
    "tenacity",
    "httpx==0.27.0",
    "claude-agent-sdk",
    "nest_asyncio",
    "ImageHash",
    "pathos",
    "slack_bolt",
    "emcee",
    "lisdf",
    "requests",
    "smepy@git+https://github.com/sebdumancic/structure_mapping.git",
    "pg3@git+https://github.com/tomsilver/pg3.git",
    "gym_sokoban@git+https://github.com/Learning-and-Intelligent-Systems/gym-sokoban.git",  # pylint: disable=line-too-long
]

# Dev tooling (formatters, linters, pytest plugins). Always paired with
# ML_DEPS in the `develop` extra so contributors get a fully working
# checkout from a single `pip install -e .[develop]`.
DEV_TOOLING = [
    "pytest-cov==2.12.1",
    "pytest-pylint==0.18.0",
    "yapf==0.32.0",
    "docformatter==1.4",
    "isort==5.10.1",
    "mypy-extensions==1.0.0",
]

setup(
    name="predicators",
    version="0.1.0",
    packages=find_packages(include=["predicators", "predicators.*"]),
    install_requires=[
        # PyBullet env-runtime — matches the dep set that boots the
        # browser POC and the gymnasium_wrapper. Pins on numpy /
        # matplotlib / pillow / dill are relaxed to >= because Pyodide
        # ships fixed versions (numpy 2.2.5, matplotlib 3.8.4, etc.)
        # and the env code doesn't depend on the exact older pins.
        "numpy>=1.23.5",
        "gym==0.26.2",
        "gymnasium>=0.28.0",
        "matplotlib>=3.6.2",
        "pillow>=10.3.0",
        # On Pyodide (sys_platform == "emscripten") the pybullet wheel
        # is loaded from a local emfs path by the browser bootstrap;
        # PyPI has no Pyodide-targeted pybullet.
        "pybullet-arm64>=3.2.8; sys_platform != 'emscripten'",
        "pyperplan",
        "tabulate>=0.9.0",
        "dill>=0.3.5.1",
        "colorlog",
        "pyyaml>=6.0",
        "types-PyYAML",
        "graphlib-backport",
        # Test/lint tooling stays in base for now (CLAUDE.md docs the
        # runtime install as including them).
        "pytest==7.1.3",
        "mypy==1.8.0",
        "pylint==2.14.5",
    ],
    include_package_data=True,
    extras_require={
        "ml": ML_DEPS,
        "develop": ML_DEPS + DEV_TOOLING,
    })
