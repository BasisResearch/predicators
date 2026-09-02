# Predicators Agent Sandbox

## Working Directory

Your working directory is the sandbox. All files you create must stay here, and every path you use is relative (for example `./my_script.py`).

## Python

The interpreter is `python3`, and the predicators package is importable:

    python3 -c "from predicators.structs import State, Type; print('OK')"

Write and run scripts in the sandbox; anything you can express in Python is fair game, on the data below and on the files you create:

    python3 my_experiment.py

## Data

The trajectories collected so far (training episodes only) are in `./data/trajectories.pkl`, refreshed before every query: a pickled list of dicts, one per episode, with `states` (a list of `predicators.structs.State`, the observed object features in `.data`), `actions` (a list of `{"arr": ndarray, "option": str}`, the option label naming the grounded skill and its parameters, for example `Pick(robot, block1)[0.05]`), `is_demo` and `train_task_idx`. `states[i]` is the state before `actions[i]`.

    import pickle
    with open("./data/trajectories.pkl", "rb") as f:
        episodes = pickle.load(f)

## Reference Files

Curated source files are in `./reference/`. Read them to learn the APIs before writing code.

## Session Logs

Your earlier queries and tool results are in `./session_logs/`, named `<NNN>_<kind>_<timestamp>.md` where `<NNN>` is a run-wide counter and `<kind>` is the query phase (`learn`, `test`, `explore`). Alphabetical order is chronological order:

    Glob ./session_logs/*.md
    Read ./session_logs/001_learn_*.md

## Scene Images

`submit_plan` saves a rendering of the scene after every step to `./test_images/`. Read them to inspect the spatial state.

## Rules

- Do not read or browse files outside the sandbox. This is enforced for the file tools, for Bash, and for `run_python`: commands or code with absolute or `../` paths that leave the sandbox, or that introspect source, are blocked. Every `python3` you start carries the same guard, so scripts cannot import or open the hidden modules either; do not edit `PYTHONPATH` or pass `-S`, `-I` or `-E` to python.
- Do not modify files in `./reference/`; they are read-only.
- Do not inspect the predicators source code (`inspect.getsource`, `inspect.getfile`, reading `.py` files from site-packages, or any other route). Use the tools and the reference files instead.
- Do not reach into harness internals from executed code. `State.privileged`, the probe's `_ctx`, and env flags or attributes are hidden environment ground truth and are blocked. Learn the world model from observable state features and the documented tool surface only; a conclusion derived from hidden internals is an invalid result.
