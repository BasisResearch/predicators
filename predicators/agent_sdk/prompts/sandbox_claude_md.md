# Sandbox CLAUDE.md

Written into every agent sandbox by `sandbox_prompts.build_claude_md`.
It documents the sandbox mechanics only; task guidance lives in the system prompts and current queries.

<!-- section: body -->
# Predicators Agent Sandbox

## Working Directory

Your working directory is the sandbox.
All files you create must stay here, and every path you use is relative (for example `./my_script.py`).

## Python

The interpreter is `python3`, and the predicators package is importable:

    python3 -c "from predicators.structs import State, Type; print('OK')"

Write and run scripts in the sandbox; anything you can express in Python is fair game, on the data below and on the files you create:

    python3 my_experiment.py

## Data

`./data/trajectories.pkl` contains the experience made available by the current protocol as a pickled list of episode dictionaries.
Each entry has `states` (a list of `predicators.structs.State`), `actions` (dictionaries with `arr` and `option`), `is_demo`, and `train_task_idx`.
`states[i]` precedes `actions[i]`; observable object features are in the state's `.data`.
The option label names the grounded skill and parameters when available, for example `Move(widget, fixture)[0.05]`.
In continual play, records include the current episode and refresh after charged environment calls; in phased runs, the harness refreshes data before queries.
Use the current query for the available episode count and scope.

    import pickle
    with open("./data/trajectories.pkl", "rb") as f:
        episodes = pickle.load(f)

## Reference Files

Curated source files are in `./reference/`.
Read them to learn the APIs before writing code.

## Session Logs

Earlier queries and tool results are in `./session_logs/`, named `<NNN>_<kind>_<timestamp>.md` in chronological order.
The kind is `play` for continual rounds, or a phase such as `learn`, `test`, or `explore`.

    Glob ./session_logs/*.md

## Scene Images

Tool results name saved renders in `./test_images/`.
Read those files to inspect the observed scene or the outcome of an execution.

## Rules

- Do not read or browse files outside the sandbox.
  This is enforced for the file tools, for Bash, and for `run_python`: commands or code with absolute or `../` paths that leave the sandbox, or that introspect source, are blocked.
  Every `python3` you start carries the same guard, so scripts cannot import or open the hidden modules either; do not edit `PYTHONPATH` or pass `-S`, `-I` or `-E` to python.
- Do not modify files in `./reference/`; they are read-only.
- Do not inspect the predicators source code (`inspect.getsource`, `inspect.getfile`, reading `.py` files from site-packages, or any other route).
  Use the tools and the reference files instead.
- Do not reach into harness internals from executed code.
  `State.privileged`, the probe's `_ctx`, and env flags or attributes are hidden environment ground truth and are blocked.
  Base your conclusions on observable state features and the documented tool surface only; a conclusion derived from hidden internals is an invalid result.
