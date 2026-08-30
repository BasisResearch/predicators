# Predicators Agent Sandbox

## Working Directory

Your working directory is the sandbox. All files you create must stay here, and every path you use is relative (for example `./my_script.py`).

## Python

The interpreter is `python3`, and the predicators package is importable:

    python3 -c "from predicators.structs import State, Type; print('OK')"

Write and run scripts in the sandbox:

    python3 my_experiment.py

## Reference Files

Curated source files are in `./reference/`. Read them to learn the APIs before writing code.

## Session Logs

Your earlier queries and tool results are in `./session_logs/`, named `<NNN>_<kind>_<timestamp>.md` where `<NNN>` is a run-wide counter and `<kind>` is the query phase (`learn`, `test`, `explore`). Alphabetical order is chronological order:

    Glob ./session_logs/*.md
    Read ./session_logs/001_learn_*.md

## Scene Images

`submit_plan` saves a rendering of the scene after every step to `./test_images/`. Read them to inspect the spatial state.

## Rules

- Do not read or browse files outside the sandbox. This is enforced for the file tools, for Bash, and for `run_python`: commands or code with absolute or `../` paths that leave the sandbox, or that introspect source, are blocked.
- Do not modify files in `./reference/`; they are read-only.
- Do not inspect the predicators source code (`inspect.getsource`, `inspect.getfile`, reading `.py` files from site-packages, or any other route). Use the tools and the reference files instead.
- Do not reach into harness internals from executed code. `State.privileged`, the probe's `_ctx`, and env flags or attributes are hidden environment ground truth and are blocked. Learn the world model from observable state features and the documented tool surface only; a conclusion derived from hidden internals is an invalid result.
