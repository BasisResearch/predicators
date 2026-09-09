#!/usr/bin/env python3
"""Build the labelled video of a finished continual-protocol run offline.

A run is one directory (``predicators/run/paths.py``); the video
replays the recordings in it through the env and lands beside them as
``run.mp4``. The env must be configured exactly as the run was, since
its flags decide the tasks and the physics: the run's ``info.log``
starts with the ``main.py`` command it was launched with, and this
script re-parses it::

    python scripts/continual_video.py \\
        --run_dir logs/agent_continual/<exp>/seed0/run_<stamp>

Flags after ``--run_dir`` override the log's, so
``--continual_video_stride 2`` halves the frame count and
``--video_fps 30`` speeds the playback; ``--out`` puts the file
elsewhere. A run directory without an ``info.log`` (a run that logged
nowhere) needs the run's own ``main.py`` flags on the command line.

See ``predicators/run/continual_video.py`` for what the frames show.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
from predicators import utils  # noqa: E402
from predicators.envs import create_new_env  # noqa: E402
from predicators.run import paths  # noqa: E402
from predicators.run.continual_video import make_run_video  # noqa: E402
from predicators.run.scorecard import RunCard  # noqa: E402
from predicators.settings import CFG  # noqa: E402

_COMMAND_RE = re.compile(r"Running command: (.*?main\.py\s+.*?)(?:\x1b\[|$)")


def command_flags_from_log(log_path: str) -> List[str]:
    """The ``main.py`` flags of the run whose ``info.log`` this is."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _COMMAND_RE.search(line)
            if m is None:
                continue
            tokens = shlex.split(m.group(1))
            idx = next(i for i, t in enumerate(tokens)
                       if t.endswith("main.py"))
            return tokens[idx + 1:]
    raise ValueError(f"no 'Running command: ... main.py' line in {log_path}")


def _split_own_args(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    """Pull this script's own options out of ``argv``; the rest are predicators
    flags."""
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--run_dir", required=True)
    own.add_argument("--out", default=None)
    return own.parse_known_args(argv)


def main(argv: Optional[List[str]] = None) -> str:
    """Build the video; returns its path."""
    own, rest = _split_own_args(sys.argv[1:] if argv is None else argv)
    run_dir = os.path.normpath(own.run_dir)
    log = os.path.join(run_dir, "info.log")
    flags = command_flags_from_log(log) if os.path.isfile(log) else []
    # The log's flags first, the command line's after, so the latter win.
    sys.argv = [sys.argv[0], *flags, *rest]
    args = utils.parse_args()
    utils.update_config(args)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s",
                        force=True)
    card = RunCard.load(paths.scorecard_path(run_dir))
    env = create_new_env(CFG.env, do_cache=False, use_gui=False)
    try:
        path = make_run_video(env, card, run_dir, out_path=own.out)
    finally:
        env.dispose()
    if path is None:
        raise SystemExit(f"{run_dir}: no attempted level with a recording")
    print(path)
    return path


if __name__ == "__main__":
    main()
