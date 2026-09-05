#!/usr/bin/env python3
"""Build the labelled video of a finished continual-protocol run offline.

The video replays the run's recorded actions through the env, so the
env must be configured exactly as the run was: the same env flags decide
its tasks and physics. It lands beside the run's other videos, at
``videos/<the run's log subdir>/run.mp4``, and its path is written to the
run's scorecard for the viewer. Two ways to name the run:

* From the run's log. ``info.log`` starts with the ``main.py`` command
  the run was launched with; this script re-parses it::

      python scripts/continual_video.py \\
          --run_log logs/agent_continual/<exp>/seed0/run_<stamp>/info.log

* Explicitly, with the run's own ``main.py`` flags (``--env``,
  ``--approach``, ``--experiment_id``, ``--seed`` and the env flags)::

      python scripts/continual_video.py --env pybullet_boil \\
          --approach agent_continual --experiment_id boil-agent_continual \\
          --seed 0 [env flags...]

Either way, flags after ``--run_log`` override the log's, so
``--continual_video_stride 2`` halves the frame count and
``--video_fps 30`` speeds the playback. ``--out`` picks another file.

See ``predicators/run/continual_video.py`` for what the frames show.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
from predicators import utils  # noqa: E402
from predicators.envs import create_new_env  # noqa: E402
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


def _split_own_args(argv: List[str]) -> tuple:
    """Pull this script's own options out of ``argv``; the rest are predicators
    flags."""
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--run_log", default=None)
    own.add_argument("--out", default=None)
    own_args, rest = own.parse_known_args(argv)
    return own_args, rest


def main(argv: Optional[List[str]] = None) -> str:
    """Build the video; returns its path."""
    own, rest = _split_own_args(sys.argv[1:] if argv is None else argv)
    flags = command_flags_from_log(own.run_log) if own.run_log else []
    # The log's flags first, the command line's after, so the latter win.
    sys.argv = [sys.argv[0], *flags, *rest]
    args = utils.parse_args()
    utils.update_config(args)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s",
                        force=True)
    run_id = utils.get_config_path_str()
    card_path = os.path.join(CFG.continual_scorecards_dir, f"{run_id}.json")
    card = RunCard.load(card_path)
    recordings_dir = os.path.join(CFG.continual_recordings_dir, run_id)
    # The log's directory is the run's log subdir, which the video dir
    # mirrors (utils.video_run_dir); without a log there is nothing to
    # mirror and the video goes under videos/continual/<run_id>/.
    run_subdir = None
    if own.run_log:
        log_root = os.path.normpath(str(CFG.log_file or "logs"))
        run_subdir = os.path.relpath(os.path.dirname(own.run_log), log_root)
        if run_subdir.startswith(".."):
            run_subdir = None
    env = create_new_env(CFG.env, do_cache=False, use_gui=False)
    try:
        path = make_run_video(env,
                              card,
                              recordings_dir,
                              out_path=own.out,
                              card_path=card_path,
                              run_subdir=run_subdir)
    finally:
        env.dispose()
    if path is None:
        raise SystemExit(f"{run_id}: no attempted level with a recording")
    print(path)
    return path


if __name__ == "__main__":
    main()
