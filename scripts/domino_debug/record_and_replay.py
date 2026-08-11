#!/usr/bin/env python3
"""Record both ZEDs while replaying a solved plan, for offline domino pose estimation.

Wraps ``replay_plan.py`` (its sibling in this directory) in an SVO recording, and writes an
event log in the SAME clock the video frames carry — so a plan execution produces video the
markerless pose pipeline can consume, with a record of what the robot was doing when.

    # 1. pure sim, cameras rolling (safe; proves the plumbing)
    PYTHONPATH=. python scripts/domino_debug/record_and_replay.py --plan plans/push_only.txt

    # 2. dry arm — the whole RealRobot minus the arm, nothing moves
    PYTHONPATH=. python scripts/domino_debug/record_and_replay.py --plan plan.txt -- \
        --execute --dry

    # 3. MOVES THE ARM
    PYTHONPATH=. python scripts/domino_debug/record_and_replay.py --plan plan.txt -- --execute

Everything after a bare ``--`` is forwarded verbatim to replay_plan.py, so this wrapper never
has an opinion about what the robot does — it only decides when the cameras roll. The default
is replay_plan's own default, which is pure sim. Take the rungs in order, as replay_plan's own
docstring says.

### Why the clocks line up for free

The ZED SDK stamps ``TIME_REFERENCE.IMAGE`` in **Unix-epoch nanoseconds** (verified on the
SDK 3.8.2 install here: a frame written at 16:43:35 carries 1786481015.89). So ``time.time()``
in this process is directly comparable to the per-frame ``timestamp_ns`` in each camera's
``timestamps_<serial>.csv``, with no anchor frame, no offset fitting, and no hardware sync.
Every event logged here is a plain ``time.time()``, and joining video to plan is a subtraction.
That is also why the two cameras need no sync: they free-run, and each frame carries its own
absolute stamp.

Note the timeline is only *physically* meaningful under ``--execute``. In pure sim the rollout
runs faster than real time, so the frames recorded "during the plan" show a static scene.

### Lead-in matters more than it looks

``--lead-in`` records for a few seconds BEFORE the plan starts. Stage 2 of the markerless
pipeline needs one frame with every domino standing, unoccluded, and the arm out of the way —
that frame is the ceiling on everything downstream, and if the recording starts with the arm
already moving there may not be one. Lead-out likewise catches the dominoes settling.

### The recorder lives in the BabyRobotPredicator submodule

``ZedRecorderSession`` comes from ``submodules/BabyRobotPredicator/pose_estimation/
record_zed_video.py``. If the submodule is pinned to a commit before that class existed, this
script says so and exits rather than failing obscurely — point ``--babyrobot`` at a checkout
that has it, or bump the submodule.

Run in `robot-ml` (pyzed + predicators).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))          # scripts/domino_debug -> repo root
REPLAY = os.path.join(_HERE, "replay_plan.py")
DEFAULT_BABYROBOT = os.path.join(_REPO, "submodules", "BabyRobotPredicator")


def _log_line(events, stream_name, text):
    """One timestamped output line, in the same clock the frames carry."""
    events.append({"t": time.time(), "kind": stream_name, "text": text.rstrip("\n")})


def _pump(stream, stream_name, events, echo):
    for raw in iter(stream.readline, ""):
        _log_line(events, stream_name, raw)
        if echo:
            sys.stdout.write(raw)
            sys.stdout.flush()
    stream.close()


def _load_recorder(babyrobot):
    """Import ZedRecorderSession from the BabyRobotPredicator checkout, or explain why not."""
    pose_dir = os.path.join(babyrobot, "pose_estimation")
    if not os.path.isdir(pose_dir):
        raise SystemExit(f"ERROR: no pose_estimation/ under {babyrobot}\n"
                         "       pass --babyrobot <path to a BabyRobotPredicator checkout>")
    sys.path.insert(0, pose_dir)
    try:
        from record_zed_video import DEFAULT_SERIALS, ZedRecorderSession
    except ImportError as exc:
        raise SystemExit(
            f"ERROR: could not import ZedRecorderSession from {pose_dir}: {exc}\n"
            "       That class arrived with the timestamped-capture work. If the submodule is\n"
            "       pinned to an older commit, bump it or point --babyrobot at a checkout that\n"
            "       has it.")
    return DEFAULT_SERIALS, ZedRecorderSession


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, help="plan text file (passed to replay_plan.py)")
    ap.add_argument("--serials", nargs="+", default=None,
                    help="ZED serials (default: the recorder's own default pair)")
    ap.add_argument("--resolution", default="HD720",
                    choices=["VGA", "HD720", "HD1080", "HD2K"])
    ap.add_argument("--fps", type=int, default=None,
                    help="camera fps (default: the max for the resolution)")
    ap.add_argument("--out-dir", default=None, help="parent dir for take folders")
    ap.add_argument("--lead-in", type=float, default=3.0,
                    help="seconds to record BEFORE starting the plan. Stage 2 needs one clean "
                         "frame with every domino standing and the arm clear; do not set this "
                         "to 0")
    ap.add_argument("--lead-out", type=float, default=3.0,
                    help="seconds to keep recording after the plan returns, so the dominoes "
                         "are caught settling")
    ap.add_argument("--babyrobot", default=DEFAULT_BABYROBOT,
                    help="BabyRobotPredicator checkout providing the ZED recorder")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the replay subprocess (default: this one)")
    ap.add_argument("--quiet", action="store_true", help="do not echo the plan's output")
    ap.add_argument("replay_args", nargs=argparse.REMAINDER,
                    help="everything after a bare -- goes to replay_plan.py verbatim")
    args = ap.parse_args()

    forwarded = args.replay_args[1:] if args.replay_args[:1] == ["--"] else args.replay_args
    if not os.path.exists(args.plan):
        print(f"ERROR: plan file not found: {args.plan}", file=sys.stderr)
        return 2

    # replay_plan --observe opens the ZEDs itself to re-perceive between options. We are
    # holding them, so it would fail partway through a run that has already moved the arm.
    # Refuse up front rather than discover it mid-plan.
    if "--observe" in forwarded:
        print("ERROR: --observe makes replay_plan open the cameras, which this script is "
              "already holding.\n       Record OR observe, not both — drop --observe and "
              "recover the poses from the video instead.", file=sys.stderr)
        return 2

    moves_arm = "--execute" in forwarded and "--dry" not in forwarded
    if moves_arm:
        print("=" * 72)
        print("  THIS WILL MOVE THE ARM.  polymetis must already be running on the NUC.")
        print("  Clear the workspace of anything you do not want hit.")
        print("=" * 72, flush=True)

    default_serials, zed_recorder_session = _load_recorder(args.babyrobot)
    session = zed_recorder_session(serials=args.serials or default_serials,
                                   resolution=args.resolution, fps=args.fps,
                                   out_dir=args.out_dir)

    events, take_dir, rc = [], None, 1
    proc = None
    session.open()
    try:
        take_dir = session.start_take()
        _log_line(events, "event", f"take started -> {take_dir}")
        print(f"recording -> {take_dir}", flush=True)

        if args.lead_in > 0:
            _log_line(events, "event", f"lead-in {args.lead_in}s")
            time.sleep(args.lead_in)

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [_REPO, args.babyrobot, env.get("PYTHONPATH", "")]).strip(os.pathsep)
        env.setdefault("PYTHONHASHSEED", "0")
        cmd = [args.python, REPLAY, "--plan", os.path.abspath(args.plan)] + forwarded

        _log_line(events, "event", "plan start: " + " ".join(cmd))
        t_plan0 = time.time()
        # encoding/errors explicitly: `text=True` alone decodes with the locale's preferred
        # encoding, which is ascii here. predicators logs non-ASCII, and one such byte killed
        # the reader thread mid-run — losing exactly the event log this wrapper exists to
        # produce, while the take itself looked fine.
        proc = subprocess.Popen(cmd, cwd=_REPO, env=env,
                                encoding="utf-8", errors="replace",
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pumps = [threading.Thread(target=_pump, args=(s, n, events, not args.quiet), daemon=True)
                 for s, n in ((proc.stdout, "stdout"), (proc.stderr, "stderr"))]
        for t in pumps:
            t.start()
        rc = proc.wait()
        for t in pumps:
            t.join(timeout=5)
        t_plan1 = time.time()
        _log_line(events, "event", f"plan exit rc={rc} after {t_plan1 - t_plan0:.1f}s")
        print(f"\nplan finished rc={rc} in {t_plan1 - t_plan0:.1f}s", flush=True)

        if args.lead_out > 0:
            _log_line(events, "event", f"lead-out {args.lead_out}s")
            time.sleep(args.lead_out)
    except KeyboardInterrupt:
        # Ctrl-C must still produce a usable take: stop the plan, then close the SVOs
        # cleanly, or the recording is truncated garbage and the run is wasted.
        _log_line(events, "event", "KeyboardInterrupt — stopping plan and closing take")
        print("\ninterrupted; stopping plan and closing the take cleanly...", file=sys.stderr)
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        rc = 130
    finally:
        info = None
        try:
            _log_line(events, "event", "take stopping")
            info = session.stop_take()
        finally:
            session.close()

        if take_dir:
            shutil.copy(args.plan, os.path.join(take_dir, "plan.txt"))
            with open(os.path.join(take_dir, "events.json"), "w") as f:
                json.dump({
                    "clock": "unix seconds (time.time()); frame timestamp_ns is the same epoch",
                    "plan_file": os.path.abspath(args.plan),
                    "replay_args": forwarded,
                    "moved_arm": moves_arm,
                    "plan_rc": rc,
                    "events": events,
                }, f, indent=2)
            print(f"\nwrote {take_dir}/events.json and plan.txt")
            if info:
                for serial, st in (info.get("cameras") or {}).items():
                    print(f"  {serial}: {st['frames']} frames, {st['drops']} drops, "
                          f"{st['achieved_fps']:.1f} fps")
            print("\nnext: run the markerless pipeline on a camera's SVO, e.g.")
            print(f"  {args.babyrobot}/pose_estimation/markerless/run_markerless.sh \\\n"
                  f"      {take_dir}/zed_<serial>.svo /tmp/mk/run manual")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
