"""Turning an episode's recording into a pose track, off the critical path.

Step 2 records a take per episode; Step 3 scores against a track. This module
is the bridge: it runs the markerless pipeline over each take and writes the
``dominoes_traj.json`` the fit reads.

**Why it runs in the background and not inline.** The pipeline takes minutes --
roughly 3x the length of the take -- so waiting for it inside the episode loop
would serialise post-processing into execution and undo open-loop batching. It
is launched as a detached process when the take closes and joined at the end of
the run, which overlaps it with the next episode's human scene reset and arm
motion. It parallelises across takes at ~2 GB of a 24 GB card, so several
episodes in flight is the intended state rather than a hazard.

**Why a manifest.** A learning cycle produces many takes, and the fit has to
know which track belongs to which episode -- and which episodes had a camera
drop out and should not be trusted at all. The manifest is written as each take
closes, so it is complete and readable even if the run is killed before the
pipeline finishes; entries point at tracks that may not exist yet, and the
reader treats a missing one as "not ready" rather than as an error.

babyrobot is optional and must never be imported at module level.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

from predicators.settings import CFG

# Written by the recorder, read by the fit. One per run.
MANIFEST_NAME = "tracks.json"
# Stage 4's own output name, which is what the fit loads.
TRACK_NAME = "dominoes_traj.json"


class MarkerlessTrackProcessor:
    """Launches the markerless pipeline over finished takes.

    One process per take, started when the take closes and never waited
    on until the run ends. The script is the submodule's own
    ``run_markerless.sh``, which is the same staging a human runs by
    hand -- so a track produced here and one produced at the terminal
    are the same artifact.
    """

    def __init__(self,
                 script: Optional[str] = None,
                 boxes_json: Optional[str] = None,
                 z_mode: str = "contact",
                 max_frames: Optional[int] = None,
                 launcher: Any = None) -> None:
        self._script = script or _default_script()
        self._boxes_json = boxes_json
        self._z_mode = z_mode
        self._max_frames = max_frames
        # Injectable so tests drive the whole lifecycle without a GPU: takes
        # (argv, env) and returns something with poll()/wait().
        self._launcher = launcher or _popen
        self._jobs: List[Any] = []

    def launch(self, svo: str, bundle: str, serial: str) -> Optional[Any]:
        """Start the pipeline over ``svo``, writing into ``bundle``.

        Returns the handle, or None when the pipeline could not be
        started. A failure to launch is logged and swallowed: the take
        is already on disk and can be processed by hand, so it must not
        take the run down with it.
        """
        if not os.path.exists(self._script):
            logging.error(
                "cannot post-process %s: no markerless driver at %s. The "
                "take is recorded and can be processed by hand.", svo,
                self._script)
            return None
        # ``given`` rather than ``manual``: stage 2 would otherwise open a
        # drag window and wait for a human, in the middle of a learning run.
        argv = [self._script, svo, bundle, "given", "--z-mode", self._z_mode]
        env = dict(os.environ)
        env["SERIAL"] = str(serial)
        if self._max_frames:
            env["MAX_FRAMES"] = str(self._max_frames)
        boxes = _read_boxes(self._boxes_json)
        if boxes is None:
            logging.error(
                "cannot post-process %s: stage 2 needs initialization boxes "
                "and real_robot_snapshot_boxes_json is unset or unreadable. "
                "Without them the pipeline would stop for a human.", svo)
            return None
        env["BOXES"] = boxes
        os.makedirs(bundle, exist_ok=True)
        try:
            job = self._launcher(argv, env)
        except OSError as e:
            logging.error("could not start the markerless pipeline on %s: %s",
                          svo, e)
            return None
        logging.info("post-processing %s -> %s (in the background)", svo,
                     os.path.join(bundle, TRACK_NAME))
        self._jobs.append(job)
        return job

    def pending(self) -> int:
        """How many pipeline jobs are still running."""
        return sum(1 for j in self._jobs if j.poll() is None)

    def wait_all(self, timeout: Optional[float] = None) -> None:
        """Block until every launched job has finished.

        Called at run teardown. A job that fails is logged rather than
        raised: by this point the run is over, and the takes survive
        for a re-run by hand.
        """
        for job in self._jobs:
            try:
                code = job.wait(timeout=timeout)
            except Exception as e:  # pylint: disable=broad-except
                logging.error("waiting on a post-processing job failed: %s", e)
                continue
            if code not in (0, None):
                logging.error(
                    "a markerless post-processing job exited %s; its track "
                    "will be missing and that episode will be skipped", code)


def _popen(argv: List[str], env: Dict[str, str]) -> Any:
    """Start a detached pipeline process."""
    return subprocess.Popen(  # pylint: disable=consider-using-with
        argv,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT)


def _read_boxes(boxes_json: Optional[str]) -> Optional[str]:
    """The prompt boxes as the JSON string the driver's BOXES expects."""
    if not boxes_json or not os.path.exists(boxes_json):
        return None
    try:
        with open(boxes_json, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    boxes = raw.get("boxes") if isinstance(raw, dict) else raw
    if not boxes:
        return None
    return json.dumps(boxes)


def _default_script() -> str:
    """Where the markerless driver lives inside the submodule."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(root, "submodules", "BabyRobotPredicator",
                        "pose_estimation", "markerless", "run_markerless.sh")


def make_track_processor() -> MarkerlessTrackProcessor:
    """Build the processor the config asks for."""
    return MarkerlessTrackProcessor(
        boxes_json=CFG.real_robot_snapshot_boxes_json or None,
        z_mode=CFG.real_robot_track_z_mode,
        max_frames=CFG.real_robot_recording_max_frames or None)


def write_manifest(path: str, episodes: List[Dict[str, Any]]) -> None:
    """Write the run manifest, replacing whatever was there.

    Rewritten in full after every episode rather than appended to, so a
    run killed mid-way leaves a valid JSON document rather than a
    truncated one.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"episodes": episodes}, f, indent=2)
        f.write("\n")


def read_manifest(path: str) -> List[Dict[str, Any]]:
    """The manifest's episode entries, newest last."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    episodes = raw.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{path} has no episodes list")
    return episodes
