"""Rebuilding an episode's task from a short markerless take.

The live ``"zed"`` perception is the *marker* pipeline, and the 20 mm ArUco
markers are not resolvable at this camera distance -- 1 of ~7 on one camera and
0 on the other -- so "look at the scene between episodes" cannot be served that
way on this bench. This module serves it the way the markerless pipeline does:
record a short take, fit the known domino box to the masked depth, and read the
scene JSON that falls out.

**Why this does not fight the episode recorder for the cameras.** It does not
open any. A ZED admits one owner, and the owner is the recorder's session; a
snapshot is simply a second, short take on that same open session, taken while
no episode take is running. That is the whole reason this exists rather than a
second capture process.

**Where it plugs in.** ``RealRobot.reset_env`` homes the arm, blocks until a
human confirms the scene is arranged, and only then calls
``perception.observe()``. That last call is exactly the moment a snapshot
should be taken, so this class duck-types babyrobot's perception protocol
(``open`` / ``observe`` / ``close``) and is injected through
``make_real_robot(perception=...)``. Nothing in the reset flow changes.

babyrobot is optional and must never be imported at module level, as in
``real_robot_bridge``.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

from predicators.settings import CFG

_MISSING_BABYROBOT = (
    "markerless snapshot rebuild needs the private BabyRobotPredicator "
    "package, which predicators carries as the git submodule "
    "submodules/BabyRobotPredicator. Check it out and install it:\n"
    "    git submodule update --init submodules/BabyRobotPredicator\n"
    "    pip install -e submodules/BabyRobotPredicator")

# Stages 1-4 over one recording: (svo, bundle, camera_serial) -> scene JSON.
StageRunner = Callable[[str, str, str], str]


class MissingBabyRobotError(ImportError):
    """babyrobot is not importable, so no snapshot can be fitted."""


class MarkerlessSnapshotPerception:
    """A scene look served by a short take plus the markerless pipeline.

    Duck-types babyrobot's perception protocol so ``RealRobot`` can hold
    one of these wherever it would hold a ``DominoPerception``.

    ``open`` and ``close`` are deliberately no-ops: this object owns no
    cameras. Taking them here is what would collide with the episode
    recorder, and the collision is the thing being avoided.

    The pipeline runner and the scene loader are injectable so the whole
    lifecycle is testable without a GPU, a camera, or the submodule.
    """

    def __init__(self,
                 recorder: Any,
                 serial: str,
                 runner: Optional[StageRunner] = None,
                 scene_loader: Optional[Callable[[str], Any]] = None,
                 work_dir: Optional[str] = None,
                 frames: int = 5) -> None:
        self._recorder = recorder
        self._serial = str(serial)
        self._runner = runner
        self._scene_loader = scene_loader
        self._work_dir = work_dir or os.path.join("logs", "zed_snapshots")
        self._frames = int(frames)
        # Snapshots fitted this run, newest last. Read by tests, and worth
        # having in a hardware session's log.
        self.scenes: list[str] = []

    @property
    def has_perception(self) -> bool:
        """``RealRobot`` asks this before allowing a look."""
        return True

    def open(self) -> None:
        """No cameras of our own to open; the recorder owns them."""

    def close(self) -> None:
        """No cameras of our own to close."""

    def observe(self, settle_s: float = 0.0) -> Any:
        """Take a snapshot, fit it, and return the scene as an observation.

        ``settle_s`` is honoured by the take itself rather than by a
        sleep: the frames are grabbed after the human has stepped away
        and the arm is already home, so there is nothing left moving to
        settle for.
        """
        del settle_s  # nothing is in motion at a scene reset
        take_dir, svo = self._record_snapshot()
        bundle = os.path.join(self._work_dir, os.path.basename(take_dir))
        scene_json = self._run_stages(svo, bundle)
        self.scenes.append(scene_json)
        logging.info("real robot: snapshot scene fitted to %s", scene_json)
        return self._load_scene(scene_json)

    # -- helpers -----------------------------------------------------------
    def _record_snapshot(self) -> tuple[str, str]:
        """Take a short take on the recorder's session; return (dir, svo)."""
        take_dir, meta = self._recorder.snapshot(frames=self._frames)
        ext = str((meta or {}).get("svo_ext", ".svo2"))
        svo = os.path.join(take_dir, f"zed_{self._serial}{ext}")
        if not os.path.exists(svo):
            raise FileNotFoundError(
                f"the snapshot take {take_dir} has no recording for ZED "
                f"{self._serial} (expected {os.path.basename(svo)}); the "
                "camera the scene is fitted from must be one the recorder "
                "was given")
        return take_dir, svo

    def _run_stages(self, svo: str, bundle: str) -> str:
        """Fit the take, through the injected runner or the real pipeline."""
        if self._runner is not None:
            return self._runner(svo, bundle, self._serial)
        return _default_runner(svo, bundle, self._serial)

    def _load_scene(self, scene_json: str) -> Any:
        """Read the fitted scene as an observation."""
        if self._scene_loader is not None:
            return self._scene_loader(scene_json)
        # pylint: disable=import-outside-toplevel,import-error
        try:
            from babyrobot.realrobot.perception import FileDominoPerception
        except ImportError as e:
            raise MissingBabyRobotError(_MISSING_BABYROBOT) from e
        # Both pipelines write the same scene JSON, which is why the loader
        # that replays a captured scene reads a freshly fitted one unchanged.
        return FileDominoPerception(scene_json).observe(0.0)


def _default_runner(svo: str, bundle: str, serial: str) -> str:
    """Run markerless stages 1-4 over ``svo``, returning the scene JSON.

    Delegates to the submodule's own staging rather than re-deriving the
    argv for four scripts: it already resolves the interpreter the
    stages need (they want pyzed and ultralytics together, which is not
    the env running predicators) and reports a stage failure by name.
    """
    # pylint: disable=import-outside-toplevel,import-error
    try:
        from babyrobot.scene.capture_markerless import MarkerlessCapture, \
            load_boxes, resolve_python, run_stages
    except ImportError as e:
        raise MissingBabyRobotError(_MISSING_BABYROBOT) from e
    # run_stages reads ``boxes``, not ``boxes_json`` -- resolving the file is
    # the caller's job, and doing it here is what makes the rebuild unattended.
    boxes_json = CFG.real_robot_snapshot_boxes_json or None
    boxes = tuple(load_boxes(boxes_json)) if boxes_json else None
    if boxes is None:
        logging.warning(
            "real robot: no real_robot_snapshot_boxes_json, so stage 2 will "
            "open the drag window and wait for a human to draw one box per "
            "domino -- every episode. Point it at an earlier run's boxes.json "
            "to make the rebuild unattended.")
    config = MarkerlessCapture(
        camera=serial,
        boxes=boxes,
        frames=CFG.real_robot_snapshot_frames,
        resolution=CFG.real_robot_recording_resolution,
        # The twin's base->world transplant and the fit have to agree on where
        # the table is, and they are configured separately -- so the fit is
        # told, rather than left to measure its own.
        table_z=float(CFG.domino_real_table_z),
        z_mode=CFG.real_robot_snapshot_z_mode,
        viz=CFG.real_robot_snapshot_viz)
    return run_stages(resolve_python(), svo, bundle, config)
