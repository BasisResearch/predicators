"""Render busyboard init states and annotated demonstration videos.

Doc-asset generator; run from the repo root::

    PYTHONPATH=. python docs/envs/busyboard/render_busyboard_media.py

Every video annotates each frame with the board's HIDDEN state - the true
wiring and each lamp's latent charge - beside the camera view an agent
actually gets. That contrast is the whole point of the domain and is
invisible in a plain recording: for seconds at a time the charge bars
climb while the lamps stay resolutely dark, which is exactly the window
in which a press-and-look agent has no evidence.

Four clips, all on the same four-button, three-lamp board so that
differences between them are attributable to the policy rather than the
board:

* ``oracle_solve`` - press the buttons the goal calls for, then wait.
* ``press_everything_fails`` - latch every button, the policy an agent
  falls into once it knows buttons matter but not which. Lights every
  lamp it can and fails the off-target.
* ``interlock`` - a lamp's driver alone does nothing; the lamp responds
  only once its second button is on too. The many-to-one relation that
  prior busyboard environments for robot learning exclude by design.
* ``delayed_credit`` - the confound. A lamp finishes charging in the
  middle of an unrelated button press, so the press that gets the credit
  is not the press that caused it.
"""

import os
import textwrap
from typing import Any, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from predicators import utils
from predicators.envs.pybullet_busyboard import NO_ENABLER, \
    PyBulletBusyBoardEnv
from predicators.ground_truth_models import get_gt_options
from predicators.structs import EnvironmentTask, State

OUT_DIR = "docs/envs/busyboard"

FRAME_W, FRAME_H = 900, 620
PANEL_W = 460
# Every Nth simulator step becomes a video frame. A button push is ~22
# steps and a lamp needs ~48 driven steps to light, so sampling every
# other step keeps the charge ramp smooth without a 300-frame video.
FRAME_STRIDE = 2
FPS = 12
# Mid-range push parameters: the approach distance and contact height that
# clear the switch body and cross the slider's travel.
PUSH_PARAMS = np.array([0.07, 0.05], dtype=np.float32)

_FONT_DIRS = ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
_MONO_DIRS = ("/System/Library/Fonts/Menlo.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")

INK = (28, 30, 34)
MUTED = (120, 126, 134)
LIT = (219, 158, 20)
DARK = (108, 114, 122)
GOOD = (32, 132, 72)
BAD = (188, 58, 48)
RULE = (214, 216, 220)


def _font(paths: Sequence[str], size: int) -> Any:
    """First available font at ``size``, falling back to PIL's default.

    Returns ``Any`` because PIL ships no stubs for its font constructors,
    and the two of them do not even share a return type.
    """
    for path in paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(
                    path, size)  # type: ignore[no-untyped-call]
            except OSError:
                continue
    return ImageFont.load_default()


# ── Board bookkeeping ────────────────────────────────────────────


def _wiring(task: EnvironmentTask) -> List[Tuple[int, int]]:
    """The task's true (driver, enabler) per lamp, from its oracle metrics."""
    metrics = task.offline_task_metrics
    return [(int(metrics[f"wiring_driver_{i}"]),
             int(metrics[f"wiring_enabler_{i}"]))
            for i in range(int(metrics["wiring_num_lamps"]))]


def _target(task: EnvironmentTask, num_lamps: int) -> List[bool]:
    """The goal's per-lamp on/off assignment, in board order."""
    want = {
        atom.objects[0].name: atom.predicate.name == "LampOn"
        for atom in task.goal_description
    }
    return [want[f"lamp{i}"] for i in range(num_lamps)]


def _solving_buttons(env: PyBulletBusyBoardEnv, task: EnvironmentTask,
                     num_buttons: int) -> List[int]:
    """The button set that realizes the goal exactly.

    Exhaustive over the 2**num_buttons settings - this is the oracle, so
    it reads the answer off the true wiring rather than inferring it.
    """
    wiring = _wiring(task)
    target = _target(task, len(wiring))
    for mask in range(1 << num_buttons):
        on = [bool(mask >> b & 1) for b in range(num_buttons)]
        if all(
                env._driven(on, d, e) == t  # pylint: disable=protected-access
                for (d, e), t in zip(wiring, target)):
            return [b for b in range(num_buttons) if on[b]]
    raise RuntimeError("Goal is unrealizable, which task generation forbids.")


def _condition_str(driver: int, enabler: int) -> str:
    """Human-readable drive condition, e.g. ``button1 AND button2``."""
    if enabler == NO_ENABLER:
        return f"button{driver}"
    lo, hi = min(driver, enabler), max(driver, enabler)
    return f"button{lo} AND button{hi}"


# ── Annotation panel ─────────────────────────────────────────────


def _draw_panel(env: PyBulletBusyBoardEnv, state: State, task: EnvironmentTask,
                caption: str, verdict: Optional[str],
                note: Optional[str]) -> Image.Image:
    """The right-hand panel: goal, true wiring, and every hidden charge."""
    h1 = _font(_FONT_DIRS, 21)
    h2 = _font(_FONT_DIRS, 15)
    body = _font(_FONT_DIRS, 15)
    mono = _font(_MONO_DIRS, 14)
    small = _font(_FONT_DIRS, 13)

    panel = Image.new("RGB", (PANEL_W, FRAME_H), (252, 252, 250))
    draw = ImageDraw.Draw(panel)
    x0, y = 26, 24

    draw.text((x0, y), caption, font=h1, fill=INK)
    y += 34
    draw.line((x0, y, PANEL_W - 26, y), fill=RULE, width=1)
    y += 20

    buttons = sorted((o for o in state if o.type.name == "button"),
                     key=lambda o: o.name)
    lamps = sorted((o for o in state if o.type.name == "lamp"),
                   key=lambda o: o.name)
    wiring = _wiring(task)
    target = _target(task, len(lamps))

    draw.text((x0, y), "GOAL", font=h2, fill=MUTED)
    y += 22
    for lamp, want in zip(lamps, target):
        draw.text((x0 + 6, y),
                  f"{lamp.name}: {'LIT' if want else 'DARK'}",
                  font=body,
                  fill=LIT if want else DARK)
        y += 20
    y += 10

    draw.text((x0, y), "HIDDEN WIRING  (never observed)", font=h2, fill=MUTED)
    y += 22
    for i, (driver, enabler) in enumerate(wiring):
        draw.text((x0 + 6, y),
                  f"lamp{i} <- {_condition_str(driver, enabler)}",
                  font=mono,
                  fill=INK)
        y += 19
    driven_any = {b for d, e in wiring for b in (d, e) if b != NO_ENABLER}
    decoys = [b.name for i, b in enumerate(buttons) if i not in driven_any]
    if decoys:
        draw.text((x0 + 6, y),
                  f"decoys: {', '.join(decoys)}",
                  font=small,
                  fill=MUTED)
        y += 19
    y += 12

    draw.text((x0, y), "BUTTONS", font=h2, fill=MUTED)
    y += 22
    bx = x0 + 6
    for button in buttons:
        on = state.get(button, "is_on") > 0.5
        draw.rectangle((bx, y, bx + 20, y + 20),
                       fill=(70, 76, 84) if on else (226, 228, 232),
                       outline=RULE)
        draw.text((bx + 2, y + 24), button.name[-1], font=small, fill=MUTED)
        bx += 30
    y += 48

    # Hidden charge beside its observable readout. The gap between the two
    # IS this domain's partial observability.
    draw.text((x0, y),
              "HIDDEN CHARGE  ->  WHAT THE ROBOT SEES",
              font=h2,
              fill=MUTED)
    y += 24
    bar_w = PANEL_W - 2 * x0 - 130
    for i, lamp in enumerate(lamps):
        charge = float(lamp.charge or 0.0)
        brightness = float(state.get(lamp, "brightness"))
        draw.text((x0, y + 1), f"lamp{i}", font=small, fill=INK)
        draw.rectangle((x0 + 48, y, x0 + 48 + bar_w, y + 13),
                       fill=(233, 235, 238),
                       outline=RULE)
        draw.rectangle(
            (x0 + 48, y, x0 + 48 + max(1, int(bar_w * charge)), y + 13),
            fill=(96, 132, 196))
        onset_x = x0 + 48 + int(bar_w * env.BRIGHTNESS_ONSET)
        draw.line((onset_x, y - 2, onset_x, y + 15), fill=(160, 60, 60))
        swatch = tuple(
            int((1 - brightness) * d + brightness * l)
            for d, l in zip((150, 154, 160), LIT))
        draw.rectangle((x0 + 60 + bar_w, y, x0 + 84 + bar_w, y + 13),
                       fill=swatch,
                       outline=RULE)
        y += 26
    draw.text((x0 + 48, y - 4),
              "red mark = onset; left of it the lamp looks dead",
              font=small,
              fill=MUTED)
    y += 24

    if verdict is not None:
        ok = verdict.startswith("SOLVED")
        draw.line((x0, y, PANEL_W - 26, y), fill=RULE, width=1)
        draw.text((x0, y + 14), verdict, font=h1, fill=GOOD if ok else BAD)

    if note is not None:
        # Anchored to the panel foot rather than flowing after the bars, so
        # the explanation sits in the same place in every clip.
        lines = textwrap.wrap(note, width=44)
        box_h = 20 + 20 * len(lines)
        top = FRAME_H - box_h - 22
        draw.rectangle((x0 - 12, top, PANEL_W - 16, top + box_h),
                       fill=(245, 240, 224),
                       outline=(224, 210, 170))
        for i, line in enumerate(lines):
            draw.text((x0, top + 10 + 20 * i), line, font=body, fill=INK)

    return panel


def _frame(env: PyBulletBusyBoardEnv,
           task: EnvironmentTask,
           caption: str,
           verdict: Optional[str] = None,
           note: Optional[str] = None) -> np.ndarray:
    """One annotated video frame: camera view plus the hidden-state panel."""
    view = Image.fromarray(
        env.render()[0].astype("uint8"))  # type: ignore[no-untyped-call]
    panel = _draw_panel(env, env._get_state(), task, caption, verdict, note)  # pylint: disable=protected-access
    canvas = Image.new("RGB", (FRAME_W + PANEL_W, FRAME_H), (252, 252, 250))
    canvas.paste(view, (0, 0))
    canvas.paste(panel, (FRAME_W, 0))
    return np.asarray(canvas)


# ── Scripted trajectories ────────────────────────────────────────

# A clip is a list of steps, each one of:
#   ("press", button_idx)   push a button to on
#   ("release", button_idx) push it back to off
#   ("wait", num_steps)     hold still while the board evolves
#   ("note", text, frames)  freeze and show an explanation
Step = Tuple[Any, ...]


def _run_clip(env: PyBulletBusyBoardEnv, task: EnvironmentTask, split: str,
              task_idx: int, steps: Sequence[Step],
              verdict_at_end: bool) -> Tuple[List[np.ndarray], bool]:
    """Play a scripted clip, returning annotated frames."""
    options = {o.name: o for o in get_gt_options("pybullet_busyboard")}
    env.reset(split, task_idx)
    state = env._get_state()  # pylint: disable=protected-access
    robot = next(o for o in state if o.type.name == "robot")
    buttons = sorted((o for o in state if o.type.name == "button"),
                     key=lambda o: o.name)

    frames: List[np.ndarray] = []
    note: Optional[str] = None
    step_count = 0

    for step in steps:
        kind = step[0]
        if kind == "note":
            text, hold = step[1], step[2]
            note = text
            frames.extend([_frame(env, task, "", None, note)] * hold)
            continue
        if kind in ("press", "release"):
            button_idx = step[1]
            name = "PressButton" if kind == "press" else "ReleaseButton"
            caption = f"{kind} button{button_idx}"
            option = options[name].ground([robot, buttons[button_idx]],
                                          PUSH_PARAMS)
            option.initiable(state)
            for _ in range(300):
                state = env.step(option.policy(state))
                step_count += 1
                if step_count % FRAME_STRIDE == 0:
                    frames.append(_frame(env, task, caption, None, note))
                if option.terminal(state):
                    break
            continue
        assert kind == "wait"
        wait = options["Wait"].ground([robot], np.array([], dtype=np.float32))
        wait.initiable(state)
        for _ in range(step[1]):
            state = env.step(wait.policy(state))
            step_count += 1
            if step_count % FRAME_STRIDE == 0:
                frames.append(_frame(env, task, "wait", None, note))

    solved = env.goal_reached()
    if verdict_at_end:
        verdict = "SOLVED" if solved else "FAILED"
        frames.extend([_frame(env, task, "done", verdict, note)] * (2 * FPS))
    else:
        frames.extend([frames[-1]] * FPS)
    return frames, solved


def _write(name: str, frames: List[np.ndarray]) -> None:
    """Encode one clip to ``OUT_DIR``."""
    path = os.path.join(OUT_DIR, f"{name}.mp4")
    # Widened rather than ignored: imageio's writer signature does not
    # admit a plain list of ndarrays.
    video: Any = frames
    imageio.mimwrite(path, video, fps=FPS, quality=8)
    print(f"wrote {name}.mp4 ({len(frames)} frames, "
          f"{len(frames) / FPS:.1f}s)")


def _main() -> None:
    utils.reset_config({
        "env": "pybullet_busyboard",
        "seed": 0,
        "num_train_tasks": 1,
        "num_test_tasks": 3,
        "pybullet_camera_width": FRAME_W,
        "pybullet_camera_height": FRAME_H,
    })
    os.makedirs(OUT_DIR, exist_ok=True)
    env = PyBulletBusyBoardEnv(use_gui=False)
    env._generate_train_tasks()  # pylint: disable=protected-access
    env._generate_test_tasks()  # pylint: disable=protected-access

    for split, idx in (("train", 0), ("test", 0), ("test", 1)):
        env.reset(split, idx)
        num_buttons = env._num_active_buttons  # pylint: disable=protected-access
        num_lamps = env._num_active_lamps  # pylint: disable=protected-access
        name = f"init_{num_buttons}buttons_{num_lamps}lamps.png"
        imageio.imwrite(os.path.join(OUT_DIR, name),
                        env.render()[0].astype("uint8"))
        print("wrote", name)

    split, idx = "test", 0
    task = env.get_task(split, idx)
    env.reset(split, idx)
    num_buttons = env._num_active_buttons  # pylint: disable=protected-access
    wiring = _wiring(task)
    solving = _solving_buttons(env, task, num_buttons)
    print(f"board {split}[{idx}]: {num_buttons} buttons, wiring {wiring}, "
          f"goal {_target(task, len(wiring))}, solving press set {solving}")

    # The two buttons lamp0's drive condition needs, and one that is
    # irrelevant to it - the cast for the interlock and confound clips.
    lamp0_driver, lamp0_enabler = wiring[0]
    irrelevant = next(b for b in range(num_buttons)
                      if b not in (lamp0_driver, lamp0_enabler))

    clips: List[Tuple[str, List[Step], bool]] = [
        ("oracle_solve", [("note", "The goal needs two lamps lit and one "
                           "dark. Press exactly the buttons their drive "
                           "conditions call for.", 2 * FPS)] +
         [("press", b) for b in solving] + [("wait", 110)], True),
        ("press_everything_fails",
         [("note",
           "Knowing that buttons matter but not which, latch them all.",
           2 * FPS)] + [("press", b)
                        for b in range(num_buttons)] + [("wait", 110)], True),
        ("interlock", [
            ("note", f"lamp0 needs button{lamp0_driver} AND "
             f"button{lamp0_enabler}. Press just one of them.", 2 * FPS),
            ("press", lamp0_enabler),
            ("wait", 90),
            ("note", "Nothing. Not a slow response - the charge bar never "
             "left zero, so the drive condition was never met.", 3 * FPS),
            ("press", lamp0_driver),
            ("wait", 120),
            ("note", "With both on, the charge climbs and lamp0 lights. "
             "This conjunction is the relation prior busyboard "
             "benchmarks exclude by design.", 3 * FPS),
        ], False),
        (
            "delayed_credit",
            [
                ("note", f"Press button{lamp0_driver}, then "
                 f"button{lamp0_enabler}: lamp0's condition is now met and it "
                 "starts charging invisibly.", 2 * FPS),
                ("press", lamp0_driver),
                ("press", lamp0_enabler),
                ("wait", 26),
                # Tenseless on purpose: this note is still on screen at the
                # moment lamp0 lights, so "nothing is visible yet" would be
                # false exactly when the viewer is looking at it.
                ("note",
                 f"button{irrelevant} is unrelated to lamp0. Watch the "
                 "board during its press.", 2 * FPS),
                ("press", irrelevant),
                ("note",
                 f"lamp0 lit during the button{irrelevant} press - but "
                 f"button{irrelevant} had nothing to do with it. This is the "
                 "mis-attribution a press-and-look agent makes.", 4 * FPS),
                ("wait", 90),
            ],
            False),
    ]

    for name, steps, verdict in clips:
        frames, solved = _run_clip(env, task, split, idx, steps, verdict)
        _write(name, frames)
        if verdict:
            print(f"  {name}: solved={solved}")


if __name__ == "__main__":
    _main()
