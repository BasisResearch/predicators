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

Task cards (``task_<split>_<idx>.png``): the initial board of one task
with its goal and the oracle's press set written underneath, for a spread
of seed-0 train and test tasks that covers every goal type and board size.
"""

import os
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from predicators import utils
from predicators.envs.pybullet_busyboard import NO_ENABLER, \
    PyBulletBusyBoardEnv, canonical_wiring, core_board
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG
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

# matplotlib ships DejaVu with the package, so its copy is the one path
# that exists on every machine the script runs on.
_MPL_FONTS = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
_FONT_DIRS = ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              os.path.join(_MPL_FONTS, "DejaVuSans.ttf"))
_MONO_DIRS = ("/System/Library/Fonts/Menlo.ttc",
              "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              os.path.join(_MPL_FONTS, "DejaVuSansMono.ttf"))

INK = (28, 30, 34)
MUTED = (120, 126, 134)
LIT = (219, 158, 20)
DARK = (108, 114, 122)
GOOD = (32, 132, 72)
BAD = (188, 58, 48)
RULE = (214, 216, 220)


def _font(paths: Sequence[str], size: int) -> Any:
    """First available font at ``size``, falling back to PIL's default.

    Returns ``Any`` because PIL ships no stubs for its font
    constructors, and the two of them do not even share a return type.
    """
    for path in paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(
                    path, size)  # type: ignore[no-untyped-call]
            except OSError:
                continue
    return ImageFont.load_default(size)


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


def _bname(button_idx: int) -> str:
    """Compact button name for the panel, e.g. ``red b0``."""
    env_cls = PyBulletBusyBoardEnv
    return (f"{env_cls.color_name(env_cls.button_color_index(button_idx))} "
            f"b{button_idx}")


def _lname(lamp_idx: int) -> str:
    """Compact lamp name for the panel, e.g. ``yellow lamp0``."""
    env_cls = PyBulletBusyBoardEnv
    return (f"{env_cls.color_name(env_cls.lamp_color_index(lamp_idx))} "
            f"lamp{lamp_idx}")


def _condition_str(driver: int, enabler: int) -> str:
    """Human-readable drive condition, e.g. ``green b1 AND blue b2``."""
    if enabler == NO_ENABLER:
        return _bname(driver)
    lo, hi = min(driver, enabler), max(driver, enabler)
    return f"{_bname(lo)} AND {_bname(hi)}"


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
                  f"{_lname(i)} <- {_condition_str(driver, enabler)}",
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
    # Room for "magenta" at the small size before the bar starts.
    label_w = 70
    bar_w = PANEL_W - 2 * x0 - 130
    for i, lamp in enumerate(lamps):
        charge = env._charges.get(lamp.name, 0.0)  # pylint: disable=protected-access
        brightness = float(state.get(lamp, "brightness"))
        env_cls = PyBulletBusyBoardEnv
        draw.text((x0, y + 1),
                  env_cls.color_name(env_cls.lamp_color_index(i)),
                  font=small,
                  fill=INK)
        draw.rectangle((x0 + label_w, y, x0 + label_w + bar_w, y + 13),
                       fill=(233, 235, 238),
                       outline=RULE)
        draw.rectangle((x0 + label_w, y,
                        x0 + label_w + max(1, int(bar_w * charge)), y + 13),
                       fill=(96, 132, 196))
        onset_x = x0 + label_w + int(bar_w * env.BRIGHTNESS_ONSET)
        draw.line((onset_x, y - 2, onset_x, y + 15), fill=(160, 60, 60))
        lit_rgb = tuple(
            int(255 * c)
            for c in env_cls.color_rgba(env_cls.lamp_color_index(i))[:3])
        swatch = tuple(
            int((1 - brightness) * d + brightness * l)
            for d, l in zip((150, 154, 160), lit_rgb))
        draw.rectangle(
            (x0 + label_w + 12 + bar_w, y, x0 + label_w + 36 + bar_w, y + 13),
            fill=swatch,
            outline=RULE)
        y += 26
    draw.text((x0 + label_w, y - 4),
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
            caption = f"{kind} {PyBulletBusyBoardEnv.button_label(button_idx)}"
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
        "num_test_tasks": 8,
        "pybullet_camera_width": FRAME_W,
        "pybullet_camera_height": FRAME_H,
    })
    os.makedirs(OUT_DIR, exist_ok=True)
    env = PyBulletBusyBoardEnv(use_gui=False)
    env._generate_train_tasks()  # pylint: disable=protected-access
    env._generate_test_tasks()  # pylint: disable=protected-access

    # One still per board size the run produces, and for the clips the
    # smallest test board that carries every lamp - the busiest board that
    # still reads clearly in a frame.
    sizes: Dict[Tuple[int, int], Tuple[str, int]] = {}
    for split, count in (("train", 1), ("test", 8)):
        for idx in range(count):
            env.reset(split, idx)
            key = (env._num_active_buttons, env._num_active_lamps)  # pylint: disable=protected-access
            sizes.setdefault(key, (split, idx))
    for (num_buttons, num_lamps), (split, idx) in sorted(sizes.items()):
        env.reset(split, idx)
        name = f"init_{num_buttons}buttons_{num_lamps}lamps.png"
        imageio.imwrite(os.path.join(OUT_DIR, name),
                        env.render()[0].astype("uint8"))
        print("wrote", name)

    max_lamps = max(num_lamps for _, num_lamps in sizes)
    split, idx = min((key, where) for key, where in sizes.items()
                     if key[1] == max_lamps and where[0] == "test")[1]
    task = env.get_task(split, idx)
    env.reset(split, idx)
    num_buttons = env._num_active_buttons  # pylint: disable=protected-access
    wiring = _wiring(task)
    solving = _solving_buttons(env, task, num_buttons)
    print(f"board {split}[{idx}]: {num_buttons} buttons, wiring {wiring}, "
          f"goal {_target(task, len(wiring))}, solving press set {solving}")

    # The cast for the interlock and confound clips: a lamp with a
    # conjunctive drive, the two buttons it needs, and one button that is
    # irrelevant to it.
    conj = next(i for i, (_, e) in enumerate(wiring) if e != NO_ENABLER)
    lamp0_driver, lamp0_enabler = wiring[conj]
    held = {lamp0_driver, lamp0_enabler}

    def _completes_another_lamp(button: int) -> bool:
        """Whether pressing ``button`` on top of the held pair lights some
        OTHER lamp, which would muddy a clip about this one."""
        return any({d, e} - {NO_ENABLER} <= held | {button}
                   for i, (d, e) in enumerate(wiring) if i != conj)

    irrelevant = next(b for b in range(num_buttons)
                      if b not in held and not _completes_another_lamp(b))
    lamp = PyBulletBusyBoardEnv.lamp_label(conj)
    drv = PyBulletBusyBoardEnv.button_label(lamp0_driver)
    enb = PyBulletBusyBoardEnv.button_label(lamp0_enabler)
    other = PyBulletBusyBoardEnv.button_label(irrelevant)

    target = _target(task, len(wiring))
    goal_note = (f"The goal needs {sum(target)} lamps lit and "
                 f"{len(target) - sum(target)} dark. Press exactly the "
                 "buttons their drive conditions call for.")
    clips: List[Tuple[str, List[Step], bool]] = [
        ("oracle_solve", [("note", goal_note, 2 * FPS)] +
         [("press", b) for b in solving] + [("wait", 110)], True),
        ("press_everything_fails",
         [("note",
           "Knowing that buttons matter but not which, latch them all.",
           2 * FPS)] + [("press", b)
                        for b in range(num_buttons)] + [("wait", 110)], True),
        ("interlock", [
            ("note", f"{lamp} needs {drv} AND {enb}. Press just one of "
             "them.", 2 * FPS),
            ("press", lamp0_enabler),
            ("wait", 90),
            ("note", "Nothing. Not a slow response - the charge bar never "
             "left zero, so the drive condition was never met.", 3 * FPS),
            ("press", lamp0_driver),
            ("wait", 120),
            ("note", f"With both on, the charge climbs and {lamp} lights. "
             "This conjunction is the relation prior busyboard "
             "benchmarks exclude by design.", 3 * FPS),
        ], False),
        (
            "delayed_credit",
            [
                ("note", f"Press {drv}, then {enb}: the condition of "
                 f"{lamp} is now met and it starts charging invisibly.",
                 2 * FPS),
                ("press", lamp0_driver),
                ("press", lamp0_enabler),
                ("wait", 26),
                # Tenseless on purpose: this note is still on screen at the
                # moment lamp0 lights, so "nothing is visible yet" would be
                # false exactly when the viewer is looking at it.
                ("note", f"{other} is unrelated to {lamp}. Watch the "
                 "board during its press.", 2 * FPS),
                ("press", irrelevant),
                ("note", f"{lamp} lit during the press of {other} - but "
                 f"{other} had nothing to do with it. This is the "
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


# ── Task cards ───────────────────────────────────────────────────

# Task counts for the card sheet. Larger than an experiment's so that
# every goal type and board size shows up; the seed-0 wiring is the same
# at any count, only the goal draw differs.
CARD_TRAIN_TASKS = 6
CARD_TEST_TASKS = 12
CARD_BAND_H = 150


def _goal_str(target: Sequence[bool]) -> str:
    """The goal in colour names, lit lamps first: ``cyan ON; yellow OFF``."""
    env_cls = PyBulletBusyBoardEnv
    names = [
        env_cls.color_name(env_cls.lamp_color_index(i))
        for i in range(len(target))
    ]
    lit = [n for n, t in zip(names, target) if t]
    dark = [n for n, t in zip(names, target) if not t]
    parts = []
    if lit:
        parts.append(", ".join(lit) + " ON")
    if dark:
        parts.append(", ".join(dark) + " OFF")
    return "; ".join(parts)


def _task_card(env: PyBulletBusyBoardEnv, task: EnvironmentTask, split: str,
               idx: int) -> np.ndarray:
    """The task's initial board with its goal and oracle answer beneath."""
    env.reset(split, idx)
    num_buttons = env._num_active_buttons  # pylint: disable=protected-access
    num_lamps = env._num_active_lamps  # pylint: disable=protected-access
    view = Image.fromarray(
        env.render()[0].astype("uint8"))  # type: ignore[no-untyped-call]
    canvas = Image.new("RGB", (FRAME_W, FRAME_H + CARD_BAND_H),
                       (252, 252, 250))
    canvas.paste(view, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title = _font(_FONT_DIRS, 34)
    body = _font(_FONT_DIRS, 30)
    x0, y = 28, FRAME_H + 14
    draw.text((x0, y), f"{split.capitalize()} task {idx}   "
              f"{num_buttons} buttons, {num_lamps} lamps",
              font=title,
              fill=INK)
    y += 46
    target = _target(task, num_lamps)
    draw.text((x0, y), f"Goal: {_goal_str(target)}", font=body, fill=INK)
    y += 40
    solving = _solving_buttons(env, task, num_buttons)
    env_cls = PyBulletBusyBoardEnv
    presses = " + ".join(
        env_cls.color_name(env_cls.button_color_index(b)) for b in solving)
    draw.text((x0, y),
              f"Oracle answer: press {presses}, wait",
              font=body,
              fill=MUTED)
    return np.asarray(canvas)


def _render_task_cards() -> None:
    """One card per distinct (board size, goal) seen in seed 0's tasks."""
    utils.reset_config({
        "env": "pybullet_busyboard",
        "seed": 0,
        "num_train_tasks": CARD_TRAIN_TASKS,
        "num_test_tasks": CARD_TEST_TASKS,
        "pybullet_camera_width": FRAME_W,
        "pybullet_camera_height": FRAME_H,
    })
    env = PyBulletBusyBoardEnv(use_gui=False)
    env._generate_train_tasks()  # pylint: disable=protected-access
    env._generate_test_tasks()  # pylint: disable=protected-access
    seen: Dict[Tuple[str, int, int, Tuple[bool, ...]], int] = {}
    for split, count in (("train", CARD_TRAIN_TASKS), ("test",
                                                       CARD_TEST_TASKS)):
        for idx in range(count):
            env.reset(split, idx)
            task = env.get_task(split, idx)
            num_lamps = env._num_active_lamps  # pylint: disable=protected-access
            key = (
                split,
                env._num_active_buttons,
                num_lamps,  # pylint: disable=protected-access
                tuple(_target(task, num_lamps)))
            seen.setdefault(key, idx)
    for (split, num_buttons, num_lamps, target), idx in sorted(seen.items()):
        task = env.get_task(split, idx)
        name = f"task_{split}_{idx}.png"
        imageio.imwrite(os.path.join(OUT_DIR, name),
                        _task_card(env, task, split, idx))
        print(f"wrote {name}: {num_buttons} buttons, {num_lamps} lamps, "
              f"goal {_goal_str(target)}")


def _print_goal_distribution(num_tasks: int = 120) -> None:
    """Print, per split, how many sampled goals light 1, 2, ... lamps on.

    each board size - the numbers behind the deck's distribution table.
    """
    for split in ("train", "test"):
        counts: Dict[Tuple[str, int], int] = {}
        env = PyBulletBusyBoardEnv(use_gui=False)
        tasks = (
            env._generate_train_tasks()  # pylint: disable=protected-access
            if split == "train" else env._generate_test_tasks())  # pylint: disable=protected-access
        for task in tasks:
            num_lamps = sum(1 for o in task.init if o.type.name == "lamp")
            num_buttons = sum(1 for o in task.init if o.type.name == "button")
            board = f"{num_buttons} buttons, {num_lamps} lamps"
            lit = sum(_target(task, num_lamps))
            counts[(board, lit)] = counts.get((board, lit), 0) + 1
        for (board, lit), n in sorted(counts.items()):
            print(f"{split:5s} {board}: {lit} lit: {n}/{num_tasks}")
        # The draw is uniform over the realizable targets of a board, so
        # the exact shares follow from enumerating them.
        min_lit = int(CFG.busyboard_min_lit_train if split ==
                      "train" else CFG.busyboard_min_lit_test)
        buttons = (CFG.busyboard_num_buttons_train
                   if split == "train" else CFG.busyboard_num_buttons_test)
        lamps = (CFG.busyboard_num_lamps_train
                 if split == "train" else CFG.busyboard_num_lamps_test)
        for num_buttons in buttons:
            for num_lamps in lamps:
                driver, enabler = canonical_wiring(num_buttons, num_lamps)
                targets = env._realizable_targets(  # pylint: disable=protected-access
                    driver, enabler, num_buttons,
                    min(num_lamps,
                        core_board()[1]), min_lit)
                by_lit: Dict[int, int] = {}
                for target in targets:
                    by_lit[sum(target)] = by_lit.get(sum(target), 0) + 1
                print(f"{split:5s} {num_buttons} buttons, {num_lamps} lamps: "
                      f"exact {len(targets)} goals: " +
                      ", ".join(f"{lit} lit x{n}"
                                for lit, n in sorted(by_lit.items())))


if __name__ == "__main__":
    _main()
    _render_task_cards()
    utils.reset_config({
        "env": "pybullet_busyboard",
        "seed": 0,
        "num_train_tasks": 120,
        "num_test_tasks": 120,
    })
    _print_goal_distribution()
