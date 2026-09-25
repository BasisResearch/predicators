"""Re-run the real-robot agent's test plan inside its own simulator.

At test time the robot agent of the 2026-09-22 cascade run
(``exp_20260922_134142``) checked its arrangement, the green domino at
0.375 m with the grey one standing upwind at 0.20 m, in BabyRobotPredicator's
``real_skills/wind_sim.py`` under 24 draws from its posterior, and predicted
a 10.1 +/- 2.3 cm slide, flat in the patch in 54% of the draws. That check
saved no images. This rebuilds it offline from the run's archived posterior,
scene, seed and wind model, with the planner's own resampling, and stops
unless it reproduces the recorded prediction.

It then re-runs one of those draws, the successful one whose slide is
closest to the mean, with both dominoes' poses recorded, and writes the
bench at chosen moments of that rollout as scene JSONs for
render_cycles_scene.py: the mat, the patch, the fan as the wind's source and
the two dominoes, seen from the gust camera. The engine models nothing else,
so nothing else is drawn. Beside each scene it writes a quick PyBullet
preview blended with the recorded test frame, to check the camera.

It imports the robot repository's code unchanged, so it runs with that
repository's dependencies (pybullet, pydantic and OpenCV), for example:

    ~/.conda/envs/robocode/bin/python \\
        scripts/paper_figures/render_robot_model_plan.py \\
        --repo ~/BabyRobotPredicator --preview <dir>

The repository must be at branch ``mn-fan-cascade-min``, the code the run
used.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import cv2  # type: ignore # pylint: disable=import-error
import numpy as np
import pybullet as p
import pybullet_data  # type: ignore
import render_scene_support as support
from PIL import Image

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parents[1] / "logs/real_robot/fan_domino_drive"
# The planner's draw count for every decision in the run (fan_plan.py's
# default, which fan_agent.sh does not override).
PLAN_DRAWS = 24
TRAJ_HZ = 30
# The fan's housing radius, and the button cap's height (m).
FAN_RADIUS, CAP_HEIGHT = 0.046, 0.012
# The Panda's fingertip link in pybullet_data's model, and a rest pose that
# keeps the elbow up.
PANDA_TIP = 11
PANDA_REST = [0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0, 0]
# The gust camera is the bench's look camera, a ZED 2i recorded at HD1080
# and archived at half size: its nominal focal length at 960 x 540, with the
# principal point at the centre of the rectified image.
WIDTH, HEIGHT, FOCAL = 960, 540, 533.0
# Pixels in the gust camera's test frame (casc_test.mp4, frame 2505) of the
# two dominoes' top faces, standing where the agent placed them. The fixtures'
# pixels are the centres of the boxes the agent's own look recorded.
TOP_PIXELS = {"grey": (274.0, 375.0), "green": (391.0, 399.0)}
# Where the lane's middle sits in the renderer's world: its lights are placed
# for the simulated domains, which centre their tables near (0.7, 1.2), 0.45
# up. A rigid shift moves the bench and the camera together.
RENDER_ANCHOR = np.array([0.7, 1.2, 0.45])
# Colours read from the gust camera's frames.
MAT, PATCH, TAPE = (.93, .93, .92, 1), (1, .36, .56, 1), (.86, .12, .12, 1)
FAN, HUB, BLADE = (.07, .07, .08, 1), (.35, .35, .37, 1), (.16, .16, .18, 1)
CLAMP, CAP, EDGE = (.92, .92, .92, 1), (.12, .36, .86, 1), (.20, .30, .78, 1)
FACES = {"green": (.24, .64, .27, 1), "grey": (.62, .64, .67, 1)}


def digest(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Bench:
    """The bench in the robot's base frame, from the run's fan frame."""

    def __init__(self, scene: Dict[str, Any], fan_height_m: float) -> None:
        self.origin = np.array(scene["objects"]["fan"]["center_base_m"][:2])
        self.direction = np.array(scene["blow_axis"]["dir_xy"])
        x, y = self.direction
        self.basis = np.array([[x, -y], [y, x]])
        self.table_z = scene["objects"]["fan"]["center_base_m"][2] - \
            fan_height_m
        self.yaw = math.atan2(y, x)
        self.render_offset = RENDER_ANCHOR - self.point((0.3, 0.0, 0.0))

    def point(self, fan_xyz: Sequence[float]) -> np.ndarray:
        """A fan-frame point (the table top at z = 0) in the base frame."""
        xy = np.asarray(fan_xyz[:2]) @ self.basis.T + self.origin
        return np.array([xy[0], xy[1], fan_xyz[2] + self.table_z])

    def pose(self, fan_pos: Sequence[float],
             fan_quat: Sequence[float]) -> Tuple[np.ndarray, Sequence[float]]:
        """A fan-frame pose in the base frame."""
        heading = p.getQuaternionFromEuler([0, 0, self.yaw])
        return self.point(fan_pos), p.multiplyTransforms([0, 0, 0], heading,
                                                         [0, 0, 0],
                                                         fan_quat)[1]


def gust_camera(scene: Dict[str, Any], boxes: Dict[str, Any], bench: Bench,
                placements: Dict[str, Tuple[float, float]],
                block_l: float) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Pose the gust camera from the fixtures and the standing dominoes."""
    points = [scene["objects"][k]["center_base_m"] for k in boxes]
    pixels = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes.values()]
    for block, (dist, lateral) in placements.items():
        points.append(bench.point((dist, lateral, block_l)))
        pixels.append(TOP_PIXELS[block])
    K = np.array([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]])
    ok, rvec, tvec = cv2.solvePnP(np.array(points, float),
                                  np.array(pixels, float),
                                  K,
                                  None,
                                  flags=cv2.SOLVEPNP_SQPNP)
    assert ok
    projected, _ = cv2.projectPoints(np.array(points, float), rvec, tvec, K,
                                     None)
    errors = np.linalg.norm(projected.reshape(-1, 2) - np.array(pixels),
                            axis=1)
    rotation, _ = cv2.Rodrigues(rvec)
    eye = -rotation.T @ tvec.ravel()
    return eye, rotation, [round(float(e), 1) for e in errors]


def view_and_projection(eye: np.ndarray, rotation: np.ndarray,
                        offset: np.ndarray) -> Tuple[List[float], List[float]]:
    """PyBullet matrices for an OpenCV camera, shifted into the render
    world."""
    forward, down = rotation.T @ [0, 0, 1], rotation.T @ [0, 1, 0]
    world_eye = eye + offset
    view = p.computeViewMatrix(list(world_eye), list(world_eye + forward),
                               list(-down))
    fov = math.degrees(2 * math.atan(HEIGHT / 2 / FOCAL))
    projection = p.computeProjectionMatrixFOV(fov, WIDTH / HEIGHT, .02, 20)
    return list(view), list(projection)


def record_rollout(wind_sim: Any, model: Any, params: Dict[str, float],
                   episode: Any) -> Tuple[Any, List[Dict[str, Any]]]:
    """Re-run one draw, keeping both dominoes' poses at TRAJ_HZ.

    The simulator keeps only the scored block's centre, so this wraps
    the engine's step to read every body after it; the physics is
    untouched.
    """
    frames: List[Dict[str, Any]] = []
    step: Callable[..., Any] = wind_sim.p.stepSimulation
    count = [0]
    settle = wind_sim.SIM_HZ // 4

    def recording_step(*args: Any, **kwargs: Any) -> Any:
        result = step(*args, **kwargs)
        client = kwargs["physicsClientId"]
        count[0] += 1
        if (count[0] - settle) % (wind_sim.SIM_HZ // TRAJ_HZ) == 0:
            poses = [
                p.getBasePositionAndOrientation(body, physicsClientId=client)
                for body in (1, 2)
            ]
            contact = bool(p.getContactPoints(1, 2, physicsClientId=client))
            frames.append(
                dict(
                    t=round((count[0] - settle) / wind_sim.SIM_HZ, 4),
                    poses=[[list(pos), list(q)] for pos, q in poses],
                    falls=[wind_sim._fall_deg(q) for _, q in poses],  # pylint: disable=protected-access
                    contact=contact))
        return result

    wind_sim.p.stepSimulation = recording_step
    try:
        outcome = wind_sim.simulate(model, params, episode)
    finally:
        wind_sim.p.stepSimulation = step
    return outcome, frames


Vector = Union[Sequence[float], np.ndarray]


def box(client: int, name: str, half: Vector, position: Vector,
        quaternion: Sequence[float], rgba: Sequence[float],
        offset: np.ndarray) -> None:
    """Add one visual-only box to the render scene, in render-world
    coordinates."""
    shape = p.createVisualShape(p.GEOM_BOX,
                                halfExtents=list(half),
                                rgbaColor=list(rgba),
                                physicsClientId=client)
    body = p.createMultiBody(0,
                             -1,
                             shape,
                             basePosition=list(np.asarray(position) + offset),
                             baseOrientation=list(quaternion),
                             physicsClientId=client)
    del body, name


def cylinder(client: int, radius: float, length: float, position: Vector,
             quaternion: Sequence[float], rgba: Sequence[float],
             offset: np.ndarray) -> None:
    """Add one visual-only cylinder, its axis along its local z."""
    shape = p.createVisualShape(p.GEOM_CYLINDER,
                                radius=radius,
                                length=length,
                                rgbaColor=list(rgba),
                                physicsClientId=client)
    p.createMultiBody(0,
                      -1,
                      shape,
                      basePosition=list(np.asarray(position) + offset),
                      baseOrientation=list(quaternion),
                      physicsClientId=client)


def turn(first: Sequence[float], then: Sequence[float]) -> Sequence[float]:
    """The rotation ``first`` followed by ``then``, as quaternions."""
    return p.multiplyTransforms([0, 0, 0], then, [0, 0, 0], first)[1]


def _patch(client: int, geo: Dict[str, Any], bench: Bench) -> None:
    """The patch, square to the fan as it lies on the bench, with its tape.

    The agent's box fit turns it about 15 degrees from the blow axis.
    Its success test used that fit; the drawing follows the bench.
    """
    goal = geo["goal"]
    along = p.getQuaternionFromEuler([0, 0, bench.yaw])
    half = np.asarray(goal["dims_m"][:2]) / 2
    middle = bench.point((*goal["center_base_m"][:2], 0.0005))
    box(client, "patch", (*half, 0.0005), middle, along, PATCH,
        bench.render_offset)
    rot = np.array(p.getMatrixFromQuaternion(along)).reshape(3, 3)
    for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1)):
        spot = middle + rot @ [sx * half[0], sy * half[1], 0.0003]
        box(client, "tape", (0.012, 0.01, 0.0005), spot, along, TAPE,
            bench.render_offset)


def _fan(client: int, bench: Bench, height: float) -> None:
    """A round fan facing down the blow axis, on a white clamp."""
    offset = bench.render_offset
    # Cylinders run along their local z; the fan's axis is the lane's x.
    facing = turn(p.getQuaternionFromEuler([0, math.pi / 2, 0]),
                  p.getQuaternionFromEuler([0, 0, bench.yaw]))
    cylinder(client, FAN_RADIUS, 0.03, bench.point((-0.015, 0, height)),
             facing, FAN, offset)
    cylinder(client, 0.012, 0.02, bench.point((0.005, 0, height)), facing, HUB,
             offset)
    for k in range(5):
        spin = p.getQuaternionFromEuler([2 * math.pi * k / 5, 0, 0])
        # Each blade pitched about its own radial axis, then set around the
        # hub, in the lane's frame.
        blade = turn(p.getQuaternionFromEuler([0, 0, math.radians(30)]), spin)
        radial = np.array(p.getMatrixFromQuaternion(spin)).reshape(
            3, 3) @ [0.004, 0, 0.026]
        position, orientation = bench.pose(
            (radial[0], radial[1], height + radial[2]), blade)
        box(client, "blade", (0.0015, 0.011, 0.017), position, orientation,
            BLADE, offset)
    along = p.getQuaternionFromEuler([0, 0, bench.yaw])
    post = height - FAN_RADIUS
    box(client, "clamp", (0.012, 0.012, post / 2),
        bench.point((-0.02, 0, post / 2)), along, CLAMP, offset)
    box(client, "clamp foot", (0.03, 0.03, 0.004),
        bench.point((-0.02, 0, 0.004)), along, CLAMP, offset)


def _button(client: int, scene: Dict[str, Any], bench: Bench) -> None:
    """The push button: a black housing with a blue cap on top.

    It stands square to the fan, as on the bench; the agent's box fit
    turns it, as it does the patch.
    """
    button = scene["objects"]["button"]
    top = button["center_base_m"][2] + button["dims_m"][2] / 2
    yaw = p.getQuaternionFromEuler([0, 0, bench.yaw])
    housing = top - CAP_HEIGHT - bench.table_z
    box(client, "button housing", (0.033, 0.033, housing / 2),
        [*button["center_base_m"][:2], bench.table_z + housing / 2], yaw, FAN,
        bench.render_offset)
    cylinder(client, 0.024, CAP_HEIGHT,
             [*button["center_base_m"][:2], top - CAP_HEIGHT / 2], yaw, CAP,
             bench.render_offset)


def _panda(client: int, scene: Dict[str, Any], bench: Bench) -> None:
    """The Panda on its mount, its fingertips on the button as during the gust.

    The robot's base frame is the fixtures' frame, so the arm stands at
    its origin. Only its pose is posed here; the wind simulator has no
    arm.
    """
    offset = bench.render_offset
    box(client, "mount", (0.15, 0.15, -bench.table_z / 2),
        [0, 0, bench.table_z / 2], [0, 0, 0, 1], FAN, offset)
    robot = p.loadURDF(str(
        Path(pybullet_data.getDataPath()) / "franka_panda/panda.urdf"),
                       basePosition=list(offset),
                       useFixedBase=True,
                       physicsClientId=client)
    button = scene["objects"]["button"]
    press = np.array([
        *button["center_base_m"][:2],
        button["center_base_m"][2] + button["dims_m"][2] / 2
    ]) + offset
    down = p.getQuaternionFromEuler([math.pi, 0, math.pi / 4])
    movable = [
        j for j in range(p.getNumJoints(robot, physicsClientId=client))
        if p.getJointInfo(robot, j, physicsClientId=client)[2] != p.JOINT_FIXED
    ]
    limits = [
        p.getJointInfo(robot, j, physicsClientId=client)[8:10] for j in movable
    ]
    # Start from the rest pose, so the solution keeps the elbow up.
    for joint, value in zip(movable, PANDA_REST):
        p.resetJointState(robot, joint, value, physicsClientId=client)
    joints = p.calculateInverseKinematics(
        robot,
        PANDA_TIP,
        list(press),
        down,
        lowerLimits=[lo for lo, _ in limits],
        upperLimits=[hi for _, hi in limits],
        jointRanges=[hi - lo for lo, hi in limits],
        restPoses=PANDA_REST,
        maxNumIterations=500,
        residualThreshold=1e-6,
        physicsClientId=client)
    for joint, value in zip(range(7), joints):
        p.resetJointState(robot, joint, value, physicsClientId=client)
    for finger in (9, 10):
        p.resetJointState(robot, finger, 0.0, physicsClientId=client)
    tip = p.getLinkState(robot,
                         PANDA_TIP,
                         computeForwardKinematics=True,
                         physicsClientId=client)[4]
    assert np.linalg.norm(np.asarray(tip) - press) < 0.005, tip


def build_scene(client: int, scene: Dict[str, Any], geo: Dict[str, Any],
                bench: Bench, dims: Tuple[float, float, float],
                blocks: Dict[str, Tuple[np.ndarray, Sequence[float]]]) -> None:
    """The bench as the model saw it, with its fixtures and the arm.

    The wind simulator itself holds only the table plane, the two
    dominoes and the wind; the patch, fan, button and arm are drawn
    where they stand.
    """
    along = p.getQuaternionFromEuler([0, 0, bench.yaw])
    offset = bench.render_offset
    # The engine's table is an unbounded plane: a mat wide enough to fill
    # the camera's view.
    centre = bench.point((0.3, 0.0, -0.005))
    box(client, "mat", (2.0, 2.0, 0.005), centre, along, MAT, offset)
    _patch(client, geo, bench)
    _fan(client, bench, geo["fan_height_m"])
    _button(client, scene, bench)
    _panda(client, scene, bench)
    thickness, width, length = dims
    for name, (position, quaternion) in blocks.items():
        box(client, f"{name} edges", (thickness / 2, width / 2, length / 2),
            position, quaternion, EDGE, offset)
        # The coloured faces, inset from the blue edges on both broad sides.
        box(client, f"{name} faces",
            (thickness / 2 + 0.0006, width / 2 - 0.006, length / 2 - 0.006),
            position, quaternion, FACES[name], offset)


def main() -> None:
    """Reproduce the recorded prediction, then export the chosen moments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",
                        type=Path,
                        default=Path.home() / "BabyRobotPredicator")
    parser.add_argument("--preview", type=Path, default=None)
    # Figure 3 shows the green domino tipping; the other moments are for
    # comparison in previews.
    parser.add_argument("--moments",
                        nargs="+",
                        choices=("knock", "tip", "final"),
                        default=["tip"])
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    # pylint: disable=import-error,import-outside-toplevel
    from real_skills import fan_plan, wind_sim  # type: ignore
    from real_skills.fan_contracts import Posterior  # type: ignore

    # pylint: enable=import-error,import-outside-toplevel
    decisions = [
        json.loads(line)
        for line in (RUN / "decisions.jsonl").read_text().splitlines()
    ]
    test = next(d["decision"] for d in decisions if d.get("test"))
    seed = int((RUN / "seed").read_text())
    posterior = Posterior.model_validate_json(
        (RUN / "posterior.json").read_text())
    scene = json.loads((RUN / "fixtures.json").read_text())
    # fan_plan.decide's order: the generator, the geometry, then the draws.
    rng = np.random.default_rng(seed)
    geo = fan_plan.geometry(scene)
    draws: List[Dict[str, float]] = fan_plan.resample(posterior, PLAN_DRAWS,
                                                      rng)
    model = wind_sim.load_model(str(RUN / "models/current.py"))
    also: Dict[str, Any] = test["also"]
    episode = wind_sim.Episode(dist_m=test["at"][0],
                               lateral_m=test["at"][1],
                               hold_s=2.0,
                               block=test["block"],
                               fan_height_m=geo["fan_height_m"],
                               others=((also["block"], also["at"][0],
                                        also["at"][1]), ))
    outcomes = [wind_sim.simulate(model, d, episode) for d in draws]
    slides = np.array([o.slide_m for o in outcomes])
    solved = [
        o.settled and not o.off_mat
        and fan_plan.in_goal(o.final_xy, geo["goal"], o.toppled)["solved"]
        for o in outcomes
    ]
    p_success = float(np.mean(solved))
    recorded_mean, recorded_std = test["slide_pred_m"]
    print(f"slide {slides.mean() * 100:.2f} +/- {slides.std() * 100:.2f} cm "
          f"(recorded {recorded_mean * 100:.2f} +/- "
          f"{recorded_std * 100:.2f}); flat in the patch in {p_success:.4f} "
          f"of draws (recorded {test['p_success']:.4f})")
    assert abs(slides.mean() - recorded_mean) < 5e-4
    assert abs(slides.std() - recorded_std) < 5e-4
    assert abs(p_success - test["p_success"]) < 1e-9

    # The draw to show: a successful one, whose slide is the mean's nearest.
    chosen = min((i for i, ok in enumerate(solved) if ok),
                 key=lambda i: abs(slides[i] - slides.mean()))
    outcome, frames = record_rollout(wind_sim, model, draws[chosen], episode)
    assert abs(outcome.slide_m - slides[chosen]) < 1e-9
    first_contact = next(f for f in frames if f["contact"])
    moments = {
        # The grey domino, blown over, meets the green one.
        "knock": first_contact,
        # The green domino tipping, halfway down.
        "tip": next(f for f in frames if f["falls"][0] >= 45),
        "final": frames[-1],
    }
    print(f"draw {chosen}: slide {outcome.slide_m * 100:.1f} cm; contact at "
          f"t={first_contact['t']:.2f} s; settled after "
          f"{frames[-1]['t']:.2f} s")

    bench = Bench(scene, geo["fan_height_m"])
    dims = (wind_sim.BLOCK_T, wind_sim.BLOCK_W, wind_sim.BLOCK_L)
    boxes = json.loads((RUN / "boxes_test.json").read_text())
    fixtures = {k: boxes[k] for k in ("fan", "button", "goal")}
    placements = {
        test["block"]: tuple(test["at"]),
        also["block"]: tuple(also["at"])
    }
    eye, rotation, errors = gust_camera(scene, fixtures, bench, placements,
                                        wind_sim.BLOCK_L)
    view, projection = view_and_projection(eye, rotation, bench.render_offset)
    print(f"gust camera at {np.round(eye, 3)} (base frame); reprojection "
          f"errors {errors} px")
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True).stdout.strip()
    provenance = dict(
        repository="BasisResearch/BabyRobotPredicator",
        commit=commit,
        simulator_sha256=digest(repo / "real_skills/wind_sim.py"),
        planner_sha256=digest(repo / "real_skills/fan_plan.py"),
        wind_model_sha256=digest(RUN / "models/current.py"),
        posterior_sha256=digest(RUN / "posterior.json"),
        scene_sha256=digest(RUN / "fixtures.json"),
        seed=seed,
        draws=PLAN_DRAWS,
        reproduced=dict(slide_m=[float(slides.mean()),
                                 float(slides.std())],
                        p_success=p_success),
        draw=dict(index=chosen,
                  params=draws[chosen],
                  slide_m=float(outcome.slide_m)),
        camera=dict(focal_px=FOCAL,
                    eye_base_m=[float(v) for v in eye],
                    reprojection_px=errors))
    names = {
        "green": 0,
        "grey": 1
    } if test["block"] == "green" else {
        "grey": 0,
        "green": 1
    }
    for moment in args.moments:
        frame = moments[moment]
        client = p.connect(p.DIRECT)
        try:
            blocks = {
                name: bench.pose(*frame["poses"][index])
                for name, index in names.items()
            }
            build_scene(client, scene, geo, bench, dims, blocks)
            name = f"real_fan_domino_model_{moment}"
            exported = support.export_visual_scene(
                client, view, projection, WIDTH, HEIGHT,
                dict(provenance, moment=moment, time_s=frame["t"]))
            (ROOT / "data/cycles_scenes" /
             f"{name}.json").write_text(json.dumps(exported, indent=1) + "\n")
            if args.preview is not None:
                args.preview.mkdir(parents=True, exist_ok=True)
                image = np.asarray(
                    p.getCameraImage(WIDTH,
                                     HEIGHT,
                                     viewMatrix=view,
                                     projectionMatrix=projection,
                                     renderer=p.ER_TINY_RENDERER,
                                     physicsClientId=client)[2],
                    np.uint8).reshape(HEIGHT, WIDTH, 4)[:, :, :3]
                photo = Image.open(
                    args.preview /
                    "gust_test_2505_left.png").convert("RGB") if (
                        args.preview /
                        "gust_test_2505_left.png").exists() else None
                render = Image.fromarray(  # type: ignore[no-untyped-call]
                    image)
                render.save(args.preview / f"{name}.png")
                if photo is not None:
                    Image.blend(photo, render,
                                .5).save(args.preview / f"{name}_overlay.png")
        finally:
            p.disconnect(client)
        print(f"wrote {name} (t={frame['t']:.2f} s)")


if __name__ == "__main__":
    main()
