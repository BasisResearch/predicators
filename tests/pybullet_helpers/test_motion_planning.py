"""Tests for PyBullet motion planning."""

import time

import numpy as np
import pybullet as p

from predicators import utils
from predicators.pybullet_helpers.camera import create_gui_connection
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.link import get_link_state
from predicators.pybullet_helpers.motion_planning import run_motion_planning
from predicators.pybullet_helpers.objects import create_pybullet_block
from predicators.pybullet_helpers.robots import \
    create_single_arm_pybullet_robot

USE_GUI = False


def test_run_motion_planning(physics_client_id):
    """Tests for run_motion_planning()."""
    ee_home_position = (1.35, 0.75, 0.75)
    ee_orn = p.getQuaternionFromEuler([0.0, np.pi / 2, -np.pi])
    ee_home_pose = Pose(ee_home_position, ee_orn)
    seed = 123
    robot = create_single_arm_pybullet_robot("fetch", physics_client_id,
                                             ee_home_pose)
    robot_init_state = tuple(ee_home_position) + tuple(
        ee_orn, ) + (robot.open_fingers, )
    robot.reset_state(robot_init_state)
    joint_initial = robot.get_joints()
    # Should succeed with a path of length 2.
    joint_target = list(joint_initial)
    path = run_motion_planning(robot,
                               joint_initial,
                               joint_target,
                               collision_bodies=set(),
                               seed=seed,
                               physics_client_id=physics_client_id)
    assert len(path) == 2
    assert np.allclose(path[0], joint_initial)
    assert np.allclose(path[-1], joint_target)
    # Should succeed, no collisions.
    ee_target_position = np.add(ee_home_position, (0.0, 0.0, -0.05))
    ee_target = Pose(ee_target_position, ee_orn)
    joint_target = robot.inverse_kinematics(ee_target, validate=True)
    path = run_motion_planning(robot,
                               joint_initial,
                               joint_target,
                               collision_bodies=set(),
                               seed=seed,
                               physics_client_id=physics_client_id)
    assert np.allclose(path[0], joint_initial)
    assert np.allclose(path[-1], joint_target)
    # Should fail because the target collides with the table.
    table_pose = (1.35, 0.75, 0.0)
    table_orientation = [0., 0., 0., 1.]
    table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                          useFixedBase=True,
                          physicsClientId=physics_client_id)
    p.resetBasePositionAndOrientation(table_id,
                                      table_pose,
                                      table_orientation,
                                      physicsClientId=physics_client_id)
    ee_target_position = np.add(ee_home_position, (0.0, 0.0, -0.6))
    ee_target = Pose(ee_target_position, ee_orn)
    joint_target = robot.inverse_kinematics(ee_target, validate=True)
    path = run_motion_planning(robot,
                               joint_initial,
                               joint_target,
                               collision_bodies={table_id},
                               seed=seed,
                               physics_client_id=physics_client_id)
    assert path is None
    # Should fail because the initial state collides with the table.
    path = run_motion_planning(robot,
                               joint_target,
                               joint_initial,
                               collision_bodies={table_id},
                               seed=seed,
                               physics_client_id=physics_client_id)
    assert path is None
    # Should succeed, but will need to move the arm up to avoid the obstacle.
    block_pose = (1.35, 0.6, 0.5)
    block_orientation = [0., 0., 0., 1.]
    block_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.2, 0.01, 0.3),
        mass=0,  # immoveable
        friction=1,
        orientation=block_orientation,
        physics_client_id=physics_client_id)
    p.resetBasePositionAndOrientation(block_id,
                                      block_pose,
                                      block_orientation,
                                      physicsClientId=physics_client_id)
    ee_target_position = (1.35, 0.4, 0.6)
    ee_target = Pose(ee_target_position, ee_orn)
    joint_target = robot.inverse_kinematics(ee_target, validate=True)
    path = run_motion_planning(robot,
                               joint_initial,
                               joint_target,
                               collision_bodies={table_id, block_id},
                               seed=seed,
                               physics_client_id=physics_client_id)
    assert path is not None
    p.removeBody(block_id, physicsClientId=physics_client_id)
    # Should fail because the hyperparameters are too limited.
    utils.reset_config({
        "pybullet_birrt_num_iters": 1,
        "pybullet_birrt_num_attempts": 1,
    })
    block_id = create_pybullet_block(
        color=(1.0, 0.0, 0.0, 1.0),
        half_extents=(0.2, 0.01, 0.3),
        mass=0,  # immoveable
        friction=1,
        orientation=block_orientation,
        physics_client_id=physics_client_id)
    p.resetBasePositionAndOrientation(block_id,
                                      block_pose,
                                      block_orientation,
                                      physicsClientId=physics_client_id)
    path = run_motion_planning(robot,
                               joint_initial,
                               joint_target,
                               collision_bodies={table_id, block_id},
                               seed=seed,
                               physics_client_id=physics_client_id)
    assert path is None
    p.removeBody(block_id, physicsClientId=physics_client_id)


def test_bystander_clearance(physics_client_id):
    """A planned path keeps positive clearance from bystander bodies.

    The hard contact margin alone tolerates ~1mm of penetration, which
    lets a "collision-free" path physically graze an obstacle (enough to
    topple a knife-edge object). With the bystander clearance, every
    checked configuration must keep the clearance from bodies the plan
    neither starts nor ends near.
    """
    ee_home_position = (1.35, 0.75, 0.75)
    ee_orn = p.getQuaternionFromEuler([0.0, np.pi / 2, -np.pi])
    ee_home_pose = Pose(ee_home_position, ee_orn)
    robot = create_single_arm_pybullet_robot("fetch", physics_client_id,
                                             ee_home_pose)
    robot_init_state = tuple(ee_home_position) + tuple(
        ee_orn, ) + (robot.open_fingers, )
    robot.reset_state(robot_init_state)
    joint_initial = robot.get_joints()
    # Thin wall between the start and the target.
    block_id = create_pybullet_block(color=(1.0, 0.0, 0.0, 1.0),
                                     half_extents=(0.2, 0.01, 0.3),
                                     mass=0,
                                     friction=1,
                                     orientation=(0., 0., 0., 1.),
                                     physics_client_id=physics_client_id)
    p.resetBasePositionAndOrientation(block_id, (1.35, 0.6, 0.5),
                                      [0., 0., 0., 1.],
                                      physicsClientId=physics_client_id)
    ee_target = Pose((1.35, 0.4, 0.6), ee_orn)
    joint_target = robot.inverse_kinematics(ee_target, validate=True)
    clearance = 0.005
    utils.reset_config({"pybullet_birrt_bystander_clearance": clearance})
    path = None
    # Motion planning is non-deterministic (RRT); try multiple seeds.
    for seed in [123, 456, 789]:
        robot.set_joints(joint_initial)
        path = run_motion_planning(robot,
                                   joint_initial,
                                   joint_target,
                                   collision_bodies={block_id},
                                   seed=seed,
                                   physics_client_id=physics_client_id)
        if path is not None:
            break
    assert path is not None
    # Neither endpoint is within the clearance of the wall, so the wall
    # is a bystander: every waypoint must keep the full clearance.
    for pt in path:
        robot.set_joints(pt)
        assert not p.getClosestPoints(robot.robot_id,
                                      block_id,
                                      clearance - 1e-6,
                                      physicsClientId=physics_client_id)
    p.removeBody(block_id, physicsClientId=physics_client_id)


def test_robot_start_escape(physics_client_id):
    """A start config with a shallow robot-vs-body contact still plans.

    The planning scene is reconstructed from observable features, so a
    phase that begins right after a grasp or a settled place can model a
    finger or wrist link several mm inside the object it just touched.
    Such a start is a fact, not a choice: it must not reject the whole
    plan; the path escapes the contact instead (never going deeper than
    it began). Start penetration deeper than the dedicated
    ``_ROBOT_START_ESCAPE_MAX_DEPTH`` bound still rejects. Deliberately
    run with the default shallow held-object margin: the escape window
    must not depend on it (it once did, and dropping a bridge margin
    override silently narrowed the window to 6 mm).
    """
    utils.reset_config({
        "pybullet_birrt_contact_margin": -0.001,
    })
    ee_home_position = (1.35, 0.75, 0.75)
    ee_orn = p.getQuaternionFromEuler([0.0, np.pi / 2, -np.pi])
    ee_home_pose = Pose(ee_home_position, ee_orn)
    robot = create_single_arm_pybullet_robot("fetch", physics_client_id,
                                             ee_home_pose)
    robot_init_state = tuple(ee_home_position) + tuple(
        ee_orn, ) + (robot.open_fingers, )
    robot.reset_state(robot_init_state)
    joint_initial = robot.get_joints()
    block_id = create_pybullet_block(color=(0.0, 0.0, 1.0, 1.0),
                                     half_extents=(0.03, 0.03, 0.03),
                                     mass=0,
                                     friction=1,
                                     orientation=(0., 0., 0., 1.),
                                     physics_client_id=physics_client_id)

    def _min_robot_dist(z: float) -> float:
        p.resetBasePositionAndOrientation(block_id, (1.35, 0.75, z),
                                          [0., 0., 0., 1.],
                                          physicsClientId=physics_client_id)
        robot.set_joints(joint_initial)
        p.performCollisionDetection(physicsClientId=physics_client_id)
        contacts = p.getContactPoints(robot.robot_id,
                                      block_id,
                                      physicsClientId=physics_client_id)
        return min((c[8] for c in contacts), default=float("inf"))

    # Raise the block toward the gripper until a robot link is modeled
    # 5-12 mm inside it (the artifact depth seen in post-grasp /
    # post-place reconstructions).
    shallow_z = None
    for z in np.arange(0.40, 0.80, 0.001):
        depth = _min_robot_dist(z)
        if -0.012 < depth < -0.005:
            shallow_z = z
            break
        if depth <= -0.012:
            break
    assert shallow_z is not None
    start_depth = _min_robot_dist(shallow_z)
    ee_target = Pose((1.35, 0.75, 0.90), ee_orn)
    joint_target = robot.inverse_kinematics(ee_target, validate=True)
    path = None
    # Motion planning is non-deterministic (RRT); try multiple seeds.
    for seed in [123, 456, 789]:
        robot.set_joints(joint_initial)
        path = run_motion_planning(robot,
                                   joint_initial,
                                   joint_target,
                                   collision_bodies={block_id},
                                   seed=seed,
                                   physics_client_id=physics_client_id)
        if path is not None:
            break
    assert path is not None
    # The escape never deepens the start contact beyond how it began
    # (plus the small slack), and the goal keeps the hard margin.
    for pt in path:
        robot.set_joints(pt)
        p.performCollisionDetection(physicsClientId=physics_client_id)
        contacts = p.getContactPoints(robot.robot_id,
                                      block_id,
                                      physicsClientId=physics_client_id)
        assert all(c[8] >= start_depth - 0.003 - 1e-6 for c in contacts)
    robot.set_joints(path[-1])
    p.performCollisionDetection(physicsClientId=physics_client_id)
    contacts = p.getContactPoints(robot.robot_id,
                                  block_id,
                                  physicsClientId=physics_client_id)
    assert all(c[8] >= -0.001 for c in contacts)
    # Start penetration deeper than the shallow margin still signals
    # genuine scene corruption and rejects the plan.
    deep_z = None
    for z in np.arange(shallow_z, 0.90, 0.002):
        if _min_robot_dist(z) < -0.025:
            deep_z = z
            break
    assert deep_z is not None
    robot.set_joints(joint_initial)
    path = run_motion_planning(robot,
                               joint_initial,
                               joint_target,
                               collision_bodies={block_id},
                               seed=123,
                               physics_client_id=physics_client_id)
    assert path is None
    p.removeBody(block_id, physicsClientId=physics_client_id)


def test_start_local_partner_demotion(physics_client_id):
    """Partner status earned only at the start expires with the start.

    A movable body the robot merely begins near is checked with the hard
    contact margin only inside the start neighborhood; beyond it the
    path may touch the body but not penetrate it. Otherwise a body
    grazed on the way out of the start keeps a penetration allowance for
    the entire path, which physically shoves it (a bottle retreat after
    a glue dab repeatedly nudged an assembled row this way). Static
    bodies cannot be shoved and keep their partner margin.
    """
    utils.reset_config({
        "pybullet_birrt_contact_margin": -0.03,
        "pybullet_birrt_bystander_clearance": 0.005,
    })
    ee_home_position = (1.35, 0.75, 0.75)
    ee_orn = p.getQuaternionFromEuler([0.0, np.pi / 2, -np.pi])
    ee_home_pose = Pose(ee_home_position, ee_orn)
    robot = create_single_arm_pybullet_robot("fetch", physics_client_id,
                                             ee_home_pose)
    robot.reset_state(
        tuple(ee_home_position) + tuple(ee_orn, ) + (robot.open_fingers, ))
    joint_initial = robot.get_joints()
    # The goal is on the far side of a thin wall, so the path must
    # travel around it: plenty of opportunity to graze it mid-flight.
    ee_target = Pose((1.35, 0.4, 0.6), ee_orn)
    joint_target = robot.inverse_kinematics(ee_target, validate=True)
    assert np.max(np.abs(np.subtract(joint_target, joint_initial))) > 0.5

    def _plan_around_wall(mass: float, seed: int):
        wall_id = create_pybullet_block(color=(1.0, 0.0, 0.0, 1.0),
                                        half_extents=(0.2, 0.01, 0.3),
                                        mass=mass,
                                        friction=1,
                                        orientation=(0., 0., 0., 1.),
                                        physics_client_id=physics_client_id)
        # Slide the wall toward the arm until the start config is just
        # within the bystander clearance of it (earning partner status)
        # without penetrating it.
        near_start = False
        for wall_y in np.arange(0.80, 0.55, -0.002):
            p.resetBasePositionAndOrientation(
                wall_id, (1.35, wall_y, 0.5), [0., 0., 0., 1.],
                physicsClientId=physics_client_id)
            robot.set_joints(joint_initial)
            contacts = p.getClosestPoints(robot.robot_id,
                                          wall_id,
                                          0.005,
                                          physicsClientId=physics_client_id)
            distances = [c[8] for c in contacts]
            if distances and min(distances) > 0.0:
                near_start = True
                break
        assert near_start
        robot.set_joints(joint_initial)
        return wall_id, run_motion_planning(
            robot,
            joint_initial,
            joint_target,
            collision_bodies={wall_id},
            seed=seed,
            physics_client_id=physics_client_id)

    for seed in [123, 456, 789]:
        wall_id, path = _plan_around_wall(1.0, seed)
        assert path is not None
        num_far = 0
        for pt in path:
            if np.max(np.abs(np.subtract(pt, joint_initial))) < 0.5:
                continue
            num_far += 1
            robot.set_joints(pt)
            p.performCollisionDetection(physicsClientId=physics_client_id)
            contacts = p.getContactPoints(robot.robot_id,
                                          wall_id,
                                          physicsClientId=physics_client_id)
            # Beyond the start neighborhood, touching is still legal but
            # penetrating the movable wall is not.
            assert all(c[8] >= -1e-6 for c in contacts)
        assert num_far > 0
        p.removeBody(wall_id, physicsClientId=physics_client_id)
    # The same wall, static: nothing the path does can displace it, so
    # its partner margin stands for the whole path and planning is not
    # made harder. (With this geometry the undemoted margin is real:
    # seed 123 routes a link 17 mm through the static wall.)
    wall_id, path = _plan_around_wall(0.0, 123)
    assert path is not None
    p.removeBody(wall_id, physicsClientId=physics_client_id)


def test_held_attachments(physics_client_id):
    """Bodies rigidly attached to the held object are collision-checked.

    A goal that keeps the held object itself clear of an obstacle but
    sweeps a welded attachment into it must be rejected; the same goal
    without the attachment plans fine.
    """
    utils.reset_config({})
    ee_home_position = (1.35, 0.75, 0.75)
    ee_orn = p.getQuaternionFromEuler([0.0, np.pi / 2, -np.pi])
    ee_home_pose = Pose(ee_home_position, ee_orn)
    robot = create_single_arm_pybullet_robot("fetch", physics_client_id,
                                             ee_home_pose)
    robot_init_state = tuple(ee_home_position) + tuple(
        ee_orn, ) + (robot.open_fingers, )
    robot.reset_state(robot_init_state)
    joint_initial = robot.get_joints()
    block_kwargs = {
        "color": (0.0, 0.0, 1.0, 1.0),
        "half_extents": (0.03, 0.03, 0.03),
        # Nonzero mass: Bullet generates no contacts between two static
        # bodies, and the obstacle below is static.
        "mass": 0.1,
        "friction": 1,
        "orientation": [0., 0., 0., 1.],
        "physics_client_id": physics_client_id,
    }
    # The held object hangs 10 cm under the end effector; a welded
    # partner sits 15 cm to its +y side (like a row member).
    held_id = create_pybullet_block(**block_kwargs)
    held_position = np.add(ee_home_position, (0.0, 0.0, -0.1))
    p.resetBasePositionAndOrientation(held_id,
                                      held_position, [0., 0., 0., 1.],
                                      physicsClientId=physics_client_id)
    attached_id = create_pybullet_block(**block_kwargs)
    attached_position = np.add(held_position, (0.0, 0.15, 0.0))
    p.resetBasePositionAndOrientation(attached_id,
                                      attached_position, [0., 0., 0., 1.],
                                      physicsClientId=physics_client_id)
    world_to_base_link = get_link_state(
        robot.robot_id,
        robot.end_effector_id,
        physics_client_id=physics_client_id).com_pose
    base_link_to_world = p.invertTransform(world_to_base_link[0],
                                           world_to_base_link[1])
    base_link_to_held = p.multiplyTransforms(base_link_to_world[0],
                                             base_link_to_world[1],
                                             held_position, [0., 0., 0., 1.])
    base_link_to_attached = p.multiplyTransforms(base_link_to_world[0],
                                                 base_link_to_world[1],
                                                 attached_position,
                                                 [0., 0., 0., 1.])
    # Static obstacle exactly where the ATTACHED body ends up after the
    # planned 10 cm descent; the held object and the robot stay clear.
    obstacle_id = create_pybullet_block(color=(1.0, 0.0, 0.0, 1.0),
                                        half_extents=(0.05, 0.05, 0.05),
                                        mass=0,
                                        friction=1,
                                        orientation=(0., 0., 0., 1.),
                                        physics_client_id=physics_client_id)
    p.resetBasePositionAndOrientation(obstacle_id,
                                      np.add(attached_position,
                                             (0.0, 0.0, -0.1)),
                                      [0., 0., 0., 1.],
                                      physicsClientId=physics_client_id)
    ee_target = Pose(tuple(np.add(ee_home_position, (0.0, 0.0, -0.1))), ee_orn)
    joint_target = robot.inverse_kinematics(ee_target, validate=True)
    # With the attachment checked, the goal sweeps it into the obstacle.
    path = run_motion_planning(
        robot,
        joint_initial,
        joint_target,
        collision_bodies={obstacle_id},
        seed=123,
        physics_client_id=physics_client_id,
        held_object=held_id,
        base_link_to_held_obj=base_link_to_held,
        held_attachments={attached_id: base_link_to_attached})
    assert path is None
    # Without the attachment, the same goal plans fine.
    path = None
    for seed in [123, 456, 789]:
        robot.set_joints(joint_initial)
        path = run_motion_planning(robot,
                                   joint_initial,
                                   joint_target,
                                   collision_bodies={obstacle_id},
                                   seed=seed,
                                   physics_client_id=physics_client_id,
                                   held_object=held_id,
                                   base_link_to_held_obj=base_link_to_held)
        if path is not None:
            break
    assert path is not None
    for body in (held_id, attached_id, obstacle_id):
        p.removeBody(body, physicsClientId=physics_client_id)


def test_move_to_shelf():
    """Test for Panda robot moving to put a held block into a shelf.

    Notably, the robot must change its gripper orientation from top-down
    to forward-facing, so motion planning must be in position and
    orientation.

    Also notably, the held object must be collision-checked like the robot.
    """
    utils.reset_config({"pybullet_control_mode": "reset"})

    # Set up scene.
    x_lb = 1.2
    x_ub = 1.5
    y_lb = 0.4
    y_ub = 1.1
    pick_z = 0.75
    default_orn = (0.0, 0.0, 0.0, 1.0)
    table_pose = (1.35, 0.75, 0.0)
    table_orientation = (0., 0., 0., 1.)
    table_height = 0.2
    shelf_width = (x_ub - x_lb) * 0.4
    shelf_length = (y_ub - y_lb) * 0.6
    shelf_base_height = pick_z * 0.8
    shelf_ceiling_height = pick_z * 0.2
    shelf_ceiling_thickness = 0.01
    shelf_pole_girth = 0.01
    shelf_color = (0.5, 0.3, 0.05, 1.0)
    shelf_x = x_ub - shelf_width / 2
    shelf_y = y_lb + shelf_length / 2
    block_color = (1.0, 0.0, 0.0, 1.0)
    block_size = 0.05
    block_x = (x_lb + x_ub) / 2
    block_y = y_ub - block_size
    block_z = table_height + block_size / 2
    offset_z = 0.01
    obj_mass = 0.5
    obj_friction = 1.2
    robot_ee_home_orn = (0.7071, 0.7071, 0.0, 0.0)
    home_pose = Pose((block_x, block_y, block_z + offset_z), robot_ee_home_orn)

    # Target for motion planning.
    tx = shelf_x
    ty = shelf_y
    tz = table_height + shelf_base_height + block_size / 2 + offset_z
    target_orn = (0.7071, 0.0, 0.7071, 0.0)
    target_pose = Pose((tx, ty, tz), target_orn)

    if USE_GUI:  # pragma: no cover
        physics_client_id = create_gui_connection()
        # Draw the target.
        p.addUserDebugText("*",
                           target_pose.position, [1.0, 0.0, 0.0],
                           physicsClientId=physics_client_id)
    else:
        physics_client_id = p.connect(p.DIRECT)

    # Load table.
    table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                          useFixedBase=True,
                          physicsClientId=physics_client_id)
    p.resetBasePositionAndOrientation(table_id,
                                      table_pose,
                                      table_orientation,
                                      physicsClientId=physics_client_id)

    # Create shelf.
    color = shelf_color
    orientation = default_orn
    base_pose = (shelf_x, shelf_y, table_height + shelf_base_height / 2)
    # Shelf base.
    # Create the collision shape.
    base_half_extents = [
        shelf_width / 2, shelf_length / 2, shelf_base_height / 2
    ]
    base_collision_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=base_half_extents,
        physicsClientId=physics_client_id)
    # Create the visual shape.
    base_visual_id = p.createVisualShape(p.GEOM_BOX,
                                         halfExtents=base_half_extents,
                                         rgbaColor=color,
                                         physicsClientId=physics_client_id)
    # Create the ceiling.
    link_positions = []
    link_collision_shape_indices = []
    link_visual_shape_indices = []
    pose = (
        0, 0,
        shelf_base_height / 2 + shelf_ceiling_height - \
            shelf_ceiling_thickness / 2
    )
    link_positions.append(pose)
    half_extents = [
        shelf_width / 2, shelf_length / 2, shelf_ceiling_thickness / 2
    ]
    collision_id = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=half_extents,
                                          physicsClientId=physics_client_id)
    link_collision_shape_indices.append(collision_id)
    visual_id = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=half_extents,
                                    rgbaColor=color,
                                    physicsClientId=physics_client_id)
    link_visual_shape_indices.append(visual_id)
    # Create poles connecting the base to the ceiling.
    for x_sign in [-1, 1]:
        for y_sign in [-1, 1]:
            pose = (x_sign * (shelf_width - shelf_pole_girth) / 2,
                    y_sign * (shelf_length - shelf_pole_girth) / 2,
                    shelf_base_height / 2 + shelf_ceiling_height / 2)
            link_positions.append(pose)
            half_extents = [
                shelf_pole_girth / 2, shelf_pole_girth / 2,
                shelf_ceiling_height / 2
            ]
            collision_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=physics_client_id)
            link_collision_shape_indices.append(collision_id)
            visual_id = p.createVisualShape(p.GEOM_BOX,
                                            halfExtents=half_extents,
                                            rgbaColor=color,
                                            physicsClientId=physics_client_id)
            link_visual_shape_indices.append(visual_id)

    # Create the whole body.
    num_links = len(link_positions)
    assert len(link_collision_shape_indices) == num_links
    assert len(link_visual_shape_indices) == num_links
    link_masses = [0.1 for _ in range(num_links)]
    link_orientations = [orientation for _ in range(num_links)]
    link_intertial_frame_positions = [[0, 0, 0] for _ in range(num_links)]
    link_intertial_frame_orns = [[0, 0, 0, 1] for _ in range(num_links)]
    link_parent_indices = [0 for _ in range(num_links)]
    link_joint_types = [p.JOINT_FIXED for _ in range(num_links)]
    link_joint_axis = [[0, 0, 0] for _ in range(num_links)]
    shelf_id = p.createMultiBody(
        baseCollisionShapeIndex=base_collision_id,
        baseVisualShapeIndex=base_visual_id,
        basePosition=base_pose,
        baseOrientation=orientation,
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_collision_shape_indices,
        linkVisualShapeIndices=link_visual_shape_indices,
        linkPositions=link_positions,
        linkOrientations=link_orientations,
        linkInertialFramePositions=link_intertial_frame_positions,
        linkInertialFrameOrientations=link_intertial_frame_orns,
        linkParentIndices=link_parent_indices,
        linkJointTypes=link_joint_types,
        linkJointAxis=link_joint_axis,
        physicsClientId=physics_client_id)

    # Create block.
    color = block_color
    half_extents = (block_size / 2.0, block_size / 2.0, block_size / 2.0)
    block_id = create_pybullet_block(color,
                                     half_extents,
                                     obj_mass,
                                     obj_friction,
                                     orientation=default_orn,
                                     physics_client_id=physics_client_id)
    p.resetBasePositionAndOrientation(block_id, [block_x, block_y, block_z],
                                      default_orn,
                                      physicsClientId=physics_client_id)

    # Create robot, initialized to be grasping the block.
    robot = create_single_arm_pybullet_robot("panda", physics_client_id,
                                             home_pose)
    # Close the fingers.
    joint_state = robot.get_joints()
    joint_state[robot.left_finger_joint_idx] = robot.closed_fingers
    joint_state[robot.right_finger_joint_idx] = robot.closed_fingers
    robot.set_joints(joint_state)

    # Create holding transform.
    held_obj_id = block_id
    world_to_base_link = get_link_state(
        robot.robot_id,
        robot.end_effector_id,
        physics_client_id=physics_client_id).com_pose
    base_link_to_world = np.r_[p.invertTransform(world_to_base_link[0],
                                                 world_to_base_link[1])]
    world_to_obj = np.r_[p.getBasePositionAndOrientation(
        held_obj_id, physicsClientId=physics_client_id)]
    held_obj_to_base_link = p.invertTransform(
        *p.multiplyTransforms(base_link_to_world[:3], base_link_to_world[3:],
                              world_to_obj[:3], world_to_obj[3:]))
    base_link_to_held_obj = p.invertTransform(*held_obj_to_base_link)

    def _set_state(pt: JointPositions) -> None:
        robot.set_joints(pt)
        world_to_base_link = get_link_state(
            robot.robot_id,
            robot.end_effector_id,
            physics_client_id=physics_client_id).com_pose
        world_to_held_obj = p.multiplyTransforms(world_to_base_link[0],
                                                 world_to_base_link[1],
                                                 base_link_to_held_obj[0],
                                                 base_link_to_held_obj[1])
        p.resetBasePositionAndOrientation(held_obj_id,
                                          world_to_held_obj[0],
                                          world_to_held_obj[1],
                                          physicsClientId=physics_client_id)

    # Force move to target to get the target joint positions.
    robot_state = tuple(target_pose.position) + \
        tuple(target_pose.orientation) + (robot.closed_fingers, )
    robot.reset_state(robot_state)
    target_positions = robot.get_joints()

    # Move back to start, but slightly up so that the robot is not in collision
    # with the table.
    x, y, z = home_pose.position
    z += offset_z
    robot_state = (x, y, z) + \
        tuple(home_pose.orientation) + (robot.closed_fingers, )
    robot.reset_state(robot_state)
    initial_positions = robot.get_joints()
    _set_state(initial_positions)

    collision_bodies = {shelf_id, table_id}
    # Motion planning is non-deterministic (RRT); try multiple seeds.
    plan = None
    for seed in [123, 456, 789]:
        plan = run_motion_planning(robot,
                                   initial_positions,
                                   target_positions,
                                   collision_bodies,
                                   held_object=held_obj_id,
                                   base_link_to_held_obj=base_link_to_held_obj,
                                   seed=seed,
                                   physics_client_id=physics_client_id)
        if plan is not None:
            break
    assert plan is not None

    # Replay the plan.
    if USE_GUI:  # pragma: no cover
        for state in plan:
            _set_state(state)
            for _ in range(100):
                p.stepSimulation(physicsClientId=physics_client_id)
                time.sleep(0.001)
