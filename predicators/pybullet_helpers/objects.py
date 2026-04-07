"""predicatorsbullet_helpers.objects module."""
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.utils import _Geom2D

# import numpy as np
default_orn: Quaternion = (0.0, 0.0, 0.0, 1.0)


def create_object(asset_path: str,
                  position: Pose3D = (0, 0, 0),
                  orientation: Quaternion = default_orn,
                  color: Optional[Tuple[float, float, float, float]] = None,
                  scale: float = 0.2,
                  mass: Optional[float] = None,
                  use_fixed_base: bool = False,
                  physics_client_id: int = 0) -> int:
    """Create a pot object in the environment."""
    obj_id = p.loadURDF(utils.get_env_asset_path(asset_path),
                        useFixedBase=use_fixed_base,
                        globalScaling=scale,
                        physicsClientId=physics_client_id)
    p.resetBasePositionAndOrientation(obj_id,
                                      position,
                                      orientation,
                                      physicsClientId=physics_client_id)
    if color is not None:
        visual_shapes = p.getVisualShapeData(obj_id,
                                             physicsClientId=physics_client_id)
        for shape_idx, shape_data in enumerate(visual_shapes):
            link_id = shape_data[1]
            p.changeVisualShape(obj_id,
                                link_id,
                                shapeIndex=shape_idx,
                                rgbaColor=color,
                                physicsClientId=physics_client_id)

    if mass is not None:
        p.changeDynamics(obj_id,
                         -1,
                         mass=mass,
                         physicsClientId=physics_client_id)

    return obj_id


def update_object(obj_id: int,
                  position: Optional[Pose3D] = None,
                  orientation: Quaternion = default_orn,
                  color: Optional[Tuple[float, float, float, float]] = None,
                  physics_client_id: int = 0) -> None:
    """Update the position and orientation of an object."""
    if position is not None:
        p.resetBasePositionAndOrientation(obj_id,
                                          position,
                                          orientation,
                                          physicsClientId=physics_client_id)
    if color is not None:
        # Change color of all visual shapes across all links.
        # A single link can have multiple visual shapes (e.g. box primitives
        # in a URDF), so we must iterate over shape indices explicitly.
        visual_shapes = p.getVisualShapeData(obj_id,
                                             physicsClientId=physics_client_id)
        for shape_idx, shape_data in enumerate(visual_shapes):
            link_id = shape_data[1]
            p.changeVisualShape(obj_id,
                                link_id,
                                shapeIndex=shape_idx,
                                rgbaColor=color,
                                physicsClientId=physics_client_id)


def sample_collision_free_2d_positions(
        num_samples: int,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        shape_type: str,
        shape_params: Sequence[float],
        rng: np.random.Generator,
        max_tries_per_object: int = 10,
        max_tries_total: int = 1000) -> List[Tuple[float, float]]:
    """Sample collision-free 2D positions.

    This function supports two shape types:
    - "circle": requires shape_params=[radius].
    - "rectangle": requires shape_params=[width, height, theta],
      where `theta` is the rotation in radians (between -pi and pi).

    It will sample positions inside the given (x_range, y_range) such that
    none of the shapes overlap.

    Args:
        num_samples (int): Number of positions to sample.
        x_range (Tuple[float, float]): The min and max bounds for x.
        y_range (Tuple[float, float]): The min and max bounds for y.
        shape_type (str): "circle" or "rectangle".
        shape_params (Sequence[float]): Shape-specific parameters.
        rng (np.random.Generator): Random generator for reproducible sampling.
        max_tries_per_object (int, optional): Number of attempts
            per object before discarding the entire arrangement
            and restarting. Defaults to 10.
        max_tries_total (int, optional): Maximum total attempts
            before giving up. Defaults to 1000.

    Returns:
        List[Tuple[float, float]]: A list of (x, y) positions for the shapes,
        guaranteed to be collision-free.
    """
    from predicators.utils import Circle, Rectangle \
        # pylint: disable=import-outside-toplevel

    def create_geom(px: float, py: float) -> _Geom2D:
        """Create the geometry object based on shape_type and shape_params."""
        if shape_type == "circle":
            # shape_params = [radius]
            (radius, ) = shape_params
            return Circle(px, py, radius)
        if shape_type == "rectangle":
            # shape_params = [width, height, theta]
            w, h, theta = shape_params
            return Rectangle(px, py, w, h, theta)
        raise ValueError(f"Unsupported shape_type: {shape_type}")

    positions: List[Tuple[float, float]] = []
    collision_geoms: List[_Geom2D] = []

    total_tries = 0
    while True:
        positions.clear()
        collision_geoms.clear()
        for _ in range(num_samples):
            for _ in range(max_tries_per_object):
                total_tries += 1
                if total_tries > max_tries_total:
                    raise RuntimeError("Max tries exceeded. Unable to sample "
                                       "collision-free positions.")

                # Sample random position
                px = rng.uniform(x_range[0], x_range[1])
                py = rng.uniform(y_range[0], y_range[1])

                new_geom = create_geom(px, py)
                # Check intersection with existing
                if not any(new_geom.intersects(g) for g in collision_geoms):
                    # Found a valid position
                    positions.append((px, py))
                    collision_geoms.append(new_geom)
                    break
            else:
                # Failed to place this shape, restart entire process
                break
        else:
            # We successfully placed all shapes
            return positions


def create_pybullet_block(
    color: Tuple[float, float, float, float],
    half_extents: Tuple[float, float, float],
    mass: float,
    friction: float,
    position: Pose3D = (0.0, 0.0, 0.0),
    orientation: Quaternion = (0.0, 0.0, 0.0, 1.0),
    physics_client_id: int = 0,
    add_top_triangle: bool = False,
) -> int:
    """Create a box-shaped PyBullet body and return its ID."""
    collision_id = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=half_extents,
                                          physicsClientId=physics_client_id)
    visual_id = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=half_extents,
                                    rgbaColor=color,
                                    physicsClientId=physics_client_id)
    block_id = p.createMultiBody(baseMass=mass,
                                 baseCollisionShapeIndex=collision_id,
                                 baseVisualShapeIndex=visual_id,
                                 basePosition=position,
                                 baseOrientation=orientation,
                                 physicsClientId=physics_client_id)
    p.changeDynamics(block_id,
                     linkIndex=-1,
                     lateralFriction=friction,
                     spinningFriction=friction,
                     rollingFriction=friction,
                     physicsClientId=physics_client_id)

    if add_top_triangle:
        triangle_size = min(half_extents[0], half_extents[1])
        triangle_vertices = [
            [triangle_size, 0, 0],
            [-triangle_size, triangle_size, 0],
            [-triangle_size, -triangle_size, 0],
        ]
        triangle_visual_id = p.createVisualShape(
            p.GEOM_MESH,
            vertices=triangle_vertices,
            indices=[0, 1, 2],
            rgbaColor=[1, 1, 0, 1],
            physicsClientId=physics_client_id)

        p.removeBody(block_id, physicsClientId=physics_client_id)

        block_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=position,
            baseOrientation=orientation,
            linkMasses=[0],
            linkCollisionShapeIndices=[-1],
            linkVisualShapeIndices=[triangle_visual_id],
            linkPositions=[[0, 0, half_extents[2] + 0.001]],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 1]],
            physicsClientId=physics_client_id)

        p.changeDynamics(block_id,
                         linkIndex=-1,
                         lateralFriction=friction,
                         spinningFriction=friction,
                         physicsClientId=physics_client_id)

    return block_id


def create_pybullet_sphere(
    color: Tuple[float, float, float, float],
    radius: float,
    mass: float,
    friction: float,
    position: Pose3D = (0.0, 0.0, 0.0),
    orientation: Quaternion = (0.0, 0.0, 0.0, 1.0),
    physics_client_id: int = 0,
) -> int:
    """Create a sphere-shaped PyBullet body and return its ID."""
    collision_id = p.createCollisionShape(p.GEOM_SPHERE,
                                          radius=radius,
                                          physicsClientId=physics_client_id)
    visual_id = p.createVisualShape(p.GEOM_SPHERE,
                                    radius=radius,
                                    rgbaColor=color,
                                    physicsClientId=physics_client_id)
    sphere_id = p.createMultiBody(baseMass=mass,
                                  baseCollisionShapeIndex=collision_id,
                                  baseVisualShapeIndex=visual_id,
                                  basePosition=position,
                                  baseOrientation=orientation,
                                  physicsClientId=physics_client_id)
    p.changeDynamics(sphere_id,
                     linkIndex=-1,
                     lateralFriction=friction,
                     spinningFriction=friction,
                     physicsClientId=physics_client_id)
    return sphere_id
