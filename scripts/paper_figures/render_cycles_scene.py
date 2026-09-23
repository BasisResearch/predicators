"""Render an exported PyBullet visual scene with Blender Cycles on the CPU.

Run with a Python environment containing bpy==4.5.3 and pycollada. No
physics or task generation runs here; all object transforms come from
JSON.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

# Limit Blender/TBB discovery before importing the module on shared hosts.
os.sched_setaffinity(
    0,
    sorted(os.sched_getaffinity(0))
    [:int(os.environ.get('EMPIRIC_RENDER_THREADS', '8'))])
# pylint: disable=wrong-import-position
# The Blender modules exist only in the renderer environment, not the project
# environment used for static checks.
# isort: off
import bpy  # type: ignore[import-not-found] # pylint: disable=import-error
# The PyPI module registers Blender's companion modules lazily.
_ = bpy.app.version_string
import bmesh  # type: ignore[import-not-found] # pylint: disable=import-error
from mathutils import (  # type: ignore[import-not-found] # pylint: disable=import-error
    Matrix, Quaternion, Vector)
# isort: on
# pylint: enable=wrong-import-position

REPO = Path(__file__).resolve().parents[2]


def resolve_mesh(path: str) -> Path:
    """Resolve archived cluster mesh paths against the current checkout."""
    source = Path(path)
    if source.exists():
        return source
    marker = "/predicators/envs/"
    if marker in path:
        candidate = REPO / "predicators/envs" / path.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def linear(value: float) -> float:
    """Convert one sRGB channel to linear light."""
    return value / 12.92 if value <= .04045 else ((value + .055) / 1.055)**2.4


def material_for(shape: Dict[str, Any]) -> Any:
    """Create the procedural Cycles material for one exported shape."""
    rgba = shape['rgba']
    identity = (shape['mesh'] + ' ' + shape['name']).lower()
    wood = 'table' in identity
    mat = bpy.data.materials.new(
        f"material_{shape['body']}_{shape['link']}_{shape['index']}")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    color = tuple(linear(v) for v in rgba[:3]) + (1, )
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = .48
    bsdf.inputs['Specular IOR Level'].default_value = .28
    if shape['name'] == 'cup':
        # The deep, saturated-blue cavity receives almost no indirect light in
        # these compact studio scenes. A very small ambient material response
        # keeps its interior readable without changing its geometry or the
        # lighting on any other object.
        bsdf.inputs['Emission Color'].default_value = color
        bsdf.inputs['Emission Strength'].default_value = .50
    if shape['kind'] == 2:  # colored sphere, including balloons
        bsdf.inputs['Roughness'].default_value = .29
        bsdf.inputs['Coat Weight'].default_value = .22
        bsdf.inputs['Coat Roughness'].default_value = .25
    if wood:
        # Procedural grain adds surface appearance while preserving geometry.
        tex = nodes.new('ShaderNodeTexNoise')
        tex.inputs['Scale'].default_value = 5
        tex.inputs['Detail'].default_value = 2
        coord = nodes.new('ShaderNodeTexCoord')
        scale = nodes.new('ShaderNodeVectorMath')
        scale.operation = 'MULTIPLY'
        scale.inputs[1].default_value = (2, 95, 15)
        links.new(coord.outputs['Generated'], scale.inputs[0])
        links.new(scale.outputs['Vector'], tex.inputs['Vector'])
        ramp = nodes.new('ShaderNodeValToRGB')
        base = (.68, .49, .29) if 'table' in identity else rgba[:3]
        ramp.color_ramp.elements[0].position = .12
        ramp.color_ramp.elements[0].color = tuple(
            linear(v * .85) for v in base) + (1, )
        ramp.color_ramp.elements[1].position = .88
        ramp.color_ramp.elements[1].color = tuple(
            linear(min(1, v * 1.13)) for v in base) + (1, )
        links.new(tex.outputs['Fac'], ramp.inputs['Fac'])
        links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = .10
        bump.inputs['Distance'].default_value = .00025
        links.new(tex.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
        bsdf.inputs['Roughness'].default_value = .58
    if rgba[3] < .99:
        transparent = nodes.new('ShaderNodeBsdfTransparent')
        mix = nodes.new('ShaderNodeMixShader')
        mix.inputs[0].default_value = rgba[3]
        links.new(transparent.outputs[0], mix.inputs[1])
        links.new(bsdf.outputs[0], mix.inputs[2])
        links.new(mix.outputs[0],
                  nodes.get('Material Output').inputs['Surface'])
    return mat


def primitive(shape: Dict[str, Any]) -> List[Any]:
    """Create Blender geometry for one PyBullet visual primitive."""
    kind, dims = shape['kind'], shape['dimensions']
    if kind == 5 and 'vertices' in shape:
        mesh = bpy.data.meshes.new('Recorded procedural visual')
        indices = shape['indices']
        mesh.from_pydata(shape['vertices'], [],
                         [indices[i:i + 3] for i in range(0, len(indices), 3)])
        mesh.update()
        obj = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.collection.objects.link(obj)
    elif kind == 3:  # GEOM_BOX: visual dimensions are full extents
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj = bpy.context.object
        for vertex in obj.data.vertices:
            vertex.co.x *= dims[0]
            vertex.co.y *= dims[1]
            vertex.co.z *= dims[2]
    elif kind == 2:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=64,
                                             ring_count=32,
                                             radius=dims[0])
        obj = bpy.context.object
    elif kind in (4, 7):  # GEOM_CYLINDER or GEOM_CAPSULE
        length, radius = dims[:2]
        if kind == 4:
            bpy.ops.mesh.primitive_cylinder_add(vertices=64,
                                                radius=radius,
                                                depth=length)
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32,
                                                 ring_count=24,
                                                 radius=radius)
        obj = bpy.context.object
        if kind == 7:
            for vertex in obj.data.vertices:
                vertex.co.z += length / 2 if vertex.co.z >= 0 else -length / 2
    elif kind == 6:
        bpy.ops.mesh.primitive_plane_add(size=200)
        obj = bpy.context.object
    else:
        raise ValueError(f'Unsupported visual primitive {kind}: {shape}')
    if kind in (2, 4, 7):
        for face in obj.data.polygons:
            face.use_smooth = kind != 4 or len(face.vertices) == 4
    return [obj]


def add_shape(shape: Dict[str, Any]) -> int:
    """Add one exported visual shape and return its mesh object count."""
    if shape['kind'] == 0:
        raise ValueError(
            'Invalid visual metadata: export from a client without EGL')
    if shape['kind'] == 3 and max(shape['dimensions']) == 0:
        return 0  # Empty base-link visual in the switch URDF.
    path = shape['mesh']
    if path:
        before = set(bpy.data.objects)
        mesh_path = resolve_mesh(path)
        if mesh_path.suffix.lower() == '.dae':
            import collada  # type: ignore[import-not-found] # pylint: disable=import-error,import-outside-toplevel
            document = collada.Collada(str(mesh_path),
                                       ignore=[collada.DaeBrokenRefError])
            assert document.assetInfo.upaxis == 'Z_UP', mesh_path
            for geometry in document.scene.objects('geometry'):
                for part in geometry.primitives():
                    if hasattr(part, 'triangleset'):
                        part = part.triangleset()
                    mesh = bpy.data.meshes.new(geometry.original.id)
                    vertices = part.vertex * (document.assetInfo.unitmeter
                                              or 1)
                    mesh.from_pydata(vertices.tolist(), [],
                                     part.vertex_index.tolist())
                    mesh.update()
                    # Collada stores authored per-corner normals. Preserve them
                    # rather than smoothing across every triangle, which makes
                    # planar robot panels show diagonal shading artifacts.
                    if (part.normal is not None
                            and part.normal_index is not None):
                        normal_indices = part.normal_index.reshape(-1)
                        normals = [
                            tuple(part.normal[i]) for i in normal_indices
                        ]
                        if len(normals) == len(mesh.loops):
                            mesh.normals_split_custom_set(normals)
                    obj = bpy.data.objects.new(mesh.name, mesh)
                    bpy.context.collection.objects.link(obj)
                    for face in mesh.polygons:
                        face.use_smooth = part.normal is not None
        elif mesh_path.suffix.lower() == '.obj':
            bpy.ops.wm.obj_import(filepath=str(mesh_path),
                                  forward_axis='Y',
                                  up_axis='Z')
        elif mesh_path.suffix.lower() == '.stl':
            bpy.ops.wm.stl_import(filepath=str(mesh_path),
                                  forward_axis='Y',
                                  up_axis='Z')
        else:
            raise ValueError(f'Unsupported visual mesh format: {path}')
        objects = [
            obj for obj in set(bpy.data.objects) - before if obj.type == 'MESH'
        ]
        scaling = Vector(shape['dimensions'])
    else:
        objects = primitive(shape)
        scaling = Vector((1, 1, 1))
    assert objects, shape
    x, y, z, w = shape['quaternion_xyzw']
    world = Matrix.LocRotScale(Vector(shape['position']),
                               Quaternion((w, x, y, z)), scaling)
    material = material_for(shape)
    for obj in objects:
        if path and Path(path).suffix.lower() == '.obj':
            # PartNet includes coincident opposite-facing triangles. They
            # self-shadow to black in a path tracer despite looking fine in
            # Bullet. Weld and deduplicate surfaces, then orient face normals.
            mesh = bmesh.new()
            mesh.from_mesh(obj.data)
            bmesh.ops.remove_doubles(mesh, verts=list(mesh.verts), dist=1e-7)
            mesh.verts.index_update()
            seen: Set[Tuple[int, ...]] = set()
            duplicates: List[Any] = []
            for face in mesh.faces:
                key = tuple(sorted(v.index for v in face.verts))
                if key in seen:
                    duplicates.append(face)
                seen.add(key)
            bmesh.ops.delete(mesh, geom=duplicates, context='FACES_ONLY')
            bmesh.ops.recalc_face_normals(mesh, faces=list(mesh.faces))
            mesh.to_mesh(obj.data)
            mesh.free()
            obj.data.normals_split_custom_set([(0., 0., 0.)] *
                                              len(obj.data.loops))
            obj.data.update()
            if '/switch/' in path:
                for face in obj.data.polygons:
                    face.use_smooth = False
                obj['switch_color'] = ','.join(str(v) for v in shape['rgba'])
        obj.matrix_world = world @ obj.matrix_world
        obj.name = (f"body_{shape['body']}_link_{shape['link']}_"
                    f"visual_{shape['index']}_{obj.name}")
        obj.data.materials.clear()
        obj.data.materials.append(material)
        for face in obj.data.polygons:
            face.material_index = 0
        if shape['kind'] == 3 and min(shape['dimensions']) > .01:
            obj['box_union_group'] = str(
                (shape['body'], shape['link'], shape['rgba']))
        if shape['kind'] == 3 and min(shape['dimensions']) > .01:
            bevel = obj.modifiers.new('Submillimeter visual edge finish',
                                      'BEVEL')
            bevel.width = min(.0006, min(shape['dimensions']) / 100)
            bevel.segments = 3
            bevel.affect = 'EDGES'
            bevel.harden_normals = True
            normal = obj.modifiers.new('Preserve planar faces',
                                       'WEIGHTED_NORMAL')
            normal.keep_sharp = True
    return len(objects)


def union_compound_boxes() -> None:
    """Render compound solid boxes as a union, removing coplanar wall
    patches."""
    groups: Dict[str, List[Any]] = {}
    for obj in list(bpy.data.objects):
        if obj.type == 'MESH' and 'box_union_group' in obj:
            groups.setdefault(obj['box_union_group'], []).append(obj)
    for objects in groups.values():
        if len(objects) < 2:
            continue
        for obj in objects:
            obj.modifiers.clear()
        target = objects[0]
        bpy.context.view_layer.objects.active = target
        for other in objects[1:]:
            modifier = target.modifiers.new('Union redundant solid surfaces',
                                            'BOOLEAN')
            modifier.operation = 'UNION'
            modifier.solver = 'EXACT'
            modifier.object = other
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            bpy.data.objects.remove(other, do_unlink=True)
        bevel = target.modifiers.new('Submillimeter visual edge finish',
                                     'BEVEL')
        bevel.width = .0003
        bevel.segments = 3
        # Boolean unions may create large n-gons. Flat normals keep their
        # internal triangulation from appearing as diagonal lines on the jug.
        bevel.harden_normals = False
        for face in target.data.polygons:
            face.use_smooth = False


def merge_overlapping_tables(
        shapes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Use the exact rectangular union of aligned touching table volumes.

    Boil's tables overlap over half a tabletop. Rendering both coplanar
    skins produces a false seam. This replaces only their redundant
    visual surfaces.
    """
    tables = [s for s in shapes if s['name'] == 'table' and s['kind'] == 3]
    if len(tables) != 2:
        return shapes
    first, second = tables
    assert first['quaternion_xyzw'] == second['quaternion_xyzw']
    assert first['dimensions'] == second['dimensions']
    x, y, z, w = first['quaternion_xyzw']
    rotation = Quaternion((w, x, y, z)).normalized()
    delta = rotation.inverted() @ (Vector(second['position']) -
                                   Vector(first['position']))
    axis = max(range(3), key=lambda i: abs(delta[i]))
    assert all(abs(delta[i]) < 1e-6 for i in range(3) if i != axis)
    if abs(delta[axis]) > first['dimensions'][axis] + 1e-6:
        return shapes  # A real gap must remain visible.
    combined = dict(first, dimensions=list(first['dimensions']))
    combined['dimensions'][axis] += abs(delta[axis])
    combined['position'] = list(
        (Vector(first['position']) + Vector(second['position'])) / 2)
    return [s for s in shapes if s not in tables] + [combined]


def consolidate_switch_surfaces() -> None:
    """Remove coincident surfaces across PartNet pieces without moving
    vertices."""
    groups: Dict[str, List[Any]] = {}
    for obj in list(bpy.data.objects):
        if obj.type == 'MESH' and 'switch_color' in obj:
            groups.setdefault(obj['switch_color'], []).append(obj)
    for objects in groups.values():
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        bpy.ops.object.join()
        obj = bpy.context.object
        mesh = bmesh.new()
        mesh.from_mesh(obj.data)
        bmesh.ops.remove_doubles(mesh, verts=list(mesh.verts), dist=1e-7)
        mesh.verts.index_update()
        seen: Set[Tuple[int, ...]] = set()
        duplicates: List[Any] = []
        for face in mesh.faces:
            key = tuple(sorted(v.index for v in face.verts))
            if key in seen:
                duplicates.append(face)
            seen.add(key)
        bmesh.ops.delete(mesh, geom=duplicates, context='FACES_ONLY')
        bmesh.ops.recalc_face_normals(mesh, faces=list(mesh.faces))
        mesh.to_mesh(obj.data)
        mesh.free()
        obj.data.normals_split_custom_set([(0., 0., 0.)] * len(obj.data.loops))
        for face in obj.data.polygons:
            face.use_smooth = False
        obj.data.update()


def area_light(name: str, position: Sequence[float], target: Sequence[float],
               energy: float, size: float) -> None:
    """Add a disk area light aimed at a target point."""
    data = bpy.data.lights.new(name, 'AREA')
    data.energy = energy
    data.shape = 'DISK'
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat(
        '-Z', 'Y').to_euler()


def main() -> None:
    """Render one exported scene and write its provenance report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=96)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--save-blend', action='store_true')
    args = parser.parse_args()
    source = args.scene.read_bytes()
    exported = json.loads(source)
    for mesh, digest in exported['mesh_sha256'].items():
        resolved = resolve_mesh(mesh)
        assert hashlib.sha256(
            resolved.read_bytes()).hexdigest() == digest, mesh
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.seed = 0
    scene.cycles.max_bounces = 8
    scene.cycles.transparent_max_bounces = 16
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = args.threads
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Medium High Contrast'
    scene.view_settings.exposure = -1.0
    camera = exported['camera']
    scene.render.resolution_x = round(camera['width'] * args.scale)
    scene.render.resolution_y = round(camera['height'] * args.scale)
    scene.render.resolution_percentage = 100
    count = 0
    for shape in merge_overlapping_tables(exported['shapes']):
        count += add_shape(shape)
    union_compound_boxes()
    consolidate_switch_surfaces()
    view = Matrix([camera['view'][i::4] for i in range(4)])
    projection = camera['projection']
    data = bpy.data.cameras.new('Recorded camera')
    data.type = 'PERSP'
    data.sensor_fit = 'VERTICAL'
    data.sensor_height = 24
    data.lens = data.sensor_height * projection[5] / 2
    data.clip_start = .02
    data.clip_end = 100
    obj = bpy.data.objects.new('Recorded camera', data)
    bpy.context.collection.objects.link(obj)
    obj.matrix_world = view.inverted()
    scene.camera = obj
    scene.world = bpy.data.worlds.new('Neutral studio')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (
        .78, .84, 1.0, 1)
    scene.world.node_tree.nodes['Background'].inputs[
        'Strength'].default_value = .12
    target = (.7, 1.2, .55)
    area_light('Large soft key', (-.6, -.2, 2.8), target, 110, 1.0)
    area_light('Soft fill', (1.8, .4, 2.1), target, 20, 1.5)
    area_light('Top rim', (.5, 2.3, 2.4), target, 45, 1.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.output.resolve())
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(
            filepath=str(args.output.with_suffix('.blend').resolve()))
    bpy.ops.render.render(write_still=True)
    report = dict(
        generated_by='scripts/render_cycles_scene.py; do not edit manually',
        blender_version=bpy.app.version_string,
        engine='CYCLES',
        device='CPU',
        samples=args.samples,
        scene_source=str(args.scene),
        scene_sha256=hashlib.sha256(source).hexdigest(),
        output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        visual_shapes=len(exported['shapes']),
        blender_mesh_objects=count,
        geometry=('Recorded world transforms; '
                  'box edges receive at most 0.6 mm visual bevels'),
        appearance=('Procedural materials, soft area lighting, AgX; '
                    'duplicate surfaces removed'),
        physics_steps=0)
    args.output.with_suffix('.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
