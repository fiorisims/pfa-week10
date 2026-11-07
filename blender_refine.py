"""Runs INSIDE Blender (background mode) to finish the meshes from reference_mesh.py.

    blender --background --factory-startup --python blender_refine.py -- jobs.json

For each raw mesh: voxel-remesh it (a clean, watertight surface with even triangles,
which also removes pinches at the silhouette), smooth it, decimate it to a game-friendly
triangle count, unwrap it for ``reference_strokes.bake``, and export an FBX for Unreal.  One bad model never stops the rest; a report JSON says what happened to each.
"""

import json
import math
import os
import sys
import traceback

import bmesh
import bpy
import numpy as np

VOXELS_PER_LONG_SIDE = {"lathe": 150, "slab": 200, "inflate": 170, "ground": 170, "composite": 220, "cluster": 130}
SMOOTH_STEPS = {"lathe": 3, "slab": 2, "inflate": 8, "ground": 6, "composite": 3, "cluster": 12}
MAX_TRIS = 14000
MIN_TRIS = 300
# the Model style slider (0 = generic low poly, 1 = as painted)
VOXELS_LOW = 14
TRIS_LOW = 60
FLAT_BELOW = 0.6             # below this, flat facets instead of smooth shading
HULL_BELOW = 0.2             # below this, simple gem-like convex forms


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)


def read_obj(path):
    verts, faces = [], []
    with open(path, encoding="utf-8") as src:
        for line in src:
            if line.startswith("v "):
                verts.append(tuple(float(v) for v in line.split()[1:4]))
            elif line.startswith("f "):
                faces.append(tuple(int(p.split("/")[0]) - 1 for p in line.split()[1:]))
    return verts, faces


def make_object(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def apply_modifier(obj, kind, **props):
    mod = obj.modifiers.new("step", kind)
    for key, value in props.items():
        setattr(mod, key, value)
    bpy.ops.object.modifier_apply(modifier=mod.name)


def unwrap(obj):
    """An atlas with room for the whole surface: the painting and strokes get baked into it."""
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.006)
    bpy.ops.object.mode_set(mode="OBJECT")


def save_surface(obj, path):
    """What reference_strokes.bake needs: every surface point's position, normal and atlas spot."""
    mesh = obj.data
    mesh.calc_loop_triangles()
    count = len(mesh.vertices)
    positions = np.zeros(count * 3, np.float32)
    mesh.vertices.foreach_get("co", positions)
    normals = np.zeros(count * 3, np.float32)
    mesh.vertex_normals.foreach_get("vector", normals)
    tris = np.zeros(len(mesh.loop_triangles) * 3, np.int32)
    mesh.loop_triangles.foreach_get("vertices", tris)
    loops = np.zeros(len(mesh.loop_triangles) * 3, np.int32)
    mesh.loop_triangles.foreach_get("loops", loops)
    uvs = np.zeros(len(mesh.loops) * 2, np.float32)
    mesh.uv_layers.active.data.foreach_get("uv", uvs)
    uvs = uvs.reshape(-1, 2)
    np.savez(path, positions=positions.reshape(-1, 3), normals=normals.reshape(-1, 3),
             tris=tris.reshape(-1, 3), tri_uvs=uvs[loops].reshape(-1, 3, 2))


def triangle_count(obj):
    obj.data.calc_loop_triangles()
    return len(obj.data.loop_triangles)


def to_convex_hull(obj):
    """Replace the surface with its convex hull: that is what reads as a generic low-poly form."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    result = bmesh.ops.convex_hull(bm, input=bm.verts)
    keep = {g for g in result["geom"] if isinstance(g, bmesh.types.BMFace)}
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f not in keep], context="FACES_ONLY")
    bmesh.ops.delete(bm, geom=[e for e in bm.edges if not e.link_faces], context="EDGES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def remesh_detail(obj, method, detail, dims):
    """Voxel remesh at a resolution set by the style slider; finer if that lost the shape."""
    long_side = float(dims.max())
    voxels = VOXELS_LOW * (VOXELS_PER_LONG_SIDE.get(method, 170) / float(VOXELS_LOW)) ** detail
    # thin objects need voxels smaller than their thinnest side, or they vanish
    voxel = max(min(long_side / voxels, float(max(1e-6, dims.min())) / 3.0), long_side / 500.0)
    original = obj.data.copy()
    for _attempt in range(4):
        apply_modifier(obj, "REMESH", mode="VOXEL", voxel_size=voxel, use_smooth_shade=detail >= FLAT_BELOW)
        lost = np.abs(np.array(obj.dimensions) - dims) / np.maximum(dims, 1e-6)
        if len(obj.data.vertices) >= 12 and float(lost.max()) <= 0.3:
            bpy.data.meshes.remove(original)
            return
        replaced = obj.data
        obj.data = original.copy()
        bpy.data.meshes.remove(replaced)
        voxel /= 2.0
    bpy.data.meshes.remove(original)
    raise RuntimeError("the remesh lost the shape")


def import_found(path):
    """Bring a found model in as one object."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise RuntimeError("can't read %s files" % ext)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("the file has no mesh")
    # Sketchfab files stand the model up through rotated parent empties: bake those rotations in
    # before the empties go, or the model ends up lying on its side
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    for o in list(bpy.context.scene.objects):
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def fit_found(obj, target_aspect, fit):
    """Stand it on the ground, turn it to match the painting, and lean its proportions toward it.

    target_aspect: painted height / width.  fit: 0 = keep the model's proportions, 1 = match exactly.
    """
    dims = np.array(obj.dimensions, np.float64)
    if dims.max() <= 0:
        raise RuntimeError("the model is empty")
    # the painting shows one side: face whichever side's proportions are closer to what was painted
    front_x = dims[2] / max(1e-6, dims[0])
    front_y = dims[2] / max(1e-6, dims[1])
    if abs(math.log(front_y / target_aspect)) + 0.15 < abs(math.log(front_x / target_aspect)):
        obj.rotation_euler = (0.0, 0.0, math.radians(90.0))
        bpy.ops.object.transform_apply(rotation=True)
        dims = np.array(obj.dimensions, np.float64)
    current = dims[2] / max(1e-6, dims[0])
    stretch = (current / target_aspect) ** max(0.0, min(1.0, fit))
    stretch = max(0.7, min(1.45, stretch))
    obj.scale = (stretch, 1.0, 1.0)
    bpy.ops.object.transform_apply(scale=True)
    # base centre at the origin, like the generated models
    co = np.zeros(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    shift = np.array([-(co[:, 0].min() + co[:, 0].max()) / 2.0, -(co[:, 1].min() + co[:, 1].max()) / 2.0, -co[:, 2].min()])
    obj.data.vertices.foreach_set("co", (co + shift).reshape(-1))
    obj.data.update()
    # one sensible scale: 100 units tall, whatever the source used
    factor = 100.0 / max(1e-6, float(np.array(obj.dimensions)[2]))
    obj.scale = (factor, factor, factor)
    bpy.ops.object.transform_apply(scale=True)


def adopt(entry):
    clear_scene()
    obj = import_found(entry["source_file"])
    fit_found(obj, float(entry.get("target_aspect", 1.0)), float(entry.get("fit", 0.6)))
    detail = float(entry.get("detail", 1.0))
    own_look = bool(entry.get("own_look"))
    # Low poly for a found model means fewer triangles on its own surface.  A coarse voxel rebuild,
    # right for outline models, shatters thin real-world shapes (a canoe hull came out in pieces).
    if detail < HULL_BELOW and not own_look:
        to_convex_hull(obj)
    full = int(max(MIN_TRIS, min(MAX_TRIS, entry.get("budget", 6000))))
    budget = int(TRIS_LOW * (full / float(TRIS_LOW)) ** detail)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode="OBJECT")
    tris = triangle_count(obj)
    if tris > budget:
        apply_modifier(obj, "DECIMATE", decimate_type="COLLAPSE", ratio=budget / float(tris), use_collapse_triangulate=True)
    if detail >= FLAT_BELOW:
        bpy.ops.object.shade_smooth()
    else:
        bpy.ops.object.shade_flat()
    stem = os.path.join(os.path.dirname(entry["front_art"]), entry["name"])
    size = [float(v) for v in obj.dimensions]
    if own_look:
        # keep the model's own materials: glTF carries its textures with it into Unreal
        path = stem + "_own.glb"
        bpy.ops.export_scene.gltf(filepath=path, use_selection=True, export_format="GLB", export_apply=True)
        return {"mesh": path, "tris": triangle_count(obj), "size": size}
    unwrap(obj)
    surface = stem + "_surface.npz"
    save_surface(obj, surface)
    path = stem + ".fbx"
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, apply_unit_scale=True, apply_scale_options="FBX_SCALE_NONE",
        axis_forward="-Y", axis_up="Z", bake_space_transform=False, object_types={"MESH"},
        use_mesh_modifiers=False, mesh_smooth_type="FACE", add_leaf_bones=False, bake_anim=False,
        path_mode="COPY", embed_textures=False)
    return {"mesh": path, "surface": surface, "tris": triangle_count(obj), "size": size}


def finish_block(obj, entry):
    """A building keeps its exact squared-off shape: flat walls merged into single faces, no smoothing."""
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-4)
    bpy.ops.object.mode_set(mode="OBJECT")
    apply_modifier(obj, "DECIMATE", decimate_type="DISSOLVE", angle_limit=math.radians(1.0))
    bpy.ops.object.shade_flat()
    area = max(1e-6, sum(p.area for p in obj.data.polygons))
    unwrap(obj)
    stem = os.path.splitext(entry["mesh"])[0]
    surface = stem + "_surface.npz"
    save_surface(obj, surface)
    path = stem + ".fbx"
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, apply_unit_scale=True, apply_scale_options="FBX_SCALE_NONE",
        axis_forward="-Y", axis_up="Z", bake_space_transform=False, object_types={"MESH"},
        use_mesh_modifiers=False, mesh_smooth_type="FACE", add_leaf_bones=False, bake_anim=False,
        path_mode="COPY", embed_textures=False)
    return {"mesh": path, "surface": surface, "tris": triangle_count(obj), "area": area}


def refine(entry):
    if entry.get("method") == "adopt":
        return adopt(entry)
    clear_scene()
    verts, faces = read_obj(entry["mesh"])
    obj = make_object(entry["name"], verts, faces)
    method = entry["method"]
    detail = float(entry.get("detail", 1.0))
    if method == "block":
        return finish_block(obj, entry)
    remesh_detail(obj, method, detail, np.array(obj.dimensions))
    if detail < HULL_BELOW:
        to_convex_hull(obj)
    # smoothing rounds forms off, which low-poly shapes shouldn't be
    steps = int(round(SMOOTH_STEPS.get(method, 5) * detail))
    if steps and detail >= FLAT_BELOW:
        apply_modifier(obj, "SMOOTH", factor=0.6, iterations=steps)
    tris = triangle_count(obj)
    area = max(1e-6, sum(p.area for p in obj.data.polygons))
    # budget grows with the object's on-screen share; tiny props stay cheap
    full = int(max(MIN_TRIS, min(MAX_TRIS, entry.get("budget", 6000))))
    budget = int(TRIS_LOW * (full / float(TRIS_LOW)) ** detail)
    if tris > budget:
        apply_modifier(obj, "DECIMATE", decimate_type="COLLAPSE", ratio=budget / float(tris), use_collapse_triangulate=True)
    if detail >= FLAT_BELOW:
        bpy.ops.object.shade_smooth()
    else:
        bpy.ops.object.shade_flat()                # visible facets are the look
    unwrap(obj)
    stem = os.path.splitext(entry["mesh"])[0]
    surface = stem + "_surface.npz"
    save_surface(obj, surface)
    path = stem + ".fbx"
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, apply_unit_scale=True, apply_scale_options="FBX_SCALE_NONE",
        axis_forward="-Y", axis_up="Z", bake_space_transform=False, object_types={"MESH"},
        use_mesh_modifiers=False, mesh_smooth_type="FACE", add_leaf_bones=False, bake_anim=False,
        path_mode="COPY", embed_textures=False)
    return {"mesh": path, "surface": surface, "tris": triangle_count(obj), "area": area}


def main():
    jobs_path = sys.argv[sys.argv.index("--") + 1]
    with open(jobs_path, encoding="utf-8") as src:
        entries = json.load(src)
    report = {}
    for entry in entries:
        try:
            report[entry["name"]] = refine(entry)
        except Exception as error:
            traceback.print_exc()
            report[entry["name"]] = {"error": str(error)[:200]}
    out = jobs_path.replace("blender_jobs", "blender_report")
    with open(out, "w", encoding="utf-8") as dst:
        json.dump(report, dst)


main()
