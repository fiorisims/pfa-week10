"""Reference World Builder - build a world from ONE flat image.

``reference_scene.py`` (system Python) estimates depth with Depth Anything V2,
finds the horizon, solves the painting's camera and rebuilds the scene in
real perspective.  This module turns that into the level:

* terrain shaped by the depth map, with the painting PROJECTED onto it from the
  painting's own camera (spots the camera could not see use matching colours);
* trees / bushes / rocks wherever the depth map shows something standing up,
  at their real size, with their own piece of the painting on them;
* a backdrop of everything beyond the terrain, placed so its bottom edge lines
  up with the terrain's far edge when seen from the start;
* a colour grade tuned to the painting, and the player starting at the camera.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import unreal

import rwb_layers

TERRAIN_MATERIAL = "M_RWB_TerrainPaint_v1"


def _wb():
    import world_builder
    return world_builder


def _build_terrain_paint(material):
    x, link, mask = rwb_layers._x, rwb_layers._link, rwb_layers._mask
    tex = x(material, unreal.MaterialExpressionTextureSampleParameter2D, -600, 0)   # uses the mesh UVs
    tex.set_editor_property("parameter_name", "Art")
    default = (unreal.load_asset("/Engine/EngineResources/WhiteSquareTexture")
               or unreal.load_asset("/Engine/EngineResources/DefaultTexture"))
    if default:
        tex.set_editor_property("texture", default)
    vertex = x(material, unreal.MaterialExpressionVertexColor, -800, 300)
    vrgb = mask(material, vertex, -600, 300)
    blend = rwb_layers._scalar_param(material, "ArtBlend", 0.9, -800, 480)
    amount = x(material, unreal.MaterialExpressionMultiply, -400, 420)
    link(vertex, "A", amount, "A")
    link(blend, "", amount, "B")
    mix = x(material, unreal.MaterialExpressionLinearInterpolate, -200, 120)
    link(vrgb, "", mix, "A")
    link(tex, "RGB", mix, "B")
    link(amount, "", mix, "Alpha")
    unreal.MaterialEditingLibrary.connect_material_property(mix, "", unreal.MaterialProperty.MP_BASE_COLOR)
    rough = x(material, unreal.MaterialExpressionConstant, -200, 320)
    rough.set_editor_property("r", 0.92)
    unreal.MaterialEditingLibrary.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)


def paint_settings(texture):
    """Make sure a texture is treated as paint, whatever the importer guessed.

    Bluish pictures get auto-detected as normal maps (water came out lime green).
    Don't add never_stream here: setting it after import left big models white until a restart.
    """
    wanted = {"compression_settings": unreal.TextureCompressionSettings.TC_DEFAULT, "srgb": True}
    changed = False
    for prop, value in wanted.items():
        if texture.get_editor_property(prop) != value:
            texture.set_editor_property(prop, value)
            changed = True
    if changed:
        unreal.EditorAssetLibrary.save_loaded_asset(texture, False)
    return texture


def _import_texture(path, folder, name, cache_dir):
    record_path = Path(cache_dir) / "unreal_imports.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception:
        record = {}
    stamp = "%s|%d|%d" % (folder, os.path.getsize(path), int(os.path.getmtime(path)))
    asset_path = folder + "/" + name
    if record.get(name) != stamp or not unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        task = unreal.AssetImportTask()
        task.set_editor_property("filename", str(path))
        task.set_editor_property("destination_path", folder)
        task.set_editor_property("destination_name", name)
        task.set_editor_property("replace_existing", True)
        task.set_editor_property("automated", True)
        task.set_editor_property("save", True)
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        texture = unreal.load_asset(asset_path)
        if texture:
            for prop in ("address_x", "address_y"):
                try:
                    texture.set_editor_property(prop, unreal.TextureAddress.TA_CLAMP)
                except Exception:
                    pass
            unreal.EditorAssetLibrary.save_loaded_asset(texture, False)
        record[name] = stamp
        try:
            record_path.write_text(json.dumps(record), encoding="utf-8")
        except Exception:
            pass
    texture = unreal.load_asset(asset_path)
    return paint_settings(texture) if texture else texture


def _srgb(channel):
    v = max(0.0, min(255.0, float(channel))) / 255.0
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def _terrain(scene, analysis, values, folder):
    wb = _wb()
    t = scene["terrain"]
    n = int(t["n"])
    xs, ys, heights = t["xs"], t["ys"], t["heights"]
    dx = (xs[-1] - xs[0]) / float(n - 1)
    dy = (ys[-1] - ys[0]) / float(n - 1)
    verts, normals, uvs, colors, tangents = [], [], [], [], []
    for iy in range(n):
        for ix in range(n):
            z = heights[iy][ix]
            verts.append(unreal.Vector(xs[ix], ys[iy], z))
            left = heights[iy][max(0, ix - 1)]
            right = heights[iy][min(n - 1, ix + 1)]
            down = heights[max(0, iy - 1)][ix]
            up = heights[min(n - 1, iy + 1)][ix]
            normal = unreal.Vector(-(right - left) / (2.0 * dx), -(up - down) / (2.0 * dy), 1.0)
            normal.normalize()
            normals.append(normal)
            u, v = t["uv"][iy][ix]
            uvs.append(unreal.Vector2D(u, v))
            r, g, b = t["colors"][iy][ix]
            colors.append(unreal.LinearColor(_srgb(r), _srgb(g), _srgb(b), float(t["vis"][iy][ix])))
            tangents.append(unreal.ProcMeshTangent())
    triangles = []
    for iy in range(n - 1):
        for ix in range(n - 1):
            a = iy * n + ix
            # counter-clockwise seen from above: procedural meshes cull the other way round
            triangles.extend([a, a + n + 1, a + 1, a, a + n, a + n + 1])
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.Actor, unreal.Vector(0, 0, 0))
    component = unreal.new_object(unreal.ProceduralMeshComponent, outer=actor, name="RWBTerrain")
    actor.set_editor_property("root_component", component)
    component.set_editor_property("mobility", unreal.ComponentMobility.STATIC)
    component.create_mesh_section_linear_color(0, verts, triangles, normals, uvs, [], [], [], colors, tangents, True)
    material = None
    try:
        master = rwb_layers._material(TERRAIN_MATERIAL, _build_terrain_paint)
        texture = _import_texture(t["texture"], folder + "/Art", "T_%s_scene" % rwb_layers._safe(analysis.get("name", "ref"), 24),
                                  scene["cache"])
        if master and texture:
            material = rwb_layers._mic(folder + "/Materials", "MI_%s_terrain" % rwb_layers._safe(analysis.get("name", "ref"), 24),
                                       master, scalars={"ArtBlend": float(values.get("art_blend", 0.8))},
                                       textures={"Art": texture})
    except Exception as error:
        unreal.log_warning("Reference World Builder: projected terrain material unavailable (%s); using colours." % error)
    component.set_material(0, material or wb._terrain_material())
    component.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS)
    wb._tag_actor(actor, wb.GENERATED_TAG, wb.TERRAIN_TAG, label="RWB_Terrain")
    return actor


def _grade(analysis):
    """Unbound post-process volume: tint, saturation and a soft vignette from the painting."""
    wb = _wb()
    palette = analysis.get("palette") or []
    if palette:
        total = [0.0, 0.0, 0.0]
        for entry in palette:
            for i in range(3):
                total[i] += entry["rgb"][i]
        avg = [c / len(palette) for c in total]
    else:
        avg = [200.0, 200.0, 200.0]
    peak = max(1.0, max(avg))
    tint = unreal.LinearColor(*[0.78 + 0.22 * (c / peak) for c in avg], 1.0)
    sat = max(0.85, min(1.25, 0.9 + analysis.get("mood", {}).get("average_saturation", 0.3) * 0.6))
    volume = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0, 0, 0))
    if not volume:
        return
    try:
        volume.set_editor_property("unbound", True)
        settings = volume.get_editor_property("settings")
        for name, value in (
            ("override_scene_color_tint", True), ("scene_color_tint", tint),
            ("override_color_saturation", True), ("color_saturation", unreal.Vector4(sat, sat, sat, 1.0)),
            ("override_vignette_intensity", True), ("vignette_intensity", 0.35),
        ):
            try:
                settings.set_editor_property(name, value)
            except Exception as error:
                unreal.log_warning("Reference World Builder grade: %s skipped (%s)" % (name, error))
        volume.set_editor_property("settings", settings)
    except Exception as error:
        unreal.log_warning("Reference World Builder: colour grade skipped (%s)" % error)
    wb._tag_actor(volume, wb.GENERATED_TAG, wb.LIGHT_TAG, label="RWB_ColourGrade")


def _player_start(scene, heights, world_x, world_y):
    wb = _wb()
    cam = scene["camera"]
    y = cam["y"] - 350.0                       # the third-person camera sits behind the character
    u = 0.5
    v = max(0.0, min(1.0, y / world_y + 0.5))
    z = wb._sample_height(heights, u, v) + 120.0
    location = unreal.Vector(cam["x"], y, z)
    starts = [a for a in wb._all_actors() if isinstance(a, unreal.PlayerStart)]
    if starts:
        start = starts[0]
        start.set_actor_location(location, False)
        start.set_actor_rotation(unreal.Rotator(yaw=-90.0), False)
        return start
    start = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PlayerStart, location, unreal.Rotator(yaw=-90.0))
    wb._tag_actor(start, wb.GENERATED_TAG, wb.PLAYER_TAG, label="RWB_PlayerStart")
    return start


def build_world(analysis, values, scene_script):
    """Run the reconstruction with the current sliders and build everything.  Returns a report."""
    wb = _wb()
    meta = analysis.get("scene") or {}
    cache = meta.get("cache")
    if not cache:
        raise RuntimeError("this analysis has no scene data - analyze the image again")
    out = Path(cache) / "build.json"
    args = [
        "build", "--cache", cache, "--eye", "160",
        "--depth", "%.0f" % (float(values["world_size"]) * 2.0),
        "--size", "%.3f" % float(values.get("object_scale", 1.0)),
        "--relief", "%.3f" % (float(values["height"]) / 560.0),
        "--fov", "%.1f" % float(values.get("fov", 60.0)),
        "--grid", "72", "--density", "%.3f" % float(values.get("density", 0.55)),
        "--seed", str(int(values.get("seed", 1))), "--out", str(out),
    ]
    wb._run_helper(scene_script, args, timeout=600)
    scene = json.loads(out.read_text(encoding="utf-8"))
    scene["cache"] = cache
    unreal.log("Reference World Builder scene: " + scene.get("info", ""))

    name = rwb_layers._safe(analysis.get("name", "ref"), 24)
    folder = rwb_layers.LAYER_ROOT + "/" + name
    heights = scene["terrain"]["heights"]
    world_x, world_y = scene["world_x"], scene["world_y"]

    wb._lighting(analysis, world_x, world_y, heights)
    _terrain(scene, analysis, values, folder)

    backdrop = scene.get("backdrop") or {}
    layers = list(scene.get("layers") or [])
    if backdrop.get("texture"):
        layers.insert(0, {
            "id": "bk", "name": "Backdrop", "category": "sky", "kind": "sky",
            "texture": backdrop["texture"], "texture_box": [0, 0, 1, 1], "colors": {},
            "world": {"x": backdrop["x"], "y": backdrop["y"], "z": backdrop["z"],
                      "w": backdrop["width"], "h": backdrop["height"]},
        })
    pseudo = {
        "name": analysis.get("name", "ref"), "layers": layers, "layer_dir": cache,
        "width": analysis.get("width", 1), "height": analysis.get("height", 1), "terrain_start_row": 0,
    }
    result = rwb_layers.build_objects(pseudo, values, heights, world_x, world_y, None)
    _grade(analysis)
    _player_start(scene, heights, world_x, world_y)
    try:
        unreal.EditorAssetLibrary.save_directory(folder, only_if_is_dirty=True, recursive=True)
    except Exception:
        pass
    result["info"] = scene.get("info", "")
    return result
