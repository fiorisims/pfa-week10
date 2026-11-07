"""Reference World Builder - bring generated 3D objects into Unreal.

The meshes are made outside Unreal by ``reference_mesh.py`` (numpy) and refined by
``blender_refine.py`` (Blender).  This imports them as static meshes, each with its own
piece of the painting as its texture.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import unreal

import rwb_layers

GENERATED_ROOT = "/Game/ReferenceWorldBuilder/Generated"
MODEL_MATERIAL = "M_RWB_ModelPaint_v2"
MEL = unreal.MaterialEditingLibrary


def _wb():
    import world_builder
    return world_builder


def _build_model_paint(material):
    """Painting on UV0 mixed over a flat tint.  The meshes carry their own projection in UVs."""
    x = rwb_layers._x
    link = rwb_layers._link
    tex = x(material, unreal.MaterialExpressionTextureSampleParameter2D, -600, 120)
    tex.set_editor_property("parameter_name", "Art")
    default = (unreal.load_asset("/Engine/EngineResources/WhiteSquareTexture")
               or unreal.load_asset("/Engine/EngineResources/DefaultTexture"))
    if default:
        tex.set_editor_property("texture", default)
    tint = rwb_layers._mask(material, rwb_layers._vector_param(material, "Tint", (0.5, 0.5, 0.5, 1), -600, -260), -420, -260)
    blend = rwb_layers._scalar_param(material, "ArtBlend", 1.0, -600, 380)
    mix = x(material, unreal.MaterialExpressionLinearInterpolate, -200, 40)
    link(tint, "", mix, "A")
    link(tex, "RGB", mix, "B")
    link(blend, "", mix, "Alpha")
    MEL.connect_material_property(mix, "", unreal.MaterialProperty.MP_BASE_COLOR)
    # paint already has its light and shadow in it: a share of it glows, so the scene's own
    # lighting shapes the object without turning the painter's darks to black
    self_lit = rwb_layers._scalar_param(material, "SelfLit", 0.55, -600, 520)
    glow = x(material, unreal.MaterialExpressionMultiply, -200, 400)
    link(mix, "", glow, "A")
    link(self_lit, "", glow, "B")
    MEL.connect_material_property(glow, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    rough = x(material, unreal.MaterialExpressionConstant, -200, 260)
    rough.set_editor_property("r", 0.9)
    MEL.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)


GROUND_MATERIAL = "M_RWB_GroundPaint_v1"
WATER_MATERIAL = "M_RWB_WaterPaint_v1"
WATER_TILE_CM = 520.0        # one repeat of the painted swatch on the water


def _build_water_paint(material):
    """The painted water swatch, repeated across the surface by world position (no UVs needed)."""
    x = rwb_layers._x
    link = rwb_layers._link
    material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
    material.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    world = x(material, unreal.MaterialExpressionWorldPosition, -1000, 0)
    flat = rwb_layers._mask(material, world, -820, 0, "rg")
    size = rwb_layers._scalar_param(material, "TileCm", WATER_TILE_CM, -1000, 160)
    uv = x(material, unreal.MaterialExpressionDivide, -640, 40)
    link(flat, "", uv, "A")
    link(size, "", uv, "B")
    tex = x(material, unreal.MaterialExpressionTextureSampleParameter2D, -440, 0)
    tex.set_editor_property("parameter_name", "Art")
    default = (unreal.load_asset("/Engine/EngineResources/WhiteSquareTexture")
               or unreal.load_asset("/Engine/EngineResources/DefaultTexture"))
    if default:
        tex.set_editor_property("texture", default)
    link(uv, "", tex, "UVs")
    MEL.connect_material_property(tex, "RGB", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    opacity = x(material, unreal.MaterialExpressionConstant, -440, 220)
    opacity.set_editor_property("r", 0.92)
    MEL.connect_material_property(opacity, "", unreal.MaterialProperty.MP_OPACITY)


def _build_ground_paint(material):
    """The baked map of the terrain, lit by the scene, with a little of its own light like the models."""
    x = rwb_layers._x
    tex = x(material, unreal.MaterialExpressionTextureSampleParameter2D, -600, 0)
    tex.set_editor_property("parameter_name", "Art")
    default = (unreal.load_asset("/Engine/EngineResources/WhiteSquareTexture")
               or unreal.load_asset("/Engine/EngineResources/DefaultTexture"))
    if default:
        tex.set_editor_property("texture", default)
    MEL.connect_material_property(tex, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
    glow = x(material, unreal.MaterialExpressionMultiply, -300, 200)
    rwb_layers._link(tex, "RGB", glow, "A")
    rwb_layers._link(rwb_layers._scalar_param(material, "SelfLit", 0.35, -600, 300), "", glow, "B")
    MEL.connect_material_property(glow, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    rough = x(material, unreal.MaterialExpressionConstant, -300, 400)
    rough.set_editor_property("r", 0.95)
    MEL.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)


def paint_ground(ctx, component, image_path, cache_dir):
    """Lay the baked surface map over the terrain."""
    import rwb_scene
    folder = ctx["folder"] + "/Ground"
    texture = rwb_scene._import_texture(image_path, folder, "T_%s_ground" % ctx["psd"], str(cache_dir))
    if not texture:
        return False
    master = rwb_layers._material(GROUND_MATERIAL, _build_ground_paint)
    instance = rwb_layers._mic(folder, "MI_%s_ground" % ctx["psd"], master, textures={"Art": texture})
    component.set_material(0, instance)
    return True


def paint_water(ctx, actor, tile_path, cache_dir):
    """Lay the swatch stitched from the painted water over the water plane."""
    import rwb_scene
    folder = ctx["folder"] + "/Water"
    texture = rwb_scene._import_texture(tile_path, folder, "T_%s_water" % ctx["psd"], str(cache_dir))
    if not texture:
        return False
    for prop in ("address_x", "address_y"):
        texture.set_editor_property(prop, unreal.TextureAddress.TA_WRAP)
    unreal.EditorAssetLibrary.save_loaded_asset(texture, False)
    master = rwb_layers._material(WATER_MATERIAL, _build_water_paint)
    instance = rwb_layers._mic(folder, "MI_%s_water" % ctx["psd"], master, textures={"Art": texture})
    actor.get_component_by_class(unreal.StaticMeshComponent).set_material(0, instance)
    return True


def model_material():
    return rwb_layers._material(MODEL_MATERIAL, _build_model_paint)


def _import_own_look(path, folder, name):
    """A found model kept with its own materials: Unreal's glTF import brings them in."""
    sub = folder + "/" + name
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", str(path))
    task.set_editor_property("destination_path", sub)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    for asset_path in unreal.EditorAssetLibrary.list_assets(sub, recursive=True, include_folder=False):
        asset = unreal.load_asset(asset_path)
        if isinstance(asset, unreal.StaticMesh):
            return asset
    return None


def _import_mesh(path, folder, name):
    if str(path).lower().endswith(".glb"):
        return _import_own_look(path, folder, name)
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", str(path))
    task.set_editor_property("destination_path", folder)
    task.set_editor_property("destination_name", name)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    if str(path).lower().endswith(".fbx"):
        options = unreal.FbxImportUI()
        options.set_editor_property("import_mesh", True)
        options.set_editor_property("import_as_skeletal", False)
        options.set_editor_property("import_animations", False)
        options.set_editor_property("import_materials", False)
        options.set_editor_property("import_textures", False)
        options.set_editor_property("automated_import_should_detect_type", False)
        options.set_editor_property("mesh_type_to_import", unreal.FBXImportType.FBXIT_STATIC_MESH)
        data = options.static_mesh_import_data
        data.set_editor_property("combine_meshes", True)
        data.set_editor_property("auto_generate_collision", True)
        data.set_editor_property("remove_degenerates", True)
        # keep the file's axes: reference_mesh.py already accounts for the handedness flip
        data.set_editor_property("force_front_x_axis", False)
        data.set_editor_property("convert_scene", True)
        data.set_editor_property("transform_vertex_to_absolute", True)
        task.set_editor_property("options", options)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    asset = unreal.load_asset(folder + "/" + name)
    return asset if isinstance(asset, unreal.StaticMesh) else None


def build_models(objects, camera, picture_name, cache_dir, helper, script, use_blender=True,
                 strokes=None, stroke_strength=1.0, subdir="models"):
    """Run the external mesh builder, import the results.

    helper(script, args, timeout=...) runs a system-Python script (world_builder._run_helper).
    Returns ({object id: model dict}, [report lines]).
    """
    if not objects:
        return {}, []
    out_dir = Path(cache_dir) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    job = out_dir / "job.json"
    job.write_text(json.dumps({"camera": camera, "objects": objects, "strokes": strokes,
                               "stroke_strength": stroke_strength}), encoding="utf-8")
    args = ["--objects", str(job), "--out", str(out_dir)]
    if not use_blender:
        args.append("--no-blender")
    stdout = helper(script, args, timeout=1800,
                    message="Building %d 3D model%s in Blender and wrapping your brushstrokes"
                    % (len(objects), "" if len(objects) == 1 else "s"), span=(0.0, 45.0))
    summary = _wb()._last_json(stdout)
    manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
    report = [manifest.get("message", ""), manifest.get("bake_message", "")]
    report.extend("could not model %s" % line for line in manifest.get("skipped", []))

    safe = rwb_layers._safe(picture_name, 24)
    folder = GENERATED_ROOT + "/" + safe
    record_path = out_dir / "unreal_meshes.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception:
        record = {}
    models = {}
    import rwb_scene
    total = len(manifest.get("models", []))
    for number, entry in enumerate(manifest.get("models", [])):
        _wb()._progress_message("Bringing model %d of %d into Unreal: %s" % (number + 1, total, entry.get("label")),
                                45.0 + 45.0 * number / max(1, total))
        mesh_name = "SM_" + entry["name"]
        if entry.get("bake_error"):
            report.append("%s: brushstrokes not wrapped (%s)" % (entry.get("label"), entry["bake_error"]))
        path = Path(entry["mesh"])
        stamp = "%s|%d|%d" % (path.name, path.stat().st_size, int(path.stat().st_mtime))
        asset_path = folder + "/" + mesh_name
        mesh = None
        if record.get(mesh_name) == stamp and unreal.EditorAssetLibrary.does_asset_exist(asset_path):
            mesh = unreal.load_asset(asset_path)
        if not isinstance(mesh, unreal.StaticMesh):
            mesh = _import_mesh(path, folder, mesh_name)
            if mesh:
                record[mesh_name] = stamp
        if not mesh:
            report.append("could not import the model for %s" % entry.get("label"))
            continue
        art = None if entry.get("own_look") else rwb_scene._import_texture(entry["art"], folder + "/Art", "T_" + entry["name"], str(out_dir))
        thumb = rwb_scene._import_texture(entry["thumb"], folder + "/Art", "T_thumb_" + entry["name"], str(out_dir))
        bounds = mesh.get_bounds()
        models[entry["id"]] = {
            "mesh": mesh, "art": art, "thumb": thumb, "name": entry["name"], "method": entry["method"], "label": entry.get("label"),
            "height": max(0.01, bounds.box_extent.z * 2.0),
            "width": max(0.01, bounds.box_extent.x * 2.0),
            "depth": max(0.01, bounds.box_extent.y * 2.0),
            "origin_z": bounds.origin.z - bounds.box_extent.z,
            "origin_x": bounds.origin.x,
            "origin_y": bounds.origin.y,
            "refined": bool(entry.get("refined")),
            "native_cm_per_unit": entry["size"][2] and (bounds.box_extent.z * 2.0) / entry["size"][2],
            "own_look": bool(entry.get("own_look")), "source_name": entry.get("source_name"),
        }
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return models, report


def dress(ctx, component, model, colors, key):
    """Give a generated mesh its own piece of the painting (a found model may keep its own look)."""
    if model.get("own_look"):
        return
    wb = _wb()
    tint = wb._linear((colors or {}).get("dominant") or "#808080")
    folder = ctx["folder"] + "/Materials"
    instance = rwb_layers._mic(folder, "MI_gen_%s" % re.sub(r"[^A-Za-z0-9_]", "_", key), model_material(),
                               vectors={"Tint": tint},
                               scalars={"ArtBlend": ctx.get("art_blend", 1.0)},
                               textures={"Art": model["art"]} if model.get("art") else None)
    for slot in range(max(1, component.get_num_materials())):
        component.set_material(slot, instance)
