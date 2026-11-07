"""Reference World Builder - build Unreal objects from named Photoshop layers.

``reference_layers.py`` (system Python) tells us, for every PSD layer, what the
layer's NAME says it is (castle, oak, thorns, Aurora...), where its pixels sit
in the painting, how big they are, which colours they use, and a cropped PNG
of the artwork.  This module turns that into the level:

* the NAME chooses the 3D form - a Kenney tree / rock / bush model, a castle
  or tower assembled from basic shapes, or a painted standee for characters;
* the LAYER decides position (where it sits in the picture), size and
  proportions (its silhouette), colours (each material slot is tinted from
  the matching part of the layer: canopy, trunk, accent...), and
* the layer's own ARTWORK is projected onto the front of the object, blended
  over those colours by the "Art blend" slider.

Terrain layers (ground, hills, water, paths) shape and colour the landscape,
a sky layer becomes a painted backdrop, fog/light layers tint the atmosphere.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from pathlib import Path

import unreal

MAT_DIR = "/Game/ReferenceWorldBuilder/Materials"
LAYER_ROOT = "/Game/ReferenceWorldBuilder/Layers"
KENNEY = "/Game/WorldBuilder/Imported/Kenney"
PAINT_MATERIAL = "M_RWB_LayerPaint_v1"
CARD_MATERIAL = "M_RWB_LayerCard_v1"
TINT_MATERIAL = "M_RWB_Tint_v1"

SHAPES = {
    "cube": "/Engine/BasicShapes/Cube",
    "cylinder": "/Engine/BasicShapes/Cylinder",
    "cone": "/Engine/BasicShapes/Cone",
    "sphere": "/Engine/BasicShapes/Sphere",
}

NATURE_MESHES = {
    "tree": ("tree_oak", "tree_detailed", "tree_pineRoundA"),
    "pine": ("tree_pineTallB", "tree_pineRoundA"),
    "bush": ("plant_bushDetailed", "plant_bushSmall"),
    "thorns": ("plant_bushDetailed", "plant_bushSmall"),
    "flower": ("flower_purpleA", "flower_redB", "flower_yellowC"),
    "grass": ("grass", "grass_large", "plant_bushSmall"),
    "rock": ("rock_largeB", "rock_largeE", "rock_smallA", "rock_smallH"),
    "cliff": ("cliff_large_rock", "cliff_top_rock", "cliff_blockSlope_rock"),
}
STRUCTURES = ("castle", "tower", "house", "wall", "fence", "bridge", "gate", "pillar", "lantern", "furniture", "log")
MEL = unreal.MaterialEditingLibrary

# Typical real heights (Unreal units = cm).  The painted size is blended towards
# these so a foreground princess is not ten metres tall.  weight = how much the
# real-world size wins over the painted size (0 = trust the painting).
SIZE_PRIORS = {
    "character": (180.0, 0.85), "creature": (120.0, 0.6), "prop_card": (150.0, 0.5),
    "tree": (1100.0, 0.35), "pine": (1300.0, 0.35), "bush": (160.0, 0.4), "thorns": (420.0, 0.3),
    "rock": (180.0, 0.35), "cliff": (900.0, 0.3), "log": (70.0, 0.4),
    "castle": (3200.0, 0.35), "tower": (1600.0, 0.35), "house": (750.0, 0.4), "wall": (350.0, 0.35),
    "fence": (120.0, 0.5), "bridge": (300.0, 0.3), "gate": (450.0, 0.4), "pillar": (450.0, 0.35),
    "lantern": (260.0, 0.6), "furniture": (110.0, 0.6),
    "building": (1500.0, 0.3), "sign": (150.0, 0.4), "vehicle": (150.0, 0.6),
}

# Distance tags from layer names.  0 = far edge of the terrain, 1 = the player's
# edge; negative values are beyond the far edge.  Objects keep their painted
# front-to-back order inside their band.
DEPTH_BANDS = {"fg": (0.66, 0.94), "mg": (0.33, 0.66), "bg": (-0.15, 0.33)}

_material_cache = {}


def _natural_v(analysis, image_y):
    """Image row -> terrain depth (0 far edge, 1 near edge) as painted."""
    height = float(analysis["height"])
    start = analysis.get("terrain_start_row", 0) / height
    return (image_y / height - start) / max(0.05, 1.0 - start)


def _layer_vrange(analysis, layer):
    """The painted depth span the layer's band is stretched over (None = put it mid-band)."""
    insts = layer.get("instances") or []
    if insts:
        bottoms = [_natural_v(analysis, inst["bbox"][3]) for inst in insts]
        low, high = min(bottoms), max(bottoms)
    elif layer.get("bbox"):
        low, high = _natural_v(analysis, layer["bbox"][1]), _natural_v(analysis, layer["bbox"][3])
    else:
        return None
    return (low, high) if high - low > 0.03 else None


def depth_v(analysis, layer, v):
    """Move a painted depth into the layer's FG / MG / BG band."""
    band = DEPTH_BANDS.get(layer.get("depth"))
    if not band:
        return v
    low, high = band
    span = _layer_vrange(analysis, layer)
    t = 0.5 if not span else (v - span[0]) / (span[1] - span[0])
    return low + max(0.0, min(1.0, t)) * (high - low)


def _undepth_v(analysis, layer, v):
    """Inverse of depth_v for terrain sampling: terrain depth -> painted depth (None = outside band)."""
    band = DEPTH_BANDS.get(layer.get("depth"))
    if not band:
        return v
    low, high = band
    t = (v - low) / (high - low)
    if t < 0.0 or t > 1.0:
        return None
    span = _layer_vrange(analysis, layer) or (0.0, 1.0)
    return span[0] + t * (span[1] - span[0])


def _wb():
    import world_builder
    return world_builder


def _safe(text, size=20):
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")
    return (value or "x")[:size]


# --------------------------------------------------------------------------- #
# materials
# --------------------------------------------------------------------------- #
def _x(material, cls, x, y):
    node = MEL.create_material_expression(material, cls, x, y)
    if not node:
        raise RuntimeError("could not create " + cls.__name__)
    return node


def _link(a, out, b, inp):
    if not MEL.connect_material_expressions(a, out, b, inp):
        raise RuntimeError("material link failed: %s.%s -> %s.%s" % (
            a.get_class().get_name(), out, b.get_class().get_name(), inp))


def _mask(material, source, x, y, channels="rgb", out=""):
    node = _x(material, unreal.MaterialExpressionComponentMask, x, y)
    for channel in "rgba":
        node.set_editor_property(channel, channel in channels)
    _link(source, out, node, "")
    return node


def _vector_param(material, name, value, x, y):
    node = _x(material, unreal.MaterialExpressionVectorParameter, x, y)
    node.set_editor_property("parameter_name", name)
    node.set_editor_property("default_value", unreal.LinearColor(*value))
    return node


def _scalar_param(material, name, value, x, y):
    node = _x(material, unreal.MaterialExpressionScalarParameter, x, y)
    node.set_editor_property("parameter_name", name)
    node.set_editor_property("default_value", float(value))
    return node


def _art_sample(material):
    """Front (X/Z) projection of the 'Art' texture across the object's bounds.

    UV = (worldPos - objectPivot - ProjMin) / ProjSize, using X for U and Z
    (flipped) for V, so the picture lands on the object the way it was painted.
    """
    world = _x(material, unreal.MaterialExpressionWorldPosition, -1800, 0)
    pivot = _x(material, unreal.MaterialExpressionObjectPositionWS, -1800, 140)
    rel = _x(material, unreal.MaterialExpressionSubtract, -1600, 60)
    _link(world, "", rel, "A")
    _link(pivot, "", rel, "B")
    pmin = _mask(material, _vector_param(material, "ProjMin", (0, 0, 0, 0), -1800, 280), -1600, 280)
    psize = _mask(material, _vector_param(material, "ProjSize", (100, 100, 100, 0), -1800, 420), -1600, 420)
    local = _x(material, unreal.MaterialExpressionSubtract, -1400, 120)
    _link(rel, "", local, "A")
    _link(pmin, "", local, "B")
    norm = _x(material, unreal.MaterialExpressionDivide, -1250, 160)
    _link(local, "", norm, "A")
    _link(psize, "", norm, "B")
    u = _mask(material, norm, -1100, 100, "r")
    z = _mask(material, norm, -1100, 220, "b")
    v = _x(material, unreal.MaterialExpressionOneMinus, -950, 220)
    _link(z, "", v, "")
    uv = _x(material, unreal.MaterialExpressionAppendVector, -800, 160)
    _link(u, "", uv, "A")
    _link(v, "", uv, "B")
    tex = _x(material, unreal.MaterialExpressionTextureSampleParameter2D, -600, 120)
    tex.set_editor_property("parameter_name", "Art")
    default = (unreal.load_asset("/Engine/EngineResources/WhiteSquareTexture")
               or unreal.load_asset("/Engine/EngineResources/DefaultTexture"))
    if default:
        tex.set_editor_property("texture", default)
    _link(uv, "", tex, "UVs")
    return tex


def _build_paint(material):
    tex = _art_sample(material)
    tint = _mask(material, _vector_param(material, "Tint", (0.5, 0.5, 0.5, 1), -600, -260), -420, -260)
    blend = _scalar_param(material, "ArtBlend", 0.8, -600, 380)
    amount = _x(material, unreal.MaterialExpressionMultiply, -380, 300)
    _link(tex, "A", amount, "A")
    _link(blend, "", amount, "B")
    mix = _x(material, unreal.MaterialExpressionLinearInterpolate, -200, 40)
    _link(tint, "", mix, "A")
    _link(tex, "RGB", mix, "B")
    _link(amount, "", mix, "Alpha")
    MEL.connect_material_property(mix, "", unreal.MaterialProperty.MP_BASE_COLOR)
    rough = _x(material, unreal.MaterialExpressionConstant, -200, 260)
    rough.set_editor_property("r", 0.85)
    MEL.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)


def _build_card(material):
    material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
    material.set_editor_property("two_sided", True)
    tex = _art_sample(material)
    MEL.connect_material_property(tex, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
    MEL.connect_material_property(tex, "A", unreal.MaterialProperty.MP_OPACITY_MASK)
    glow = _scalar_param(material, "Glow", 0.0, -600, 380)
    emit = _x(material, unreal.MaterialExpressionMultiply, -300, 300)
    _link(tex, "RGB", emit, "A")
    _link(glow, "", emit, "B")
    MEL.connect_material_property(emit, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    rough = _x(material, unreal.MaterialExpressionConstant, -200, 460)
    rough.set_editor_property("r", 0.9)
    MEL.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)


def _build_tint(material):
    tint = _mask(material, _vector_param(material, "Tint", (0.5, 0.5, 0.5, 1), -600, 0), -400, 0)
    MEL.connect_material_property(tint, "", unreal.MaterialProperty.MP_BASE_COLOR)
    rough = _x(material, unreal.MaterialExpressionConstant, -400, 160)
    rough.set_editor_property("r", 0.85)
    MEL.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)


def _material(name, builder):
    if _material_cache.get(name):
        return _material_cache[name]
    path = MAT_DIR + "/" + name
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        material = unreal.load_asset(path)
    else:
        tools = unreal.AssetToolsHelpers.get_asset_tools()
        material = tools.create_asset(name, MAT_DIR, unreal.Material, unreal.MaterialFactoryNew())
        if not material:
            raise RuntimeError("could not create " + name)
        try:
            builder(material)
            MEL.recompile_material(material)
            unreal.EditorAssetLibrary.save_loaded_asset(material, False)
        except Exception:
            try:
                unreal.EditorAssetLibrary.delete_loaded_asset(material)
            except Exception:
                pass
            raise
    _material_cache[name] = material
    return material


def _masters(paint_wanted):
    out = {"tint": _material(TINT_MATERIAL, _build_tint), "paint": None, "card": None}
    for key, name, builder in (("paint", PAINT_MATERIAL, _build_paint), ("card", CARD_MATERIAL, _build_card)):
        if key == "paint" and not paint_wanted:
            continue
        try:
            out[key] = _material(name, builder)
        except Exception as error:
            unreal.log_warning("Reference World Builder: %s material unavailable (%s); using flat colours." % (name, error))
    return out


def _mic(folder, name, parent, vectors=None, scalars=None, textures=None):
    path = folder + "/" + name
    instance = unreal.load_asset(path) if unreal.EditorAssetLibrary.does_asset_exist(path) else None
    if not isinstance(instance, unreal.MaterialInstanceConstant):
        tools = unreal.AssetToolsHelpers.get_asset_tools()
        instance = tools.create_asset(
            name, folder, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    MEL.set_material_instance_parent(instance, parent)
    for key, value in (vectors or {}).items():
        MEL.set_material_instance_vector_parameter_value(instance, key, value)
    for key, value in (scalars or {}).items():
        MEL.set_material_instance_scalar_parameter_value(instance, key, float(value))
    for key, value in (textures or {}).items():
        MEL.set_material_instance_texture_parameter_value(instance, key, value)
    MEL.update_material_instance(instance)
    return instance


# --------------------------------------------------------------------------- #
# artwork textures
# --------------------------------------------------------------------------- #
def _texture_name(ctx, layer, index):
    return "T_%s_%s_%s" % (ctx["psd"], layer["id"], index)


def import_textures(ctx):
    analysis = ctx["analysis"]
    folder = ctx["folder"] + "/Art"
    files = {}
    for layer in analysis.get("layers") or []:
        if layer.get("texture"):
            files[_texture_name(ctx, layer, "sky")] = layer["texture"]
        for index, inst in enumerate(layer.get("instances") or []):
            if inst.get("texture"):
                files[_texture_name(ctx, layer, index)] = inst["texture"]
    record_path = Path(analysis.get("layer_dir") or ".") / "unreal_imports.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception:
        record = {}
    tasks, fresh = [], []
    for name, path in files.items():
        if not os.path.isfile(path):
            continue
        stamp = "%s|%d|%d" % (folder, os.path.getsize(path), int(os.path.getmtime(path)))
        if record.get(name) == stamp and unreal.EditorAssetLibrary.does_asset_exist(folder + "/" + name):
            continue
        task = unreal.AssetImportTask()
        task.set_editor_property("filename", path)
        task.set_editor_property("destination_path", folder)
        task.set_editor_property("destination_name", name)
        task.set_editor_property("replace_existing", True)
        task.set_editor_property("automated", True)
        task.set_editor_property("save", False)
        tasks.append(task)
        fresh.append(name)
        record[name] = stamp
    if tasks:
        unreal.log("Reference World Builder: importing %d layer artwork textures ..." % len(tasks))
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(tasks)
    textures = {}
    for name in files:
        texture = unreal.load_asset(folder + "/" + name)
        if isinstance(texture, unreal.Texture2D):
            textures[name] = texture
            if name in fresh:
                for prop in ("address_x", "address_y"):
                    try:
                        texture.set_editor_property(prop, unreal.TextureAddress.TA_CLAMP)
                    except Exception:
                        pass
    try:
        record_path.write_text(json.dumps(record), encoding="utf-8")
    except Exception:
        pass
    return textures


# --------------------------------------------------------------------------- #
# terrain from layers
# --------------------------------------------------------------------------- #
def _coverage_at(analysis, layer, u, v_terrain):
    grid = layer.get("coverage")
    if not grid:
        return 0.0
    size = len(grid)
    v_terrain = _undepth_v(analysis, layer, v_terrain)
    if v_terrain is None:
        return 0.0
    v = _wb()._terrain_v(analysis, max(-0.5, min(1.0, v_terrain)))
    fx = max(0.0, min(size - 1.0, u * size - 0.5))
    fy = max(0.0, min(size - 1.0, v * size - 0.5))
    x0, y0 = int(fx), int(fy)
    x1, y1 = min(size - 1, x0 + 1), min(size - 1, y0 + 1)
    tx, ty = fx - x0, fy - y0
    top = grid[y0][x0] * (1 - tx) + grid[y0][x1] * tx
    bottom = grid[y1][x0] * (1 - tx) + grid[y1][x1] * tx
    return top * (1 - ty) + bottom * ty


def terrain_inputs(analysis):
    """Ground colours/brightness from ground-type layers (falls back to the composite)."""
    layers = [l for l in analysis.get("layers") or [] if l.get("kind") in ("ground", "hill", "path", "water")
              and l.get("coverage")]
    hills = [l for l in layers if l["kind"] == "hill"]
    waters = [l for l in layers if l["kind"] == "water"]
    if not layers:
        return analysis, hills, waters
    grid = dict(analysis["grid"])
    size = int(grid["size"])
    colors = [[list(c) for c in row] for row in grid["colors"]]
    lum = [list(row) for row in grid["luminance"]]
    for y in range(size):
        for x in range(size):
            weight, acc = 0.0, [0.0, 0.0, 0.0]
            for layer in layers:
                cov = layer["coverage"][y][x] if y < len(layer["coverage"]) else 0.0
                if cov <= 0.02:
                    continue
                cell = layer["cell_colors"][y][x]
                for i in range(3):
                    acc[i] += cell[i] * cov
                weight += cov
            if weight > 0.04:
                mix = [acc[i] / weight for i in range(3)]
                t = min(1.0, weight)
                colors[y][x] = [int(colors[y][x][i] * (1 - t) + mix[i] * t) for i in range(3)]
                lum[y][x] = (0.2126 * colors[y][x][0] + 0.7152 * colors[y][x][1] + 0.0722 * colors[y][x][2]) / 255.0
    grid["colors"] = colors
    grid["luminance"] = lum
    data = dict(analysis)
    data["grid"] = grid
    return data, hills, waters


def layer_heights(analysis, ground_data, vertices, height_range, hills, waters):
    wb = _wb()
    shaped = bool(hills) or any(l.get("kind") == "ground" for l in analysis.get("layers") or [])
    heights, _norm = wb._heightfield(ground_data, vertices, height_range * (0.3 if shaped else 1.0))
    water_z = (10.0 + height_range * 0.08) if waters else None
    lift = height_range * 0.12 if waters else 0.0
    for iy in range(vertices):
        v = iy / float(vertices - 1)
        for ix in range(vertices):
            u = ix / float(vertices - 1)
            h = heights[iy][ix] + lift
            if hills:
                hill = max(_coverage_at(analysis, layer, u, v) for layer in hills)
                hill = hill * hill * (3 - 2 * hill)
                h += height_range * 0.95 * hill
            if waters:
                wet = max(_coverage_at(analysis, layer, u, v) for layer in waters)
                if wet > 0.02:
                    h += (water_z - 45.0 - h) * min(1.0, wet * 1.6)
            heights[iy][ix] = h
    wb._smooth(heights, 1)
    low = min(min(row) for row in heights)
    high = max(max(row) for row in heights)
    span = max(0.001, high - low)
    norm = [[(value - low) / span for value in row] for row in heights]
    return heights, norm, water_z


def atmosphere_overrides(analysis):
    out = {}
    for layer in analysis.get("layers") or []:
        colors = layer.get("colors") or {}
        kind = layer.get("kind")
        if kind == "sky" and "sky" not in out:
            out["sky"] = colors.get("top") or colors.get("avg")
            out["sun"] = colors.get("accent")
        elif kind == "fog" and "fog" not in out:
            out["fog"] = colors.get("avg")
        elif kind == "light" and "accent" not in out:
            out["accent"] = colors.get("accent")
        elif kind == "water" and "water" not in out:
            out["water"] = colors.get("avg")
    return {k: v for k, v in out.items() if v}


# --------------------------------------------------------------------------- #
# placement + spawning
# --------------------------------------------------------------------------- #
def _place(ctx, bbox, layer=None, world=None):
    """Painting rectangle -> (ground point, width, height, depth v)."""
    wb = _wb()
    if world:                                    # already solved in 3D (single-image depth mode)
        x, y = float(world["x"]), float(world["y"])
        u = max(0.0, min(1.0, x / ctx["world_x"] + 0.5))
        v = max(0.0, min(1.0, y / ctx["world_y"] + 0.5))
        z = wb._sample_height(ctx["heights"], u, v)
        return (unreal.Vector(x, y, z), max(20.0, min(9000.0, float(world["w"]))),
                max(20.0, min(9000.0, float(world["h"]))), v)
    analysis = ctx["analysis"]
    width_px, height_px = float(analysis["width"]), float(analysis["height"])
    start = analysis.get("terrain_start_row", 0) / height_px
    x0, y0, x1, y1 = bbox
    u = max(0.0, min(1.0, (x0 + x1) * 0.5 / width_px))
    v = ((y1 / height_px) - start) / max(0.05, 1.0 - start)
    if layer is not None:
        v = depth_v(analysis, layer, v)            # FG / MG / BG tag in the layer name
    category = layer.get("category") if layer else None
    world_x, world_y = ctx["world_x"], ctx["world_y"]
    x = (u - 0.5) * world_x
    if v >= 0.0:
        v = min(1.0, v)
        y = (v - 0.5) * world_y
        z = wb._sample_height(ctx["heights"], u, v)
    else:                                     # standing above the horizon line: beyond the far edge
        y = -0.5 * world_y + v * world_y * 1.5
        z = wb._sample_height(ctx["heights"], u, 0.0)
    depth = 1.0 + (1.0 - max(-1.0, v)) * 0.9  # far things were painted smaller
    scale = world_x / width_px * ctx["size"] * depth
    width = max(20.0, (x1 - x0) * scale)
    height = max(20.0, (y1 - y0) * scale)
    prior = SIZE_PRIORS.get(category)
    if prior:
        typical, weight = prior
        typical *= ctx["size"]
        target = math.exp(weight * math.log(typical) + (1.0 - weight) * math.log(height))
        factor = target / height
        width, height = width * factor, target
    return unreal.Vector(x, y, z), min(9000.0, width), min(9000.0, height), v


def _register(label, scale=None):
    wb = _wb()
    wb._state["base_scales"][label] = tuple(scale) if scale else None
    wb._state["prop_labels"].add(label)


def _spawn(ctx, mesh, location, scale, label, yaw=0.0, roll=0.0, pitch=0.0, collision=True, shadow=True):
    wb = _wb()
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.StaticMeshActor, location, unreal.Rotator(roll=roll, pitch=pitch, yaw=yaw))
    if not actor:
        raise RuntimeError("could not spawn " + label)
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    component.set_static_mesh(mesh)
    component.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    actor.set_actor_scale3d(unreal.Vector(*scale))
    if not collision:
        component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    if not shadow:
        try:
            component.set_cast_shadow(False)
        except Exception:
            pass
    wb._tag_actor(actor, wb.GENERATED_TAG, wb.PROP_TAG, label=label)
    ctx["count"] += 1
    return actor, component


def _label(ctx, layer, suffix):
    return "RWB_%s_%s_%s" % (layer["id"], _safe(layer["name"], 18), suffix)


def _slot_role(mesh, slot):
    try:
        name = mesh.get_material(slot).get_name().lower()
    except Exception:
        return "dominant"
    if any(k in name for k in ("wood", "bark", "trunk", "brown")):
        return "bottom"
    if any(k in name for k in ("leaf", "leafs", "grass", "green", "foliage", "needle")):
        return "top"
    if name.startswith("color") or "petal" in name or "flower" in name:
        return "accent"
    if "dirt" in name:
        return "dark"
    return "dominant"


def _bounds(actors):
    lo = [1e18, 1e18, 1e18]
    hi = [-1e18, -1e18, -1e18]
    for actor in actors:
        origin, extent = actor.get_actor_bounds(False)
        for i, (o, e) in enumerate(((origin.x, extent.x), (origin.y, extent.y), (origin.z, extent.z))):
            lo[i] = min(lo[i], o - e)
            hi[i] = max(hi[i], o + e)
    return lo, [max(1.0, hi[i] - lo[i]) for i in range(3)]


def _dress(ctx, parts, texture, colors, key, card=False, glow=0.0):
    """Give every part a material: tinted from the layer, with its artwork projected on top.

    parts: [(actor, component, mesh, role or None)]
    """
    wb = _wb()
    folder = ctx["folder"] + "/Materials"
    lo, size = _bounds([p[0] for p in parts])
    masters = ctx["masters"]
    for index, (actor, component, mesh, role) in enumerate(parts):
        loc = actor.get_actor_location()
        proj_min = unreal.LinearColor(lo[0] - loc.x, lo[1] - loc.y, lo[2] - loc.z, 0.0)
        proj_size = unreal.LinearColor(size[0], size[1], size[2], 0.0)
        for slot in range(max(1, component.get_num_materials())):
            part_role = role or _slot_role(mesh, slot)
            tint_hex = colors.get(part_role) or colors.get("dominant") or "#808080"
            tint = wb._linear(tint_hex)
            if card and masters["card"] and texture:
                instance = _mic(folder, "MI_%s_%d_%d" % (key, index, slot), masters["card"],
                                vectors={"ProjMin": proj_min, "ProjSize": proj_size},
                                scalars={"Glow": glow}, textures={"Art": texture})
            elif ctx["paint"] and masters["paint"] and texture:
                instance = _mic(folder, "MI_%s_%d_%d" % (key, index, slot), masters["paint"],
                                vectors={"Tint": tint, "ProjMin": proj_min, "ProjSize": proj_size},
                                scalars={"ArtBlend": ctx["blend"]}, textures={"Art": texture})
            else:
                instance = _mic(folder, "MI_%s_c%s" % (ctx["psd"], tint_hex.lstrip("#")), masters["tint"],
                                vectors={"Tint": tint})
            component.set_material(slot, instance)


def _mesh_aspect(mesh):
    bounds = mesh.get_bounds()
    return bounds.box_extent.z / max(1.0, bounds.box_extent.x)


# --------------------------------------------------------------------------- #
# recipes
# --------------------------------------------------------------------------- #
def _nature(ctx, layer, inst, index, texture):
    meshes = ctx["meshes"].get(layer["category"]) or []
    if not meshes:
        return _card(ctx, layer, inst, index, texture)
    loc, width, height, _v = _place(ctx, inst["bbox"], layer, inst.get("world"))
    aspect = height / max(1.0, width)
    mesh = min(meshes, key=lambda m: abs(math.log(max(0.05, _mesh_aspect(m)) / max(0.05, aspect))))
    bounds = mesh.get_bounds()
    ex, ez = max(1.0, bounds.box_extent.x), max(1.0, bounds.box_extent.z)
    sx, sz = width / (2 * ex), height / (2 * ez)
    ratio = max(0.65, min(1.6, sx / sz))                 # keep the model recognisable
    mean = math.sqrt(sx * sz)
    sx, sz = mean * math.sqrt(ratio), mean / math.sqrt(ratio)
    sx, sz = max(0.05, min(80.0, sx)), max(0.05, min(80.0, sz))
    z = loc.z - (bounds.origin.z - bounds.box_extent.z) * sz - height * 0.03
    yaw = 0.0 if (ctx["paint"] and texture) else ctx["rng"].uniform(0, 360)
    label = _label(ctx, layer, "%02d" % index)
    actor, component = _spawn(ctx, mesh, unreal.Vector(loc.x, loc.y, z), (sx, sx, sz), label, yaw=yaw)
    _register(label, (sx, sx, sz))
    _dress(ctx, [(actor, component, mesh, None)], texture, inst["colors"], "%s_%s_%02d" % (ctx["psd"], layer["id"], index))
    ctx["occupied"].append((loc.x, loc.y, width * 0.5))


def _structure_parts(category, w, h):
    """Basic-shape kit for buildings.  (shape, centre x,y,z, size x,y,z, rotation, colour role)."""
    d = w * 0.55
    parts = []
    if category == "castle":
        keep_h = h * 0.55
        parts.append(("cube", (0, -d * 0.1, keep_h / 2), (w * 0.5, d * 0.5, keep_h), {}, "dominant"))
        tower_d = max(40.0, w * 0.15)
        tower_h = h * 0.72
        corners = [(-w * 0.42, -d * 0.35), (w * 0.42, -d * 0.35), (-w * 0.42, d * 0.3), (w * 0.42, d * 0.3)]
        for cx, cy in corners:
            parts.append(("cylinder", (cx, cy, tower_h / 2), (tower_d, tower_d, tower_h), {}, "dominant"))
            parts.append(("cone", (cx, cy, tower_h + h * 0.14), (tower_d * 1.35, tower_d * 1.35, h * 0.28), {}, "top"))
        main_d = max(50.0, w * 0.2)
        main_h = h * 0.82
        parts.append(("cylinder", (0, -d * 0.15, main_h / 2), (main_d, main_d, main_h), {}, "dominant"))
        parts.append(("cone", (0, -d * 0.15, main_h + h * 0.09), (main_d * 1.3, main_d * 1.3, h * 0.18), {}, "top"))
        parts.append(("cube", (0, d * 0.32, h * 0.17), (w * 0.84, max(20.0, w * 0.05), h * 0.34), {}, "bottom"))
    elif category == "tower":
        body = h * 0.72
        parts.append(("cylinder", (0, 0, body / 2), (w * 0.8, w * 0.8, body), {}, "dominant"))
        parts.append(("cone", (0, 0, body + (h - body) / 2), (w, w, h - body), {}, "top"))
    elif category == "house":
        body = h * 0.6
        roof = h - body
        depth = min(w * 0.75, roof * 2 * 0.95)
        side = roof * math.sqrt(2)
        parts.append(("cube", (0, 0, body / 2), (w, depth, body), {}, "dominant"))
        parts.append(("cube", (0, 0, body), (w * 1.06, side, side), {"roll": 45.0}, "top"))
    elif category == "wall":
        parts.append(("cube", (0, 0, h / 2), (w, max(25.0, h * 0.18), h), {}, "dominant"))
    elif category == "fence":
        parts.append(("cube", (0, 0, h / 2), (w, 10.0, h), {}, "dominant"))
    elif category == "bridge":
        deck = max(20.0, h * 0.25)
        parts.append(("cube", (0, 0, deck / 2), (w, max(150.0, w * 0.25), deck), {}, "dominant"))
        parts.append(("cube", (0, max(150.0, w * 0.25) / 2, deck + h * 0.2), (w, 10.0, h * 0.4), {}, "dark"))
    elif category == "gate":
        post = max(20.0, w * 0.16)
        for side in (-1, 1):
            parts.append(("cylinder", (side * (w / 2 - post / 2), 0, h / 2), (post, post, h), {}, "dominant"))
        parts.append(("cube", (0, 0, h - h * 0.08), (w, post, h * 0.16), {}, "top"))
    elif category == "pillar":
        parts.append(("cylinder", (0, 0, h / 2), (w, w, h), {}, "dominant"))
    elif category == "lantern":
        post_h = h * 0.75
        head = max(15.0, min(w * 0.6, h * 0.3))
        parts.append(("cylinder", (0, 0, post_h / 2), (max(6.0, w * 0.18), max(6.0, w * 0.18), post_h), {}, "dark"))
        parts.append(("sphere", (0, 0, post_h + head / 2), (head, head, head), {}, "accent"))
    elif category == "log":
        parts.append(("cylinder", (0, 0, h / 2), (h, h, w), {"pitch": 90.0}, "bottom"))
    else:                                                    # furniture and anything boxy
        parts.append(("cube", (0, 0, h / 2), (w, w * 0.6, h), {}, "dominant"))
    return parts


def _structure(ctx, layer, inst, index, texture):
    loc, width, height, _v = _place(ctx, inst["bbox"], layer, inst.get("world"))
    category = layer["category"]
    base_z = loc.z - height * 0.02
    if category == "bridge" and ctx["water_z"] is not None:
        base_z = max(base_z, ctx["water_z"] + 15.0)
    spawned = []
    for n, (shape, centre, size, rot, role) in enumerate(_structure_parts(category, width, height)):
        mesh = ctx["shapes"].get(shape)
        if not mesh:
            continue
        label = _label(ctx, layer, "%02d_%d" % (index, n))
        location = unreal.Vector(loc.x + centre[0], loc.y + centre[1], base_z + centre[2])
        if not rot:                                          # respect each shape's own pivot
            origin = mesh.get_bounds().origin
            location = unreal.Vector(location.x - origin.x * size[0] / 100.0,
                                     location.y - origin.y * size[1] / 100.0,
                                     location.z - origin.z * size[2] / 100.0)
        actor, component = _spawn(ctx, mesh, location, (size[0] / 100.0, size[1] / 100.0, size[2] / 100.0), label,
                                  roll=rot.get("roll", 0.0), pitch=rot.get("pitch", 0.0))
        _register(label, None)                               # buildings glow but do not pulse
        spawned.append((actor, component, mesh, role))
    if spawned:
        _dress(ctx, spawned, texture, inst["colors"], "%s_%s_%02d" % (ctx["psd"], layer["id"], index))
    if category == "lantern" and spawned:
        head = spawned[-1][0].get_actor_location()
        light = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PointLight, head, unreal.Rotator())
        if light:
            comp = light.get_component_by_class(unreal.PointLightComponent)
            if comp:
                comp.set_editor_property("intensity", 4000.0)
                comp.set_editor_property("attenuation_radius", max(600.0, height * 4))
                comp.set_light_color(_wb()._linear(inst["colors"].get("accent", "#FFD27A")))
            _wb()._tag_actor(light, _wb().GENERATED_TAG, _wb().LIGHT_TAG, label=_label(ctx, layer, "light%02d" % index))
    ctx["occupied"].append((loc.x, loc.y, max(width, height) * 0.6))


def _card(ctx, layer, inst, index, texture, glow=0.0):
    loc, width, height, _v = _place(ctx, inst["bbox"], layer, inst.get("world"))
    cube = ctx["shapes"]["cube"]
    label = _label(ctx, layer, "card%02d" % index)
    location = unreal.Vector(loc.x, loc.y, loc.z + height * 0.5 - height * 0.02)
    actor, component = _spawn(ctx, cube, location, (width / 100.0, 0.04, height / 100.0), label, collision=False)
    _register(label, (width / 100.0, 0.04, height / 100.0))
    _dress(ctx, [(actor, component, cube, "dominant")], texture, inst["colors"],
           "%s_%s_%02d" % (ctx["psd"], layer["id"], index), card=True, glow=glow)
    ctx["occupied"].append((loc.x, loc.y, width * 0.5))


def _sky(ctx, layer, texture):
    if not texture:
        return
    world = layer.get("world")
    if world:                                    # lined up with the painting's camera
        width, height = float(world["w"]), float(world["h"])
        location = unreal.Vector(float(world["x"]), float(world["y"]), float(world["z"]))
    else:
        box = layer.get("texture_box") or layer.get("bbox")
        aspect = (box[3] - box[1]) / float(max(1, box[2] - box[0]))
        width = ctx["world_x"] * 3.2
        height = max(ctx["world_y"] * 0.6, width * aspect)
        low = min(min(row) for row in ctx["heights"])
        location = unreal.Vector(0.0, -0.5 * ctx["world_y"] - ctx["world_y"] * 1.6, low - height * 0.12 + height / 2)
    label = _label(ctx, layer, "backdrop")
    actor, component = _spawn(ctx, ctx["shapes"]["cube"], location, (width / 100.0, 0.1, height / 100.0), label,
                              collision=False, shadow=False)
    _register(label, None)
    _dress(ctx, [(actor, component, ctx["shapes"]["cube"], "dominant")], texture, layer.get("colors") or {},
           "%s_%s_sky" % (ctx["psd"], layer["id"]), card=True, glow=1.0)


def _nearest(palette, rgb):
    best, best_d = 0, 1e18
    for i, value in enumerate(palette):
        r, g, b = _wb()._hex_rgb(value)
        d = (r - rgb[0]) ** 2 + (g - rgb[1]) ** 2 + (b - rgb[2]) ** 2
        if d < best_d:
            best, best_d = i, d
    return best


def _scatter(ctx, layer):
    meshes = ctx["meshes"].get(layer["category"]) or ctx["meshes"].get("grass") or []
    coverage = layer.get("coverage")
    if not meshes or not coverage:
        return
    analysis = ctx["analysis"]
    start = analysis.get("terrain_start_row", 0) / float(analysis["height"])
    size = len(coverage)
    rng = ctx["rng"]
    colors = layer.get("colors") or {}
    palette = colors.get("palette") or [colors.get("dominant", "#6A8F4A")]
    made = 0
    for gy in range(size):
        v_img = (gy + 0.5) / size
        if v_img < start:
            continue
        for gx in range(size):
            cov = coverage[gy][gx]
            if cov < 0.15:
                continue
            count = int(cov * 3.0 * ctx["density"] + rng.random())
            for _ in range(count):
                if made >= 150:
                    return
                u = (gx + rng.random()) / size
                v = ((gy + rng.random()) / size - start) / max(0.05, 1.0 - start)
                v = max(0.0, min(1.0, depth_v(analysis, layer, v)))
                z = _wb()._sample_height(ctx["heights"], u, v)
                if ctx["water_z"] is not None and z < ctx["water_z"] + 8.0:
                    continue
                mesh = rng.choice(meshes)
                scale = rng.uniform(1.0, 1.9) * max(0.4, ctx["size"])
                bounds = mesh.get_bounds()
                pos = unreal.Vector((u - 0.5) * ctx["world_x"], (v - 0.5) * ctx["world_y"],
                                    z - (bounds.origin.z - bounds.box_extent.z) * scale)
                label = _label(ctx, layer, "s%03d" % made)
                actor, component = _spawn(ctx, mesh, pos, (scale, scale, scale), label, yaw=rng.uniform(0, 360))
                _register(label, (scale, scale, scale))
                cell = layer["cell_colors"][gy][gx]
                q = palette[_nearest(palette, cell)]
                tints = {"dominant": q, "top": q, "accent": q, "bottom": colors.get("dark", q), "dark": colors.get("dark", q)}
                _dress(ctx, [(actor, component, mesh, None)], None, tints, "unused")
                made += 1


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_objects(analysis, values, heights, world_x, world_y, water_z):
    wb = _wb()
    psd = _safe(analysis.get("name", "ref"), 24)
    ctx = {
        "analysis": analysis, "values": values, "heights": heights,
        "world_x": world_x, "world_y": world_y, "water_z": water_z,
        "paint": bool(values.get("paint_art", True)), "blend": float(values.get("art_blend", 0.8)),
        "size": float(values.get("object_scale", 1.0)), "density": float(values.get("density", 0.55)),
        "psd": psd, "folder": LAYER_ROOT + "/" + psd, "rng": random.Random(int(values.get("seed", 1))),
        "occupied": [], "count": 0,
    }
    ctx["masters"] = _masters(ctx["paint"])
    ctx["shapes"] = {key: unreal.load_asset(path) for key, path in SHAPES.items()}
    ctx["meshes"] = {
        category: [m for m in (unreal.load_asset(KENNEY + "/" + n) for n in names) if isinstance(m, unreal.StaticMesh)]
        for category, names in NATURE_MESHES.items()
    }
    try:
        textures = import_textures(ctx)
    except Exception as error:
        unreal.log_warning("Reference World Builder: artwork import failed (%s); using colours only." % error)
        textures = {}

    report = []
    for layer in analysis.get("layers") or []:
        kind, category = layer.get("kind"), layer.get("category")
        before = ctx["count"]
        try:
            if kind == "sky":
                _sky(ctx, layer, textures.get(_texture_name(ctx, layer, "sky")))
            elif kind == "scatter":
                _scatter(ctx, layer)
            elif kind in ("object", "card"):
                for index, inst in enumerate(layer.get("instances") or []):
                    texture = textures.get(_texture_name(ctx, layer, index))
                    if kind == "card":
                        _card(ctx, layer, inst, index, texture)
                    elif category in NATURE_MESHES:
                        _nature(ctx, layer, inst, index, texture)
                    elif category in STRUCTURES:
                        _structure(ctx, layer, inst, index, texture)
                    else:
                        _card(ctx, layer, inst, index, texture)
        except Exception as error:
            import traceback
            unreal.log_error(traceback.format_exc())
            report.append("%s -> FAILED (%s)" % (layer.get("name"), error))
            continue
        made = ctx["count"] - before
        tag = {"fg": " foreground", "mg": " middle ground", "bg": " background"}.get(layer.get("depth"), "")
        line = "%s -> %s (%s%s)%s" % (layer.get("name"), category, kind, tag, "  [%d actors]" % made if made else "")
        report.append(line)
        unreal.log("Reference World Builder layer: " + line)
    try:
        unreal.EditorAssetLibrary.save_directory(ctx["folder"], only_if_is_dirty=True, recursive=True)
    except Exception as error:
        unreal.log_warning("Reference World Builder: could not save layer assets (%s)" % error)
    return {"report": report, "occupied": ctx["occupied"], "count": ctx["count"]}
