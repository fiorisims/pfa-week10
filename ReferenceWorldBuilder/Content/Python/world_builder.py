"""Reference World Builder - import an image, walk through its colours.

The tool reads a reference image (PNG/JPG or a Photoshop PSD/PSB) through the
system Python helper ``reference_analyze.py``.  Its palette supplies the mood,
its luminance becomes a walkable heightfield, and its colours decide where
trees, rocks, flowers and grass appear.  Everything is generated with normal
Unreal Python, so there is no C++ compile step.

Run by Unreal from ``init_unreal.py``, or open the panel from
Tools > Open Reference World Builder.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import unreal

import rwb_layers
import rwb_scene
import rwb_world

# project_dir() is relative to the engine's folder: helpers run from elsewhere, so it must be absolute
PROJECT_DIR = Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir()))
ROOT_DIR = PROJECT_DIR.parent
ANALYZE_SCRIPT = ROOT_DIR / "reference_analyze.py"
PICK_SCRIPT = ROOT_DIR / "reference_pick.py"
KEYWORDS_FILE = ROOT_DIR / "rwb_layer_keywords.json"
SCENE_SCRIPT = ROOT_DIR / "reference_scene.py"

ASSET_PATH = "/Game/ReferenceWorldBuilder/EUW_ReferenceWorldBuilder"
OLD_EUW_PATH = "/Game/WorldBuilder/EUW_WorldBuilderPanel"
SOURCE_EUW_PATH = "/Landmass/Landscape/ErosionBrush/BP/Widgets/EUW_Landscape_Erosion"
HOST_PANEL = "VerticalBox_SettingsPreset"

MATERIAL_DIR = "/Game/ReferenceWorldBuilder/Materials"
KENNEY = "/Game/WorldBuilder/Imported/Kenney"

GENERATED_TAG = "RWB_Generated"
PROP_TAG = "RWB_Prop"
TERRAIN_TAG = "RWB_Terrain"
WATER_TAG = "RWB_Water"
LIGHT_TAG = "RWB_Light"
REACTION_TAG = "RWB_ReactionLight"
PLAYER_TAG = "RWB_PlayerStart"
TEMPLATE_HIDDEN_TAG = "RWB_TemplateHidden"

MENU_OWNER = "ReferenceWorldBuilderMenu"
FOLDER = "ReferenceWorldBuilder"

TREE_MESHES = ("tree_detailed", "tree_oak", "tree_pineRoundA", "tree_pineTallB")
ROCK_MESHES = ("rock_largeB", "rock_largeE", "rock_smallA", "rock_smallH", "cliff_blockSlope_rock")
FLOWER_MESHES = ("flower_purpleA", "flower_redB", "flower_yellowC")
GRASS_MESHES = ("grass", "grass_large", "plant_bushDetailed", "plant_bushSmall")

_callbacks = {}
_bound_widget = None
_watch_handle = None
_tick_handle = None
_widget_asset = None
_roboto_font = None

_state = {
    "analysis": None,
    "path": "",
    "values": {},
    "base_scales": {},
    "prop_labels": set(),
    "reactions": True,
    "accent": "#FFD27A",
    "water_color": "#2E7FA8",
    "tick_error_at": 0.0,
}

_pie = {
    "world": None,
    "props": [],
    "light": None,
    "near": set(),
}

_materials = {}


# --------------------------------------------------------------------------- #
# small Unreal helpers
# --------------------------------------------------------------------------- #
def _margin(left=8.0, top=3.0, right=8.0, bottom=3.0):
    return unreal.Margin(left, top, right, bottom)


def _new(cls, owner, name):
    return cls(outer=owner, name=name)


def _add(panel, child, padding=None, fill=False):
    slot = panel.add_child(child)
    if padding is not None and hasattr(slot, "set_padding"):
        slot.set_padding(padding)
    if fill and hasattr(slot, "set_size"):
        size = unreal.SlateChildSize()
        size.set_editor_property("size_rule", unreal.SlateSizeRule.FILL)
        size.set_editor_property("value", 1.0)
        slot.set_size(size)
    return slot


def _font(size, bold=False):
    global _roboto_font
    if _roboto_font is None:
        _roboto_font = unreal.load_asset("/Engine/EngineFonts/Roboto")
    if _roboto_font:
        return unreal.SlateFontInfo(
            font_object=_roboto_font,
            typeface_font_name="Bold" if bold else "Regular",
            size=size,
        )
    return unreal.SlateFontInfo(size=size)


def _text(owner, name, text, strong=False, color=None, size=13):
    item = _new(unreal.TextBlock, owner, name)
    item.set_text(text)
    item.set_auto_wrap_text(True)
    item.set_font(_font(size, strong))
    if color:
        item.set_color_and_opacity(unreal.SlateColor(color))
    return item


def _button(owner, name, text, tone="slate"):
    item = _new(unreal.Button, owner, name)
    colors = {
        "forest": unreal.LinearColor(0.07, 0.31, 0.21, 1.0),
        "clay": unreal.LinearColor(0.46, 0.26, 0.15, 1.0),
        "sky": unreal.LinearColor(0.10, 0.31, 0.47, 1.0),
        "slate": unreal.LinearColor(0.16, 0.20, 0.24, 1.0),
    }
    try:
        item.set_background_color(colors.get(tone, colors["slate"]))
    except Exception:
        pass
    item.add_child(_text(owner, name + "Label", text, True, unreal.LinearColor(0.95, 0.96, 0.92, 1.0), 12))
    return item


def _section(owner, parent, name, title, caption=None):
    card = _new(unreal.Border, owner, name + "Card")
    try:
        card.set_brush_color(unreal.LinearColor(0.045, 0.075, 0.075, 1.0))
        card.set_padding(_margin(8, 7, 8, 7))
    except Exception:
        pass
    content = _new(unreal.VerticalBox, owner, name + "Content")
    card.add_child(content)
    _add(content, _text(owner, name + "Title", title, True, unreal.LinearColor(0.72, 0.90, 0.84, 1.0), 12), _margin(3, 1, 3, 1))
    if caption:
        _add(content, _text(owner, name + "Caption", caption, False, unreal.LinearColor(0.52, 0.67, 0.63, 1.0), 10), _margin(3, 0, 3, 5))
    _add(parent, card, _margin(8, 4, 8, 4))
    return content


def _field_row(owner, panel, key, label, default):
    row = _new(unreal.HorizontalBox, owner, key + "Row")
    caption = _text(owner, key + "Label", label)
    caption.set_min_desired_width(112.0)
    _add(row, caption, _margin(8, 3, 4, 3))
    field = _new(unreal.EditableTextBox, owner, key + "Box")
    field.set_text(default)
    _add(row, field, _margin(0, 3, 8, 3), fill=True)
    _add(panel, row)


def _slider_row(owner, panel, key, label, value, minimum, maximum, decimals=False):
    row = _new(unreal.HorizontalBox, owner, key + "Row")
    caption = _text(owner, key + "Label", label)
    caption.set_min_desired_width(112.0)
    _add(row, caption, _margin(8, 3, 4, 3))
    slider = _new(unreal.Slider, owner, key + "Slider")
    slider.set_min_value(minimum)
    slider.set_max_value(maximum)
    slider.set_value(value)
    _add(row, slider, _margin(0, 5, 8, 5), fill=True)
    readout = _text(owner, key + "Readout", f"{value:.2f}" if decimals else str(round(value)))
    readout.set_min_desired_width(52.0)
    _add(row, readout, _margin(0, 3, 8, 3))
    _add(panel, row)


def _check_row(owner, panel, key, label, checked=False):
    row = _new(unreal.HorizontalBox, owner, key + "Row")
    box = _new(unreal.CheckBox, owner, key + "Box")
    box.set_is_checked(checked)
    _add(row, box, _margin(8, 5, 5, 5))
    _add(row, _text(owner, key + "Label", label, size=12), _margin(0, 5, 8, 5), fill=True)
    _add(panel, row)


def _find(widget, name):
    item = widget.find_child_widget_by_name(name)
    if not item:
        raise RuntimeError("Reference World Builder UI element is missing: " + name)
    return item


def _maybe(widget, name):
    return widget.find_child_widget_by_name(name)


def _status(widget, message):
    try:
        _find(widget, "StatusText").set_text(message)
    except Exception:
        pass
    unreal.log("Reference World Builder: " + message)


def _refresh_readouts(widget):
    for key in ("WorldSize", "Height", "Density", "Radius", "ArtBlend", "ObjectScale", "Fov", "StrokeStrength", "ModelDetail"):
        item = _maybe(widget, key + "Readout")
        slider = _maybe(widget, key + "Slider")
        if item and slider:
            value = slider.get_value()
            item.set_text(f"{value:.2f}" if key in ("Density", "ArtBlend", "ObjectScale", "StrokeStrength", "ModelDetail") else str(round(value)))


# --------------------------------------------------------------------------- #
# system-python bridge
# --------------------------------------------------------------------------- #
def _system_python():
    candidates = [
        shutil.which("python.exe"),
        shutil.which("python"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python312\python.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python311\python.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Python312\python.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Python311\python.exe"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise RuntimeError("Python with pygame was not found. Install it or put it on PATH.")


class Cancelled(Exception):
    """The user pressed Cancel on the progress window."""


_progress = {"task": None, "at": 0.0}


class busy:
    """A progress window for long jobs.  Without one Unreal freezes and looks like nothing happened.

    with busy("Reading your picture"): ...  Nested uses share the outermost window.
    """

    def __init__(self, title):
        self.title = title
        self.task = None

    def __enter__(self):
        if _progress["task"] is None:
            self.task = unreal.ScopedSlowTask(100.0, self.title)
            self.task.__enter__()
            self.task.make_dialog(True)
            _progress["task"] = self.task
            _progress["at"] = 0.0
            self.task.enter_progress_frame(0.0, self.title)
        return self

    def __exit__(self, kind, value, trace):
        if self.task is not None:
            _progress["task"] = None
            self.task.__exit__(kind, value, trace)
        return False


def _progress_message(message, to=None):
    """Repaint the progress window; ``to`` = percent done (it never moves backwards).

    Raises Cancelled if its Cancel button was pressed.
    """
    task = _progress["task"]
    if task is None:
        return
    advance = max(0.0, float(to) - _progress["at"]) if to is not None else 0.0
    _progress["at"] += advance
    task.enter_progress_frame(advance, message)
    if task.should_cancel():
        raise Cancelled()


def _read_progress(path):
    """(fraction, text) a helper last reported through rwb_progress.py."""
    try:
        with open(path, encoding="utf-8") as src:
            fraction, _, text = src.read().partition("|")
        return float(fraction), text
    except (OSError, ValueError):
        return 0.0, ""


def _run_helper(script, args, timeout=240, message=None, span=(0.0, 100.0)):
    """Run a system-Python script.  With a ``message`` the progress window follows what it reports.

    span: the part of the progress bar (in percent) this helper fills.
    """
    command = [_system_python(), str(script)] + list(args)
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    environment["SDL_VIDEODRIVER"] = "dummy"
    environment["SDL_AUDIODRIVER"] = "dummy"
    creation = 0x08000000 if os.name == "nt" else 0
    progress_file = os.path.join(tempfile.gettempdir(), "rwb_progress_%d_%d.txt" % (os.getpid(), int(time.time() * 1000)))
    environment["RWB_PROGRESS"] = progress_file
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        process = subprocess.Popen(command, cwd=str(ROOT_DIR), stdout=out, stderr=err, env=environment,
                                   creationflags=creation)
        started = time.time()
        try:
            while process.poll() is None:
                elapsed = time.time() - started
                if elapsed > timeout:
                    raise RuntimeError("%s took longer than %d minutes" % (Path(script).name, timeout // 60))
                if message:
                    fraction, text = _read_progress(progress_file)
                    _progress_message("%s  (%d s)" % (text or message, elapsed),
                                      span[0] + (span[1] - span[0]) * fraction)
                time.sleep(0.1)
        finally:
            if process.poll() is None:              # cancelled or timed out: don't leave it running
                process.kill()
                process.wait()
            try:
                os.remove(progress_file)
            except OSError:
                pass
        out.seek(0)
        err.seek(0)
        stdout = out.read().decode("utf-8-sig", "replace")
        stderr = err.read().decode("utf-8-sig", "replace")
    if process.returncode != 0:
        detail = (stderr or stdout or "unknown error").strip()
        raise RuntimeError(detail.splitlines()[-1] if detail else "helper failed")
    return stdout


def _last_json(text):
    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        try:
            return json.loads(line)
        except ValueError:
            continue
    raise RuntimeError("the analyzer did not return JSON")


def analyze_reference(path):
    if not ANALYZE_SCRIPT.is_file():
        raise RuntimeError("reference_analyze.py is missing next to the Unreal project")
    stem = "".join(c if c.isalnum() else "_" for c in Path(path).stem)[:40] or "reference"
    layers_dir = PROJECT_DIR / "Saved" / "RWBLayers" / stem
    stdout = _run_helper(
        ANALYZE_SCRIPT,
        [str(path), "--grid", "24", "--colors", "6", "--layers-dir", str(layers_dir), "--keywords", str(KEYWORDS_FILE)]
        + ([] if _setting["choice"] == "auto" else ["--setting", _setting["choice"]]),
        timeout=1200,
        message="Reading your picture: objects, outlines, brushstrokes (about a minute)",
    )
    return _last_json(stdout)


def pick_reference():
    if not PICK_SCRIPT.is_file():
        return ""
    stdout = _run_helper(PICK_SCRIPT, [])
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    return lines[-1] if lines else ""


# --------------------------------------------------------------------------- #
# colours + materials
# --------------------------------------------------------------------------- #
def _hex_rgb(value):
    value = str(value).strip().lstrip("#")
    if len(value) != 6:
        return (90, 120, 140)
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _srgb_to_linear(channel):
    value = channel / 255.0
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _linear(value):
    r, g, b = _hex_rgb(value)
    return unreal.LinearColor(_srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b), 1.0)


def _color(value):
    r, g, b = _hex_rgb(value)
    return unreal.Color(r, g, b, 255)


def _mix_hex(a, b, t):
    ra, ga, ba = _hex_rgb(a)
    rb, gb, bb = _hex_rgb(b)
    return "#%02X%02X%02X" % (
        int(ra + (rb - ra) * t),
        int(ga + (gb - ga) * t),
        int(ba + (bb - ba) * t),
    )


def _asset(name):
    return unreal.load_asset(name)


def _create_material(name):
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    return tools.create_asset(name, MATERIAL_DIR, unreal.Material, unreal.MaterialFactoryNew())


def _save(asset):
    if asset:
        unreal.EditorAssetLibrary.save_loaded_asset(asset, False)
    return asset


def _terrain_material():
    key = "terrain"
    if key in _materials:
        return _materials[key]
    path = MATERIAL_DIR + "/M_RWB_Terrain"
    material = _asset(path)
    if not material:
        material = _create_material("M_RWB_Terrain")
        library = unreal.MaterialEditingLibrary
        vertex = library.create_material_expression(material, unreal.MaterialExpressionVertexColor, -460, 0)
        roughness = library.create_material_expression(material, unreal.MaterialExpressionConstant, -460, 160)
        roughness.set_editor_property("r", 0.92)
        library.connect_material_property(vertex, "", unreal.MaterialProperty.MP_BASE_COLOR)
        library.connect_material_property(roughness, "", unreal.MaterialProperty.MP_ROUGHNESS)
        library.recompile_material(material)
        _save(material)
    _materials[key] = material
    return material


def _water_material(color_hex):
    key = "water"
    base_path = MATERIAL_DIR + "/M_RWB_Water"
    base = _asset(base_path)
    if not base:
        base = _create_material("M_RWB_Water")
        library = unreal.MaterialEditingLibrary
        base.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
        base.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
        tint = library.create_material_expression(base, unreal.MaterialExpressionVectorParameter, -460, 0)
        tint.set_editor_property("parameter_name", "WaterColor")
        tint.set_editor_property("default_value", _linear(color_hex))
        opacity = library.create_material_expression(base, unreal.MaterialExpressionConstant, -460, 160)
        opacity.set_editor_property("r", 0.66)
        library.connect_material_property(tint, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
        library.connect_material_property(opacity, "", unreal.MaterialProperty.MP_OPACITY)
        library.recompile_material(base)
        _save(base)

    instance_path = MATERIAL_DIR + "/MI_RWB_Water"
    instance = _asset(instance_path)
    if not instance:
        tools = unreal.AssetToolsHelpers.get_asset_tools()
        instance = tools.create_asset(
            "MI_RWB_Water", MATERIAL_DIR, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew()
        )
    if instance:
        unreal.MaterialEditingLibrary.set_material_instance_parent(instance, base)
        unreal.MaterialEditingLibrary.set_material_instance_vector_parameter_value(instance, "WaterColor", _linear(color_hex))
        unreal.MaterialEditingLibrary.update_material_instance(instance)
        _save(instance)
        _materials[key] = instance
        return instance
    _materials[key] = base
    return base


def _overlay_material(color_hex):
    key = "overlay"
    base_path = MATERIAL_DIR + "/M_RWB_Overlay"
    base = _asset(base_path)
    if not base:
        base = _create_material("M_RWB_Overlay")
        library = unreal.MaterialEditingLibrary
        base.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
        base.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
        tint = library.create_material_expression(base, unreal.MaterialExpressionVectorParameter, -460, 0)
        tint.set_editor_property("parameter_name", "GlowColor")
        tint.set_editor_property("default_value", _linear(color_hex))
        opacity = library.create_material_expression(base, unreal.MaterialExpressionConstant, -460, 160)
        opacity.set_editor_property("r", 0.26)
        library.connect_material_property(tint, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
        library.connect_material_property(opacity, "", unreal.MaterialProperty.MP_OPACITY)
        library.recompile_material(base)
        _save(base)

    instance_path = MATERIAL_DIR + "/MI_RWB_Overlay"
    instance = _asset(instance_path)
    if not instance:
        tools = unreal.AssetToolsHelpers.get_asset_tools()
        instance = tools.create_asset(
            "MI_RWB_Overlay", MATERIAL_DIR, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew()
        )
    if instance:
        unreal.MaterialEditingLibrary.set_material_instance_parent(instance, base)
        unreal.MaterialEditingLibrary.set_material_instance_vector_parameter_value(instance, "GlowColor", _linear(color_hex))
        unreal.MaterialEditingLibrary.update_material_instance(instance)
        _save(instance)
        _materials[key] = instance
        return instance
    _materials[key] = base
    return base


# --------------------------------------------------------------------------- #
# actor tags + clearing
# --------------------------------------------------------------------------- #
def _tags(actor):
    try:
        return [str(tag) for tag in actor.get_editor_property("tags")]
    except Exception:
        return []


def _has_tag(actor, tag):
    return tag in _tags(actor)


def _tag_actor(actor, *tags, label=None, folder=FOLDER):
    actor.set_editor_property("tags", [unreal.Name(tag) for tag in tags])
    if label:
        actor.set_actor_label(label)
    if folder and hasattr(actor, "set_folder_path"):
        actor.set_folder_path(folder)
    return actor


def _all_actors():
    return unreal.EditorLevelLibrary.get_all_level_actors()


def _restore_template():
    for actor in _all_actors():
        if _has_tag(actor, TEMPLATE_HIDDEN_TAG):
            tags = [tag for tag in _tags(actor) if tag != TEMPLATE_HIDDEN_TAG]
            actor.set_editor_property("tags", [unreal.Name(tag) for tag in tags])
            try:
                actor.set_is_temporarily_hidden_in_editor(False)
                actor.set_actor_hidden_in_game(False)
            except Exception:
                pass


def _clear_generated():
    for actor in list(_all_actors()):
        if _has_tag(actor, GENERATED_TAG):
            try:
                unreal.EditorLevelLibrary.destroy_actor(actor)
            except Exception:
                pass
    _restore_template()
    _state["water_z"] = None                    # no water until a build makes some
    _state["base_scales"] = {}
    _state["prop_labels"] = set()
    _pie["world"] = None
    _pie["props"] = []
    _pie["near"] = set()
    _pie["light"] = None


def _hide_template():
    for actor in _all_actors():
        if _has_tag(actor, GENERATED_TAG) or _has_tag(actor, TEMPLATE_HIDDEN_TAG):
            continue
        hide = isinstance(actor, unreal.StaticMeshActor)
        if not hide:
            try:
                hide = isinstance(actor, unreal.LandscapeProxy)
            except Exception:
                hide = "Landscape" in actor.get_class().get_name()
        if not hide:
            continue
        tags = _tags(actor)
        tags.append(TEMPLATE_HIDDEN_TAG)
        actor.set_editor_property("tags", [unreal.Name(tag) for tag in tags])
        try:
            actor.set_is_temporarily_hidden_in_editor(True)
            actor.set_actor_hidden_in_game(True)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# heightfield
# --------------------------------------------------------------------------- #
def _smooth(values, passes=2):
    size = len(values)
    for _ in range(passes):
        copy = [row[:] for row in values]
        for y in range(size):
            for x in range(size):
                total = 0.0
                count = 0
                for oy in (-1, 0, 1):
                    for ox in (-1, 0, 1):
                        nx, ny = x + ox, y + oy
                        if 0 <= nx < size and 0 <= ny < size:
                            total += copy[ny][nx]
                            count += 1
                values[y][x] = total / float(count)
    return values


def _heightfield(data, vertices, height_range):
    grid = data["grid"]
    size = int(grid["size"])
    lum = grid["luminance"]
    start_row = int(round(data.get("terrain_start_row", 0) / float(max(1, data["height"])) * (size - 1)))
    start_row = max(0, min(size - 2, start_row))
    rows = max(2, size - start_row)

    raw = [[0.0 for _ in range(vertices)] for _ in range(vertices)]
    for iy in range(vertices):
        v = iy / float(vertices - 1)
        fy = start_row + v * (rows - 1)
        y0 = int(math.floor(fy))
        y1 = min(size - 1, y0 + 1)
        ty = fy - y0
        for ix in range(vertices):
            u = ix / float(vertices - 1)
            fx = u * (size - 1)
            x0 = int(math.floor(fx))
            x1 = min(size - 1, x0 + 1)
            tx = fx - x0
            top = lum[y0][x0] * (1.0 - tx) + lum[y0][x1] * tx
            bottom = lum[y1][x0] * (1.0 - tx) + lum[y1][x1] * tx
            raw[iy][ix] = top * (1.0 - ty) + bottom * ty

    low = min(min(row) for row in raw)
    high = max(max(row) for row in raw)
    span = max(0.001, high - low)
    norm = [[(value - low) / span for value in row] for row in raw]
    _smooth(norm, 2)
    heights = [[10.0 + (max(0.0, min(1.0, value)) ** 1.18) * height_range for value in row] for row in norm]
    return heights, norm


def _sample_height(heights, u, v):
    size = len(heights)
    fx = max(0.0, min(1.0, u)) * (size - 1)
    fy = max(0.0, min(1.0, v)) * (size - 1)
    x0 = int(math.floor(fx))
    y0 = int(math.floor(fy))
    x1 = min(size - 1, x0 + 1)
    y1 = min(size - 1, y0 + 1)
    tx = fx - x0
    ty = fy - y0
    top = heights[y0][x0] * (1.0 - tx) + heights[y0][x1] * tx
    bottom = heights[y1][x0] * (1.0 - tx) + heights[y1][x1] * tx
    return top * (1.0 - ty) + bottom * ty


def _terrain_v(data, v):
    start = data.get("terrain_start_row", 0) / float(max(1, data["height"]))
    return max(0.0, min(1.0, start + v * (1.0 - start)))


def _sample_color(data, u, v):
    grid = data["grid"]
    size = int(grid["size"])
    x = max(0, min(size - 1, int(round(u * (size - 1)))))
    y = max(0, min(size - 1, int(round(_terrain_v(data, v) * (size - 1)))))
    return tuple(grid["colors"][y][x])


def _sample_luminance(data, u, v):
    grid = data["grid"]
    size = int(grid["size"])
    x = max(0, min(size - 1, int(round(u * (size - 1)))))
    y = max(0, min(size - 1, int(round(_terrain_v(data, v) * (size - 1)))))
    return float(grid["luminance"][y][x])


def _build_terrain(data, world_x, world_y, height_range, vertices=40, heights=None, norm=None):
    if heights is None or norm is None:
        heights, norm = _heightfield(data, vertices, height_range)
    vertices = len(heights)
    dx = world_x / float(vertices - 1)
    dy = world_y / float(vertices - 1)
    verts = []
    normals = []
    uvs = []
    colors = []
    tangents = []
    accent = _hex_rgb(_state["accent"])

    for iy in range(vertices):
        for ix in range(vertices):
            x = -world_x * 0.5 + ix * dx
            y = -world_y * 0.5 + iy * dy
            z = heights[iy][ix]
            verts.append(unreal.Vector(x, y, z))
            left = heights[iy][max(0, ix - 1)]
            right = heights[iy][min(vertices - 1, ix + 1)]
            down = heights[max(0, iy - 1)][ix]
            up = heights[min(vertices - 1, iy + 1)][ix]
            normal = unreal.Vector(-(right - left) / (2.0 * dx), -(up - down) / (2.0 * dy), 1.0)
            normal.normalize()
            normals.append(normal)
            u = ix / float(vertices - 1)
            v = iy / float(vertices - 1)
            uvs.append(unreal.Vector2D(u, v))
            tangents.append(unreal.ProcMeshTangent())

            red, green, blue = _sample_color(data, u, v)
            level = norm[iy][ix]
            shade = 0.72 + 0.42 * level
            peak = max(0.0, (level - 0.78) / 0.22) * 0.22
            red = min(255, int(red * shade * (1.0 - peak) + accent[0] * peak))
            green = min(255, int(green * shade * (1.0 - peak) + accent[1] * peak))
            blue = min(255, int(blue * shade * (1.0 - peak) + accent[2] * peak))
            colors.append(
                unreal.LinearColor(
                    _srgb_to_linear(red), _srgb_to_linear(green), _srgb_to_linear(blue), 1.0
                )
            )

    triangles = []
    for iy in range(vertices - 1):
        for ix in range(vertices - 1):
            a = iy * vertices + ix
            b = a + 1
            c = a + vertices + 1
            d = a + vertices
            # counter-clockwise seen from above: procedural meshes cull the other way round
            triangles.extend([a, c, b, a, d, c])

    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.Actor, unreal.Vector(0, 0, 0))
    component = unreal.new_object(unreal.ProceduralMeshComponent, outer=actor, name="RWBTerrain")
    actor.set_editor_property("root_component", component)
    component.set_editor_property("mobility", unreal.ComponentMobility.STATIC)
    component.create_mesh_section_linear_color(
        0, verts, triangles, normals, uvs, [], [], [], colors, tangents, True
    )
    component.set_material(0, _terrain_material())
    component.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS)
    component.set_editor_property("cast_shadow", True)
    _tag_actor(actor, GENERATED_TAG, TERRAIN_TAG, label="RWB_Terrain")
    return actor, heights


def _build_water(world_x, world_y, water_z, color_hex):
    plane = _asset("/Engine/BasicShapes/Plane")
    if not isinstance(plane, unreal.StaticMesh):
        raise RuntimeError("the engine Plane mesh is missing")
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.StaticMeshActor, unreal.Vector(0, 0, water_z), unreal.Rotator()
    )
    if not actor:
        raise RuntimeError("could not spawn the water plane")
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    if not component:
        raise RuntimeError("the water plane has no static mesh component")
    component.set_static_mesh(plane)
    component.set_material(0, _water_material(color_hex))
    # water is something you go into, in every build mode: the terrain under it is what you stand on
    component.set_collision_profile_name("NoCollision")
    component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    _state["water_z"] = water_z                 # so wading works wherever water exists
    actor.set_actor_scale3d(unreal.Vector(world_x / 100.0, world_y / 100.0, 1.0))
    _tag_actor(actor, GENERATED_TAG, WATER_TAG, label="RWB_Water")
    return actor


# --------------------------------------------------------------------------- #
# props
# --------------------------------------------------------------------------- #
def _load_meshes(names):
    meshes = []
    for name in names:
        asset = _asset(KENNEY + "/" + name)
        if isinstance(asset, unreal.StaticMesh):
            meshes.append(asset)
    return meshes


def _spawn_prop(mesh, x, y, z, scale, yaw, label, color):
    bounds = mesh.get_bounds()
    ground = -(bounds.origin.z - bounds.box_extent.z) * scale
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.StaticMeshActor, unreal.Vector(x, y, z + ground), unreal.Rotator(yaw=yaw)
    )
    if not actor:
        return None
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    if not component:
        unreal.EditorLevelLibrary.destroy_actor(actor)
        return None
    component.set_static_mesh(mesh)
    component.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    actor.set_actor_scale3d(unreal.Vector(scale, scale, scale))
    _tag_actor(actor, GENERATED_TAG, PROP_TAG, str(label), label=label)
    _state["base_scales"][label] = (scale, scale, scale)
    _state["prop_labels"].add(label)
    return actor


def _build_props(data, heights, world_x, world_y, water_z, density, seed):
    rng = random.Random(seed)
    tree_meshes = _load_meshes(TREE_MESHES)
    rock_meshes = _load_meshes(ROCK_MESHES)
    flower_meshes = _load_meshes(FLOWER_MESHES)
    grass_meshes = _load_meshes(GRASS_MESHES)
    groups = {"tree": tree_meshes, "rock": rock_meshes, "flower": flower_meshes, "grass": grass_meshes}
    if not any(groups.values()):
        return []

    count = 26
    made = []
    for iy in range(count):
        for ix in range(count):
            u = (ix + 0.5) / count
            v = (iy + 0.5) / count
            x = (u - 0.5) * world_x
            y = (v - 0.5) * world_y
            z = _sample_height(heights, u, v)
            if z <= water_z + 12.0:
                continue
            if rng.random() > density * 0.92:
                continue
            lum = _sample_luminance(data, u, v)
            red, green, blue = _sample_color(data, u, v)
            high = max(red, green, blue) / 255.0
            low = min(red, green, blue) / 255.0
            saturation = 0.0 if high <= 0 else (high - low) / high
            if high == low:
                hue = 0.0
            elif high == red:
                hue = (60.0 * ((green - blue) / 255.0 / max(0.001, high - low))) % 360.0
            elif high == green:
                hue = 60.0 * ((blue - red) / 255.0 / max(0.001, high - low)) + 120.0
            else:
                hue = 60.0 * ((red - green) / 255.0 / max(0.001, high - low)) + 240.0

            roll = rng.random()
            if lum < 0.30 and saturation < 0.48:
                group = "rock" if roll < 0.62 else "grass"
            elif 70.0 <= hue <= 170.0:
                group = "tree" if roll < 0.60 else "grass"
            elif saturation > 0.42 and (hue < 42.0 or hue > 302.0):
                group = "flower" if roll < 0.55 else "grass"
            else:
                group = "grass" if roll < 0.58 else "rock"

            meshes = groups.get(group) or [m for items in groups.values() for m in items]
            if not meshes:
                continue
            mesh = rng.choice(meshes)
            if group == "tree":
                scale = rng.uniform(1.0, 2.2)
            elif group == "rock":
                scale = rng.uniform(0.7, 1.9)
            elif group == "flower":
                scale = rng.uniform(1.0, 1.7)
            else:
                scale = rng.uniform(1.1, 2.0)
            label = "RWB_Prop_%03d" % (len(made) + 1)
            color = "#%02X%02X%02X" % (red, green, blue)
            actor = _spawn_prop(mesh, x, y, z, scale, rng.uniform(0.0, 360.0), label, color)
            if actor:
                made.append(actor)
            if len(made) >= 190:
                return made

    if not made:
        for index in range(10):
            u = rng.uniform(0.35, 0.65)
            v = rng.uniform(0.35, 0.65)
            x = (u - 0.5) * world_x
            y = (v - 0.5) * world_y
            z = _sample_height(heights, u, v)
            mesh = rng.choice(groups["tree"] or groups["grass"]) if (groups["tree"] or groups["grass"]) else None
            if mesh:
                label = "RWB_Prop_%03d" % (len(made) + 1)
                actor = _spawn_prop(mesh, x, y, z, rng.uniform(1.0, 1.8), rng.uniform(0, 360), label, "#709060")
                if actor:
                    made.append(actor)
    return made


# --------------------------------------------------------------------------- #
# lighting + spawn
# --------------------------------------------------------------------------- #
def _lighting(data, world_x, world_y, heights, overrides=None):
    overrides = overrides or {}
    mood = data["mood"]
    palette = data.get("palette") or []
    accent = overrides.get("accent") or mood.get("accent", "#FFD27A")
    warm = max(palette, key=lambda item: item["rgb"][0] - item["rgb"][2]) if palette else {"hex": "#FFE7C2"}
    cool = max(palette, key=lambda item: item["rgb"][2] - item["rgb"][0]) if palette else {"hex": "#8FB8DA"}
    sun_hex = _mix_hex(overrides.get("sun") or warm.get("hex", "#FFE7C2"), "#FFF3DA", 0.55)
    sky_hex = _mix_hex(overrides.get("sky") or cool.get("hex", "#8FB8DA"), "#7FB6E8", 0.35 if overrides.get("sky") else 0.60)
    fog_hex = overrides.get("fog") or _mix_hex(sky_hex, accent, 0.22)
    water_hex = overrides.get("water") or _mix_hex(cool.get("hex", "#8FB8DA"), "#2E7FA8", 0.62)
    _state["accent"] = accent
    _state["water_color"] = water_hex

    actors = _all_actors()
    sun = next((actor for actor in actors if isinstance(actor, unreal.DirectionalLight)), None)
    if not sun:
        sun = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 1400))
        _tag_actor(sun, GENERATED_TAG, LIGHT_TAG, label="RWB_Sun")
    sun_component = sun.get_component_by_class(unreal.DirectionalLightComponent)
    if sun_component:
        sun_component.set_editor_property("intensity", 3.0 + mood["average_luminance"] * 7.0)
        sun_component.set_light_color(_linear(sun_hex))
    sun.set_actor_rotation(unreal.Rotator(pitch=-52.0, yaw=-28.0), False)
    if sun_component:
        try:
            sun_component.set_atmosphere_sun_light(True)
        except Exception:
            pass

    # The startup level is empty, so make sure there is a sky to look at.
    if not any(isinstance(actor, unreal.SkyAtmosphere) for actor in actors):
        atmosphere = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SkyAtmosphere, unreal.Vector(0, 0, 0))
        if atmosphere:
            _tag_actor(atmosphere, GENERATED_TAG, LIGHT_TAG, label="RWB_SkyAtmosphere")

    sky = next((actor for actor in actors if isinstance(actor, unreal.SkyLight)), None)
    if not sky:
        sky = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 1200))
        if sky:
            _tag_actor(sky, GENERATED_TAG, LIGHT_TAG, label="RWB_SkyLight")
            component = sky.get_component_by_class(unreal.SkyLightComponent)
            if component:
                for name, value in (
                    ("mobility", unreal.ComponentMobility.MOVABLE),
                    ("real_time_capture", True),
                ):
                    try:
                        component.set_editor_property(name, value)
                    except Exception:
                        pass
    if sky:
        sky_component = sky.get_component_by_class(unreal.SkyLightComponent)
        if sky_component:
            sky_component.set_intensity(0.7 + mood["average_luminance"] * 1.6)
            sky_component.set_light_color(_linear(sky_hex))
            try:
                sky_component.set_lower_hemisphere_color(_linear(fog_hex))
                sky_component.recapture_sky()
            except Exception:
                pass

    fog = next((actor for actor in actors if isinstance(actor, unreal.ExponentialHeightFog)), None)
    if not fog:
        fog = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.ExponentialHeightFog, unreal.Vector(0, 0, 0))
        if fog:
            _tag_actor(fog, GENERATED_TAG, LIGHT_TAG, label="RWB_Fog")
    if fog:
        fog_component = fog.get_component_by_class(unreal.ExponentialHeightFogComponent)
        if fog_component:
            fog_component.set_fog_density(0.012 + (1.0 - mood["average_luminance"]) * 0.045)
            fog_component.set_fog_inscattering_color(_linear(fog_hex))

    return accent, water_hex


def _place_player(world_x, world_y, heights, water_z, occupied=None):
    occupied = occupied or []

    def free(u, v):
        x, y = (u - 0.5) * world_x, (v - 0.5) * world_y
        return all((x - ox) ** 2 + (y - oy) ** 2 > (r + 120.0) ** 2 for ox, oy, r in occupied)

    # The bottom of a concept image is the foreground, so start there and look
    # "into" the picture (towards the top of the image, which is -Y here).
    best = None
    for v in (0.98, 0.94, 0.89, 0.83, 0.76, 0.68):
        for u in (0.5, 0.42, 0.58, 0.34, 0.66):
            z = _sample_height(heights, u, v)
            if z > water_z + 30.0 and free(u, v):
                best = (u, v, z)
                break
        if best:
            break
    u, v, z = best or (0.5, 0.85, _sample_height(heights, 0.5, 0.85))
    location = unreal.Vector((u - 0.5) * world_x, (v - 0.5) * world_y, z + 150.0)
    starts = [actor for actor in _all_actors() if isinstance(actor, unreal.PlayerStart)]
    if starts:
        start = starts[0]
        start.set_actor_location(location, False, True)
        start.set_actor_rotation(unreal.Rotator(yaw=-90.0), False)
        return start
    start = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PlayerStart, location, unreal.Rotator(yaw=-90.0))
    _tag_actor(start, GENERATED_TAG, PLAYER_TAG, label="RWB_PlayerStart")
    return start


def _reaction_light(location, accent):
    light = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PointLight, location, unreal.Rotator())
    component = light.get_component_by_class(unreal.PointLightComponent)
    if component:
        for name, value in (
            ("mobility", unreal.ComponentMobility.MOVABLE),
            ("intensity", 0.0),
            ("attenuation_radius", 900.0),
        ):
            try:
                component.set_editor_property(name, value)
            except Exception:
                pass
        component.set_light_color(_linear(accent))
    _tag_actor(light, GENERATED_TAG, LIGHT_TAG, REACTION_TAG, label="RWB_ReactionLight")
    return light


def _build_world(widget):
    analysis = _state.get("analysis")
    if not analysis:
        raise RuntimeError("Analyze a reference image first.")
    values = _read_values(widget)
    if not values:
        return
    inventory = analysis.get("inventory") or {}
    if not values.get("exact_layout") and inventory and not inventory.get("error"):
        return _build_scatter_world(widget, analysis, values)
    if inventory.get("error"):
        unreal.log_warning("Reference World Builder: inventory failed (%s); using the layout mode." % inventory["error"])
    if values.get("use_layers") and analysis.get("layers"):
        return _build_world_from_layers(widget, analysis, values)
    if values.get("perspective") and (analysis.get("scene") or {}).get("cache"):
        return _build_world_from_scene(widget, analysis, values)

    world_x = float(values["world_size"])
    world_y = world_x * max(0.62, min(1.55, analysis["height"] / float(max(1, analysis["width"]))))
    heights, norm = _heightfield(analysis, 40, float(values["height"]))
    water_z = 0.0
    if values["water"]:
        flat = sorted(value for row in norm for value in row)
        water_norm = flat[int(len(flat) * 0.22)] if flat else 0.22
        water_norm = max(0.06, min(0.82, water_norm))
        water_z = 10.0 + (water_norm ** 1.18) * float(values["height"])

    accent, water_hex = _lighting(analysis, world_x, world_y, heights)
    _overlay_material(accent)
    _terrain_actor, heights = _build_terrain(analysis, world_x, world_y, float(values["height"]))
    if values["water"]:
        _build_water(world_x, world_y, water_z, water_hex)
    _build_props(analysis, heights, world_x, world_y, water_z, float(values["density"]), int(values["seed"]))
    _place_player(world_x, world_y, heights, water_z)
    start = _maybe_start()
    location = start.get_actor_location() if start else unreal.Vector(0, 0, 400)
    _reaction_light(unreal.Vector(location.x, location.y, location.z + 60.0), accent)
    if values["hide_template"]:
        _hide_template()

    generated = [actor for actor in _all_actors() if _has_tag(actor, GENERATED_TAG)]
    try:
        unreal.get_editor_subsystem(unreal.EditorActorSubsystem).set_selected_level_actors(generated)
    except Exception:
        pass
    _state["values"] = values
    _state["reactions"] = values["reactions"]
    _status(
        widget,
        "Built %d props from %s. Palette: %s. Press Play to walk through it."
        % (len(_state["prop_labels"]), analysis.get("name", "reference"), analysis["mood"]["name"]),
    )


def _build_world_from_layers(widget, analysis, values):
    """PSD path: every named layer becomes the object its name describes."""
    world_x = float(values["world_size"])
    world_y = world_x * max(0.62, min(1.55, analysis["height"] / float(max(1, analysis["width"]))))
    height_range = float(values["height"])
    ground_data, hills, waters = rwb_layers.terrain_inputs(analysis)
    heights, norm, water_z = rwb_layers.layer_heights(analysis, ground_data, 40, height_range, hills, waters)
    overrides = rwb_layers.atmosphere_overrides(analysis)
    accent, water_hex = _lighting(analysis, world_x, world_y, heights, overrides)
    _state["accent"] = accent
    _overlay_material(accent)
    _build_terrain(ground_data, world_x, world_y, height_range, heights=heights, norm=norm)
    if water_z is not None:
        _build_water(world_x, world_y, water_z, water_hex)
    result = rwb_layers.build_objects(analysis, values, heights, world_x, world_y, water_z)
    _place_player(world_x, world_y, heights, water_z if water_z is not None else 0.0, result["occupied"])
    start = _maybe_start()
    location = start.get_actor_location() if start else unreal.Vector(0, 0, 400)
    _reaction_light(unreal.Vector(location.x, location.y, location.z + 60.0), accent)
    if values["hide_template"]:
        _hide_template()
    _state["values"] = values
    _state["reactions"] = values["reactions"]
    info = _maybe(widget, "LayerInfo")
    if info:
        info.set_text("\n".join(result["report"][:30]) or "No layers produced objects.")
    _status(
        widget,
        "Built %d actors from %d PSD layers of %s. Press Play to walk through it."
        % (result["count"], len(analysis.get("layers") or []), analysis.get("name", "reference")),
    )


def _build_scatter_world(widget, analysis, values):
    """Default: fresh terrain, with everything placed where it was painted."""
    result = rwb_world.build_world(analysis, values, ROOT_DIR, PROJECT_DIR, KEYWORDS_FILE)
    _state["accent"] = (analysis.get("inventory") or {}).get("accent") or analysis["mood"].get("accent", "#FFD27A")
    _overlay_material(_state["accent"])
    start = _maybe_start()
    location = start.get_actor_location() if start else unreal.Vector(0, 0, 400)
    _reaction_light(unreal.Vector(location.x, location.y, location.z + 60.0), _state["accent"])
    if values["hide_template"]:
        _hide_template()
    _state["values"] = values
    _state["reactions"] = values["reactions"]
    info = _maybe(widget, "LayerInfo")
    if info:
        info.set_text("\n".join(result["report"][:28]) or "Nothing recognised in the image.")
    for line in result["report"]:
        unreal.log("Reference World Builder: " + line)
    _refresh_model_rows(widget)
    _status(widget, "Built a world with %d actors from %s. Press Play to walk through it."
            % (result["count"], analysis.get("name", "reference")))


# --------------------------------------------------------------------------- #
# per-model counts
# --------------------------------------------------------------------------- #
_row_callbacks = []            # delegates hold weak references: these keep the row buttons alive
_row_readouts = {}
_row_generation = 0
ROW_STEPS = (("-", -1), ("+", 1), ("+5", 5))


def _set_model_count(widget, key, wanted):
    rwb_world.set_keep_out(_checked(widget, "KeepCopiesOutBox", False))
    wanted = max(0, wanted)
    got = rwb_world.set_count(key, wanted)
    readout = _row_readouts.get(key)
    if readout:
        readout.set_text(str(got))
    message = "%s: %d in the world." % (key, got)
    if got < wanted:
        message = "%s: only found room for %d of %d. Try a bigger World size, or generate again." % (key, got, wanted)
    _status(widget, message)
    return got


def _row_action(widget, action):
    def run():
        target = _current_widget(widget)
        try:
            action(target)
        except Exception as error:
            import traceback
            unreal.log_error(traceback.format_exc())
            _status(target, "Could not change that model: %s" % error)

    def callback():
        _queue_action(id(callback), run)
    return callback


def _model_row(widget, parent, row, generation, index):
    prefix = "ModelRow%d_%d_" % (generation, index)
    key = row["key"]
    line = _new(unreal.HorizontalBox, widget, prefix + "Line")
    if row.get("thumb"):
        try:
            frame = _new(unreal.SizeBox, widget, prefix + "Frame")
            frame.set_width_override(40.0)
            frame.set_height_override(40.0)
            image = _new(unreal.Image, widget, prefix + "Thumb")
            image.set_brush_from_texture(row["thumb"], False)
            frame.add_child(image)
            _add(line, frame, _margin(4, 2, 4, 2))
        except Exception as error:
            unreal.log_warning("Reference World Builder: no thumbnail for %s (%s)" % (key, error))
    name = _button(widget, prefix + "Name", row["title"], "slate")
    select = _row_action(widget, lambda target: _status(
        target, "Selected %d of %s in the level." % (rwb_world.select_model(key), row["title"])))
    name.on_clicked.add_callable(select)
    _row_callbacks.append(select)
    _add(line, name, _margin(0, 2, 4, 2), fill=True)
    readout = _text(widget, prefix + "Count", str(row["count"]), True, size=13)
    readout.set_min_desired_width(30.0)
    _row_readouts[key] = readout
    for number, (label, step) in enumerate(ROW_STEPS):
        button = _button(widget, "%sStep%d" % (prefix, number), label, "forest" if step > 0 else "clay")

        def change(target, step=step):
            current = next((r["count"] for r in rwb_world.model_rows() if r["key"] == key), 0)
            _set_model_count(target, key, current + step)

        callback = _row_action(widget, change)
        button.on_clicked.add_callable(callback)
        _row_callbacks.append(callback)
        if number == 1:
            _add(line, readout, _margin(4, 8, 4, 2))
        _add(line, button, _margin(0, 2, 2, 2))
    _add(parent, line)
    tools = _new(unreal.HorizontalBox, widget, prefix + "Tools")
    shape_text = _text(widget, prefix + "ShapeText", "Shape: %s" % (row.get("shape") or "Auto"), size=11)
    shape_button = _new(unreal.Button, widget, prefix + "Shape")
    shape_button.add_child(shape_text)

    def change_shape(target):
        choice = rwb_world.next_shape(key)
        _status(target, "Rebuilding %s ..." % row["title"])
        with busy("Rebuilding %s" % row["title"]):
            name = rwb_world.set_shape(key, choice)
        shape_text.set_text("Shape: %s" % name)
        _status(target, "%s is now: %s. Press Shape again to try the next one." % (row["title"], name))

    callback = _row_action(widget, change_shape)
    shape_button.on_clicked.add_callable(callback)
    _row_callbacks.append(callback)
    _add(tools, shape_button, _margin(44, 0, 4, 4), fill=True)
    redraw = _button(widget, prefix + "Redraw", "Redraw outline", "sky")
    callback = _row_action(widget, lambda target: open_sketch(target, replace=key))
    redraw.on_clicked.add_callable(callback)
    _row_callbacks.append(callback)
    _add(tools, redraw, _margin(0, 0, 2, 4))
    _add(parent, tools)
    _describe_row(widget, parent, row, prefix, key)


MATCH_TILE = 64.0


def _match_texture(path, index):
    """A match's preview as a texture for the panel (imported once; previews are small)."""
    if not path:
        return None
    stem = "T_match_" + "".join(c if c.isalnum() else "_" for c in Path(path).stem)[:40]
    return rwb_scene._import_texture(path, "/Game/ReferenceWorldBuilder/Found/Previews", stem,
                                     str(Path(path).parent))


def _describe_row(widget, parent, row, prefix, key):
    """Type what it is, Find, then click the picture of the model you want."""
    line = _new(unreal.HorizontalBox, widget, prefix + "Describe")
    box = _new(unreal.EditableTextBox, widget, prefix + "Query")
    try:
        box.set_hint_text("What is it? e.g. birdcage")
    except Exception:
        pass
    search = row.get("search") or {}
    if search.get("query"):
        box.set_text(search["query"])
    _add(line, box, _margin(44, 0, 4, 2), fill=True)
    find = _button(widget, prefix + "Find", "Find model", "sky")

    def run_find(target):
        query = str(box.get_text()).strip()
        if not query:
            _status(target, "Type what the object is, then press Find model.")
            return
        with busy("Finding '%s'" % query):
            result = rwb_world.find_models(key, query)
        notes = ("  (%s)" % "; ".join(result["notes"])) if result.get("notes") else ""
        _status(target, "%d matches for '%s'. Click the one that looks right.%s"
                % (len(result["matches"]), query, notes) if result["matches"] else "Nothing found for '%s'. Try other words.%s" % (query, notes))
        _refresh_model_rows(target)

    callback = _row_action(widget, run_find)
    find.on_clicked.add_callable(callback)
    _row_callbacks.append(callback)
    _add(line, find, _margin(0, 0, 2, 2))
    if row.get("shape") == "Found model":
        undo = _button(widget, prefix + "Undo", "Use painted", "clay")

        def run_undo(target):
            with busy("Going back to the painted model"):
                rwb_world.forget_match(key)
            _status(target, "%s is the model made from your painting again." % row["title"])
            _refresh_model_rows(target)

        callback = _row_action(widget, run_undo)
        undo.on_clicked.add_callable(callback)
        _row_callbacks.append(callback)
        _add(line, undo, _margin(0, 0, 2, 2))
    _add(parent, line)
    matches = search.get("matches") or []
    if not matches:
        return
    strip = _new(unreal.HorizontalBox, widget, prefix + "Matches")
    for index, match in enumerate(matches):
        tile = _new(unreal.Button, widget, "%sMatch%d" % (prefix, index))
        try:
            texture = _match_texture(match.get("thumb"), index)
            frame = _new(unreal.SizeBox, widget, "%sMatchFrame%d" % (prefix, index))
            frame.set_width_override(MATCH_TILE)
            frame.set_height_override(MATCH_TILE)
            if texture:
                image = _new(unreal.Image, widget, "%sMatchImage%d" % (prefix, index))
                image.set_brush_from_texture(texture, False)
                frame.add_child(image)
            else:
                frame.add_child(_text(widget, "%sMatchName%d" % (prefix, index), match.get("name", "?")[:14], size=9))
            tile.add_child(frame)
            tile.set_tool_tip_text("%s - %s (%s)" % (match.get("name"), match.get("author") or match.get("source"), match.get("license")))
        except Exception as error:
            unreal.log_warning("Reference World Builder: no preview for %s (%s)" % (match.get("name"), error))

        def run_use(target, index=index, match=match):
            own = _checked(target, "OwnLookBox", False)
            with busy("Fitting '%s' to your painting" % match.get("name")):
                rwb_world.use_match(key, index, own)
            _status(target, "%s is now '%s' (%s%s)." % (row["title"], match.get("name"), match.get("license"),
                                                        ", credited in ModelLibrary/_online/CREDITS.txt" if match.get("license", "").startswith("CC-BY") else ""))
            _refresh_model_rows(target)

        callback = _row_action(widget, run_use)
        tile.on_clicked.add_callable(callback)
        _row_callbacks.append(callback)
        _add(strip, tile, _margin(44 if index == 0 else 2, 0, 2, 4))
    _add(parent, strip)


def _refresh_model_rows(widget):
    """The list of generated models.  Row widgets need new names every time: old ones linger."""
    global _row_generation
    parent = _maybe(widget, "ModelList")
    if not parent:
        return
    parent.clear_children()
    _retired_callbacks.extend(_row_callbacks)
    del _row_callbacks[:]
    _row_readouts.clear()
    _row_generation += 1
    for index, row in enumerate(rwb_world.model_rows()):
        _model_row(widget, parent, row, _row_generation, index)


# --------------------------------------------------------------------------- #
# Shape Sketch window
# --------------------------------------------------------------------------- #
DRAW_SCRIPT = ROOT_DIR / "reference_draw.py"
_draw = {"process": None, "cache": None, "next_check": 0.0, "busy": False}


def open_sketch(widget, replace=None):
    """Open the drawing window.  It runs alongside Unreal, which watches for what it sends.

    replace: a model's key.  The window starts from that model's outline and its result replaces it.
    """
    if not rwb_world._world:
        _status(widget, "Generate the world first: drawn shapes go into it.")
        return
    process = _draw.get("process")
    if process is not None and process.poll() is None:
        _status(widget, "The Shape Sketch window is already open.")
        return
    cache = rwb_world._world["cache_dir"]
    environment = os.environ.copy()
    environment.pop("SDL_VIDEODRIVER", None)
    environment["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    _draw["cache"] = cache
    command = [_system_python(), str(DRAW_SCRIPT), "--scene", str(Path(cache).resolve()),
               "--out", str(rwb_world.drawn_dir(cache))]
    if replace:
        fix = rwb_world.redraw_args(replace)
        command += ["--replace", fix["key"], "--start-cutout", fix["cutout"],
                    "--start-bbox", ",".join(str(int(v)) for v in fix["bbox"])]
    _draw["process"] = subprocess.Popen(
        command,
        cwd=str(ROOT_DIR), env=environment, creationflags=0x08000000 if os.name == "nt" else 0)
    _ensure_tick()
    _status(widget, ("Shape Sketch is open with that model's outline. Fix it, then press Send to Unreal."
                     if replace else "Shape Sketch is open. Draw on your painting, then press Send to Unreal."))


def poll_sketch():
    """Called from the tick: build whatever the window has sent."""
    # never while a build is running: the progress window keeps this tick alive during one
    if (_draw["cache"] is None or _draw["busy"] or _actions["running"] or _progress["task"] is not None
            or time.time() < _draw["next_check"]):
        return
    _draw["next_check"] = time.time() + 0.5
    jobs = rwb_world.poll_outbox(_draw["cache"])
    if not jobs:
        return
    widget = _current_widget()
    _draw["busy"] = True
    try:
        for job in jobs:
            if widget:
                _status(widget, "Building %s ..." % job.get("label"))
            with busy("Building %s" % job.get("label")):
                label = rwb_world.add_drawn(job)
            if widget:
                _status(widget, "%s is in the world, where you drew it." % label)
        if widget:
            _refresh_model_rows(widget)
    except Cancelled:
        if widget:
            _status(widget, "Cancelled the drawn shape.")
    except Exception as error:
        import traceback
        unreal.log_error(traceback.format_exc())
        if widget:
            _status(widget, "Could not build the drawn shape: %s" % error)
    finally:
        _draw["busy"] = False


def _build_world_from_scene(widget, analysis, values):
    """Single image: depth + perspective reconstruction (reference_scene.py)."""
    result = rwb_scene.build_world(analysis, values, SCENE_SCRIPT)
    _state["accent"] = analysis["mood"].get("accent", "#FFD27A")
    _overlay_material(_state["accent"])
    start = _maybe_start()
    location = start.get_actor_location() if start else unreal.Vector(0, 0, 400)
    _reaction_light(unreal.Vector(location.x, location.y, location.z + 60.0), _state["accent"])
    if values["hide_template"]:
        _hide_template()
    _state["values"] = values
    _state["reactions"] = values["reactions"]
    info = _maybe(widget, "LayerInfo")
    if info:
        info.set_text(result.get("info", "") + "\n" + "\n".join(result["report"][:20]))
    _status(widget, "Built %d actors from %s in perspective. Press Play to walk through it."
            % (result["count"], analysis.get("name", "reference")))


def _maybe_start():
    for actor in _all_actors():
        if _has_tag(actor, PLAYER_TAG) or isinstance(actor, unreal.PlayerStart):
            return actor
    return None


# --------------------------------------------------------------------------- #
# proximity reactions in Play mode
# --------------------------------------------------------------------------- #
def _pie_props(world):
    if _pie["world"] is not None and _pie["world"] == world:
        return _pie["props"]
    actors = []
    for tag in (PROP_TAG,):
        try:
            actors = unreal.GameplayStatics.get_all_actors_with_tag(world, unreal.Name(tag))
        except Exception:
            actors = []
        if actors:
            break
    props = []
    for actor in actors:
        label = actor.get_actor_label()
        base = _state["base_scales"].get(label, (1.0, 1.0, 1.0))
        component = actor.get_component_by_class(unreal.StaticMeshComponent)
        if component:
            props.append((actor, label, base, component))
    light = None
    try:
        lights = unreal.GameplayStatics.get_all_actors_with_tag(world, unreal.Name(REACTION_TAG))
        if lights:
            light = lights[0]
    except Exception:
        pass
    _pie["world"] = world
    _pie["props"] = props
    _pie["light"] = light
    _pie["near"] = set()
    return props


def _reset_prop(entry):
    actor, label, base, component = entry
    if base:
        try:
            actor.set_actor_scale3d(unreal.Vector(base[0], base[1], base[2]))
        except Exception:
            pass
    try:
        component.set_overlay_material(None)
    except Exception:
        pass


def _pie_tick(_delta):
    try:
        _run_queued_actions()
    except Exception as error:
        unreal.log_error("Reference World Builder: %s" % error)
    try:
        poll_sketch()
    except Exception as error:
        unreal.log_warning("Reference World Builder: sketch watcher: %s" % error)
    if _is_commandlet() or not _state.get("reactions", True):
        return
    try:
        subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        if not subsystem.is_in_play_in_editor():
            _pie["pawn"] = None
            _pie["walk_speed"] = None
            if _pie["world"] is not None:
                _pie["world"] = None
                _pie["props"] = []
                _pie["near"] = set()
                _pie["light"] = None
            return
        world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
        if not world:
            return
        pawn = unreal.GameplayStatics.get_player_pawn(world, 0)
        if not pawn:
            return
        _wade(pawn)
        props = _pie_props(world)
        if not props:
            return
        here = pawn.get_actor_location()
        values = _state.get("values") or {}
        radius = max(120.0, float(values.get("radius", 500.0)))
        overlay = _materials.get("overlay") or _overlay_material(_state["accent"])
        now = time.time()
        nearest = radius
        for entry in props:
            actor, label, base, component = entry
            there = actor.get_actor_location()
            distance = math.sqrt(
                (there.x - here.x) ** 2 + (there.y - here.y) ** 2 + (there.z - here.z) ** 2
            )
            nearest = min(nearest, distance)
            if distance < radius:
                if label not in _pie["near"]:
                    _pie["near"].add(label)
                    try:
                        component.set_overlay_material(overlay)
                    except Exception:
                        pass
                if base:
                    pulse = 1.0 + 0.055 * math.sin(now * 3.4 + (hash(label) % 628) * 0.01)
                    actor.set_actor_scale3d(unreal.Vector(base[0] * pulse, base[1] * pulse, base[2] * pulse))
            elif label in _pie["near"]:
                _pie["near"].discard(label)
                _reset_prop(entry)

        light = _pie["light"]
        if light:
            light.set_actor_location(unreal.Vector(here.x, here.y, here.z + 90.0), False, True)
            component = light.get_component_by_class(unreal.PointLightComponent)
            if component:
                fade = max(0.0, 1.0 - nearest / radius) if nearest < radius else 0.0
                pulse = 0.85 + 0.15 * math.sin(now * 5.0)
                component.set_editor_property("intensity", 1800.0 * fade * pulse)
                component.set_light_color(_linear(_state["accent"]))
    except Exception as error:
        now = time.time()
        if now - _state.get("tick_error_at", 0.0) > 6.0:
            _state["tick_error_at"] = now
            unreal.log_warning("Reference World Builder reaction tick skipped: {}".format(error))


WADING_SPEED = 0.45             # share of walking speed while in water


def wading_speed(walk_speed, feet_z, water_z, height=180.0):
    """Slower the deeper you are: ankle-deep hardly matters, chest-deep is a slog."""
    if water_z is None or feet_z >= water_z:
        return walk_speed
    depth = min(1.0, (water_z - feet_z) / (height * 0.6))
    return walk_speed * (1.0 - (1.0 - WADING_SPEED) * depth)


def _wade(pawn):
    """Water slows you down: it's something you go into now, not a floor."""
    movement = pawn.get_component_by_class(unreal.CharacterMovementComponent)
    if not movement:
        return
    if _pie.get("walk_speed") is None or _pie.get("pawn") != pawn:
        _pie["pawn"] = pawn
        _pie["walk_speed"] = movement.get_editor_property("max_walk_speed")
    origin, extent = pawn.get_actor_bounds(True)
    feet = origin.z - extent.z
    movement.set_editor_property("max_walk_speed", wading_speed(_pie["walk_speed"], feet, _state.get("water_z")))


def _ensure_tick():
    global _tick_handle
    if _tick_handle is None:
        _tick_handle = unreal.register_slate_post_tick_callback(_pie_tick)


# --------------------------------------------------------------------------- #
# panel UI
# --------------------------------------------------------------------------- #
def _clean_host_graph(asset):
    try:
        widget = unreal.get_editor_subsystem(unreal.EditorUtilitySubsystem).find_utility_widget_from_blueprint(asset)
    except Exception:
        widget = None
    if not widget:
        return
    for name in (
        "HorizontalBox_PresetButtons", "HorizontalBox_MoveDelete", "HorizontalBox_ErosionKeys",
        "Button_AddErosionKey", "Button_Simulate", "Button_Reset", "Text_Iterations", "TextBlock_13",
        "Button_LoadSettings", "Button_SaveToDataAsset", "Button_NewDataAsset",
        "Button_DataAssetCategory", "Button_Delete", "Button_SwapBack", "Button_SwapFwd",
    ):
        old = widget.find_child_widget_by_name(name)
        if old:
            old.set_visibility(unreal.SlateVisibility.COLLAPSED)


def _build_runtime_ui(widget, force=False):
    if widget.find_child_widget_by_name("RWBTitle") and widget.find_child_widget_by_name("OwnLookBox") and not force:
        return
    if widget.find_child_widget_by_name("RWBTitle"):
        force = True                    # an older panel saved in the asset: rebuild it with the new controls
    global _callbacks, _bound_widget
    if force:
        _retired_callbacks.extend(_callbacks.values() if isinstance(_callbacks, dict) else [])
        _callbacks = {}
        _bound_widget = None
    panel = _find(widget, HOST_PANEL)
    panel.clear_children()
    _clean_host_graph(_widget_asset)

    scroll = _new(unreal.ScrollBox, widget, "RWBScroll")
    _add(panel, scroll, fill=True)
    content = _new(unreal.VerticalBox, widget, "RWBContent")
    scroll.add_child(content)

    header = _new(unreal.Border, widget, "RWBHeader")
    try:
        header.set_brush_color(unreal.LinearColor(0.055, 0.16, 0.18, 1.0))
        header.set_padding(_margin(12, 10, 12, 10))
    except Exception:
        pass
    header_box = _new(unreal.VerticalBox, widget, "RWBHeaderContent")
    header.add_child(header_box)
    _add(header_box, _text(widget, "RWBTitle", "REFERENCE WORLD BUILDER", True, unreal.LinearColor(0.80, 0.97, 0.92, 1.0), 22))
    _add(
        header_box,
        _text(widget, "RWBSubtitle", "A drawing becomes a place: palette -> mood, light -> ground, colour -> life.",
              False, unreal.LinearColor(0.57, 0.78, 0.74, 1.0), 11),
        _margin(0, 4, 0, 0),
    )
    _add(content, header, _margin(8, 8, 8, 4))

    reference = _section(
        widget, content, "Reference", "REFERENCE",
        "PNG/JPG or a Photoshop PSD/PSB. The helper reads the image with the system Python.",
    )
    row = _new(unreal.HorizontalBox, widget, "PathRow")
    _add(row, _text(widget, "PathLabel", "Image", size=12), _margin(8, 3, 6, 3))
    field = _new(unreal.EditableTextBox, widget, "ReferencePathBox")
    field.set_text("")
    _add(row, field, _margin(0, 3, 6, 3), fill=True)
    browse = _button(widget, "BrowseButton", "Browse...", "sky")
    _add(row, browse, _margin(0, 3, 4, 3))
    _add(reference, row)
    actions = _new(unreal.HorizontalBox, widget, "ReferenceActions")
    _add(actions, _button(widget, "AnalyzeButton", "Analyze reference", "forest"), _margin(8, 5, 4, 5))
    _add(reference, actions)
    swatch_row = _new(unreal.HorizontalBox, widget, "PaletteRow")
    for index in range(6):
        swatch = _new(unreal.Border, widget, "PaletteSwatch%d" % index)
        try:
            swatch.set_brush_color(unreal.LinearColor(0.09, 0.12, 0.14, 1.0))
        except Exception:
            pass
        _add(swatch_row, swatch, _margin(8, 4, 2, 4), fill=True)
    _add(reference, swatch_row, _margin(0, 2, 6, 2))
    _add(reference, _text(widget, "ReferenceInfo", "No reference analyzed yet.", False, unreal.LinearColor(0.56, 0.70, 0.68, 1.0), 10), _margin(8, 2, 8, 4))
    _add(reference, _text(widget, "LayerInfo", "PSD layers: name them after what they are (Castle, Oak tree, Thorns, Lake, Sky, Aurora...).",
                          False, unreal.LinearColor(0.70, 0.82, 0.62, 1.0), 10), _margin(8, 2, 8, 4))

    build = _section(
        widget, content, "Build", "BUILD",
        "Brightness becomes terrain height. Colour regions decide what grows where.",
    )
    setting_row = _new(unreal.HorizontalBox, widget, "SettingRow")
    _add(setting_row, _button(widget, "SettingButton", _setting_label(), "sky"), _margin(8, 4, 8, 4))
    _add(setting_row, _text(widget, "SettingHint", "Click to change the kind of place, then Analyze again.", size=10),
         _margin(0, 8, 8, 4), fill=True)
    _add(build, setting_row)
    _slider_row(widget, build, "WorldSize", "World size", 4000.0, 2000.0, 6000.0)
    _slider_row(widget, build, "Height", "Terrain height", 560.0, 120.0, 1200.0)
    _slider_row(widget, build, "Density", "Prop density", 0.55, 0.05, 1.0, True)
    _field_row(widget, build, "Seed", "Seed", "20261104")
    _check_row(widget, build, "Water", "Water where the picture has water (checked before it's made)", True)
    _check_row(widget, build, "HideTemplate", "Hide the template floor and props", True)
    _check_row(widget, build, "AutoGenerate", "Generate the world right after analyzing", True)
    _check_row(widget, build, "GenerateModels", "Create new 3D models from the painting (refined in Blender if installed)", True)
    _check_row(widget, build, "ExtraCopies", "Add extra copies of repeated things outside the painted view", True)
    _check_row(widget, build, "ExactLayout", "Recreate the painting's exact layout instead (older modes below)", False)
    _check_row(widget, build, "UseLayers", "PSD: build objects from layer names", True)
    _check_row(widget, build, "Perspective", "Single image: rebuild depth + perspective, project the painting", True)
    _slider_row(widget, build, "Fov", "Painting lens (FOV)", 60.0, 30.0, 100.0)
    _check_row(widget, build, "PaintArt", "PSD: paint each layer's artwork onto its object", True)
    _slider_row(widget, build, "ArtBlend", "Art blend", 0.8, 0.0, 1.0, True)
    _slider_row(widget, build, "ObjectScale", "Object size", 1.0, 0.3, 2.5, True)
    _slider_row(widget, build, "StrokeStrength", "Brushstrokes", 1.0, 0.0, 2.5, True)
    _slider_row(widget, build, "ModelDetail", "Model style: low poly (0) to as painted (1)", 1.0, 0.0, 1.0, True)
    _check_row(widget, build, "GrowGrass", "Grow grass on grassy ground", True)
    build_actions = _new(unreal.HorizontalBox, widget, "BuildActions")
    _add(build_actions, _button(widget, "GenerateButton", "Generate world", "forest"), _margin(8, 6, 4, 6))
    _add(build_actions, _button(widget, "ClearButton", "Clear", "clay"), _margin(0, 6, 8, 6))
    _add(build, build_actions)

    models = _section(
        widget, content, "Models", "MODELS",
        "How many of each model. Click a name to select it in the level; 0 removes it. Changes apply straight away.",
    )
    _check_row(widget, models, "KeepCopiesOut", "Keep added copies out of the painted view", False)
    _check_row(widget, models, "OwnLook", "Found models keep their own colours (instead of your painting's)", False)
    _add(models, _button(widget, "DrawButton", "Draw a new shape...", "sky"), _margin(8, 4, 8, 6))
    _add(models, _new(unreal.VerticalBox, widget, "ModelList"))

    play = _section(
        widget, content, "Play", "PLAY",
        "In Play mode, props near the character glow and breathe. A soft light follows you.",
    )
    _slider_row(widget, play, "Radius", "Reaction radius", 520.0, 120.0, 1400.0)
    _check_row(widget, play, "Reactions", "Enable proximity reactions", True)
    play_actions = _new(unreal.HorizontalBox, widget, "PlayActions")
    _add(play_actions, _button(widget, "PlayButton", "Play in editor", "sky"), _margin(8, 6, 4, 6))
    _add(play_actions, _button(widget, "StopButton", "Stop", "slate"), _margin(0, 6, 8, 6))
    _add(play, play_actions)
    _add(
        play,
        _text(widget, "PlayNote", "Tip: drag the reference into the path field, or use Browse. Build again after changing sliders.",
              False, unreal.LinearColor(0.52, 0.67, 0.64, 1.0), 10),
        _margin(8, 2, 8, 4),
    )

    _add(content, _text(widget, "StatusText", "Ready. Choose a reference image to begin.", False, unreal.LinearColor(0.62, 0.82, 0.76, 1.0), 11), _margin(12, 6, 12, 12))
    _status(widget, "Ready. Choose a reference image to begin.")
    bind_widget(widget)


_BUTTON_ACTIONS = {}  # filled in below, once the action functions exist


def _current_widget(fallback=None):
    try:
        if _widget_asset:
            live = unreal.get_editor_subsystem(unreal.EditorUtilitySubsystem).find_utility_widget_from_blueprint(_widget_asset)
            if live and live.find_child_widget_by_name("RWBTitle"):
                return live
    except Exception:
        pass
    return fallback


# A click must not do its work inside the click.  The button's delegate is still running then, and a
# build rebuilds rows, lets Slate tick (the progress window) and triggers garbage collection: if the
# Python handler is released while it is on the stack, Unreal crashes with an access violation in the
# Python plugin.  So a click only queues the work, and the next tick runs it with no delegate on the stack.
_actions = {"queue": [], "running": False}


def _queue_action(key, run):
    """Ask for ``run`` to happen on the next tick.  Double-clicks and clicks during a build do nothing."""
    if _actions["running"] or any(item[0] == key for item in _actions["queue"]):
        return False
    _actions["queue"].append((key, run))
    _ensure_tick()
    return True


def _run_queued_actions():
    """Called from the tick.  The progress window ticks during a job, so never start one inside another."""
    if _actions["running"] or not _actions["queue"]:
        return
    _key, run = _actions["queue"].pop(0)
    _actions["running"] = True
    try:
        run()
    finally:
        _actions["running"] = False


def _make_click_callback(name, widget):
    def run():
        target = _current_widget(widget)
        unreal.log("Reference World Builder: %s clicked" % name)
        action = _BUTTON_ACTIONS.get(name)
        if not action:
            unreal.log_error("Reference World Builder: no action for " + name)
            return
        try:
            action(target)
        except Exception as error:
            import traceback
            unreal.log_error(traceback.format_exc())
            _status(target, "%s failed: %s" % (name.replace("Button", ""), error))

    def callback():
        _queue_action(name, run)
    return callback


def _make_slider_callback(widget):
    def callback(_value):
        _refresh_readouts(widget)
    return callback


_retired_callbacks = []         # replaced handlers: kept alive, because their delegates may still fire


def bind_widget(widget):
    global _callbacks, _bound_widget
    if not isinstance(_callbacks, dict):
        _callbacks = {}
    if _bound_widget is None or _bound_widget != widget:
        _retired_callbacks.extend(_callbacks.values())
        _callbacks = {}
    rebound = []
    for name in ("BrowseButton", "AnalyzeButton", "GenerateButton", "ClearButton", "PlayButton", "StopButton", "DrawButton",
                 "SettingButton"):
        button = _maybe(widget, name)
        if not button:
            continue
        callback = _callbacks.get(name)
        try:
            bound = callback is not None and button.on_clicked.contains_callable(callback)
        except Exception:
            bound = callback is not None
        if bound:
            continue
        callback = _make_click_callback(name, widget)
        button.on_clicked.add_callable(callback)
        _callbacks[name] = callback
        rebound.append(name)
    for key in ("WorldSize", "Height", "Density", "Radius", "ArtBlend", "ObjectScale", "Fov", "StrokeStrength", "ModelDetail"):
        slider = _maybe(widget, key + "Slider")
        name = key + "Slider"
        if not slider:
            continue
        callback = _callbacks.get(name)
        try:
            bound = callback is not None and slider.on_value_changed.contains_callable(callback)
        except Exception:
            bound = callback is not None
        if not bound:
            callback = _make_slider_callback(widget)
            slider.on_value_changed.add_callable(callback)
            _callbacks[name] = callback
    if rebound and _bound_widget is not None and _bound_widget == widget:
        unreal.log_warning("Reference World Builder: reconnected buttons " + ", ".join(rebound))
    _refresh_readouts(widget)
    _bound_widget = widget


# --------------------------------------------------------------------------- #
# UI actions
# --------------------------------------------------------------------------- #
def _read_values(widget):
    try:
        seed = int(str(_find(widget, "SeedBox").get_text()).strip())
    except Exception:
        seed = 20261104
    return {
        "world_size": _find(widget, "WorldSizeSlider").get_value(),
        "height": _find(widget, "HeightSlider").get_value(),
        "density": _find(widget, "DensitySlider").get_value(),
        "radius": _find(widget, "RadiusSlider").get_value(),
        "seed": seed,
        "water": _find(widget, "WaterBox").is_checked(),
        "hide_template": _find(widget, "HideTemplateBox").is_checked(),
        "reactions": _find(widget, "ReactionsBox").is_checked(),
        "use_layers": _checked(widget, "UseLayersBox", True),
        "paint_art": _checked(widget, "PaintArtBox", True),
        "art_blend": _slider_value(widget, "ArtBlendSlider", 0.8),
        "object_scale": _slider_value(widget, "ObjectScaleSlider", 1.0),
        "perspective": _checked(widget, "PerspectiveBox", True),
        "exact_layout": _checked(widget, "ExactLayoutBox", False),
        "generate_models": _checked(widget, "GenerateModelsBox", True),
        "extra_copies": _checked(widget, "ExtraCopiesBox", True),
        "keep_copies_out": _checked(widget, "KeepCopiesOutBox", False),
        "stroke_strength": _slider_value(widget, "StrokeStrengthSlider", 1.0),
        "model_detail": _slider_value(widget, "ModelDetailSlider", 1.0),
        "grow_grass": _checked(widget, "GrowGrassBox", True),
        "auto_generate": _checked(widget, "AutoGenerateBox", True),
        "fov": _slider_value(widget, "FovSlider", 60.0),
        "setting": _setting["choice"],
    }


# the kind of place: "auto" reads it from the picture (reference_setting)
SETTING_CHOICES = ["auto", "nature", "city", "interior", "desert", "snow", "coast"]
_setting = {"choice": "auto"}


def _setting_label():
    choice = _setting["choice"]
    return "Setting: Auto (from the picture)" if choice == "auto" else "Setting: " + choice.capitalize()


def cycle_setting(widget):
    """Next kind of place.  It decides what the next analysis looks for and how the world is built."""
    index = SETTING_CHOICES.index(_setting["choice"]) if _setting["choice"] in SETTING_CHOICES else 0
    _setting["choice"] = SETTING_CHOICES[(index + 1) % len(SETTING_CHOICES)]
    label = _maybe(widget, "SettingButtonLabel")
    if label:
        label.set_text(_setting_label())
    _status(widget, "%s. Press Analyze reference to read the picture again as this kind of place."
            % _setting_label())


def _checked(widget, name, default):
    item = _maybe(widget, name)
    return item.is_checked() if item else default


def _slider_value(widget, name, default):
    item = _maybe(widget, name)
    return item.get_value() if item else default


def browse_reference(widget):
    try:
        path = pick_reference()
    except Exception as error:
        _status(widget, "File picker failed: " + str(error))
        return
    if not path:
        _status(widget, "No file chosen.")
        return
    _find(widget, "ReferencePathBox").set_text(path)
    _status(widget, "Selected " + os.path.basename(path) + ". Analyzing...")
    analyze_clicked(widget)


def analyze_clicked(widget):
    path = str(_find(widget, "ReferencePathBox").get_text()).strip().strip('"')
    if not path:
        _status(widget, "Choose a reference image first.")
        return
    _status(widget, "Analyzing " + os.path.basename(path) + " ...")
    try:
        with busy("Reading " + os.path.basename(path)):
            data = analyze_reference(path)
    except Cancelled:
        _status(widget, "Analysis cancelled.")
        return
    except Exception as error:
        _status(widget, "Analysis failed: " + str(error))
        return
    _state["analysis"] = data
    _state["path"] = path
    _state["accent"] = data["mood"].get("accent", "#FFD27A")
    _show_palette(widget, data)
    _find(widget, "ReferenceInfo").set_text(
        "%dx%d  |  %s  |  luminance %.2f  saturation %.2f  |  water %.2f"
        % (
            data["width"], data["height"], data["mood"]["name"],
            data["mood"]["average_luminance"], data["mood"]["average_saturation"],
            data["water_level"],
        )
    )
    info = _maybe(widget, "LayerInfo")
    layers = data.get("layers")
    if info:
        if layers:
            lines = []
            for layer in layers[:30]:
                n = len(layer.get("instances") or [])
                kind = layer.get("kind")
                what = {"object": "%s x%d" % (layer.get("category"), max(1, n)),
                        "card": "painted card" + (" x%d" % n if n > 1 else ""),
                        "scatter": "scattered " + str(layer.get("category")),
                        "ignore": "skipped: " + str(layer.get("reason", ""))}.get(kind, str(kind))
                tag = {"fg": "  [FG - near]", "mg": "  [MG - middle]", "bg": "  [BG - far]"}.get(layer.get("depth"), "")
                lines.append("%s  ->  %s%s" % (layer.get("name"), what, tag))
            if len(layers) > 30:
                lines.append("... and %d more (see Output Log)" % (len(layers) - 30))
            info.set_text("\n".join(lines))
            for line in lines:
                unreal.log("Reference World Builder layer: " + line)
        else:
            scene = data.get("scene") or {}
            if scene.get("error"):
                info.set_text("Flat image. Perspective setup failed: %s" % scene["error"])
            elif scene.get("depth"):
                info.set_text("Flat image -> depth (%s), horizon at %d%% of the picture. Generate rebuilds it in perspective."
                              % (scene["depth"], round(100.0 * scene["horizon"] / max(1, scene["height"]))))
            else:
                info.set_text("Flat image -> perspective only (no depth: %s). Horizon at %d%%."
                              % (scene.get("depth_error"), round(100.0 * scene.get("horizon", 0) / max(1, scene.get("height", 1)))))
    inventory = data.get("inventory") or {}
    place = inventory.get("setting") or {}
    if place.get("line"):
        night = "  |  night: lit signs become lights" if place.get("night") else ""
        _find(widget, "ReferenceInfo").set_text(str(_find(widget, "ReferenceInfo").get_text()) + "\n" + place["line"] + night)
        unreal.log("Reference World Builder: " + place["line"] + night)
    values = _read_values(widget)
    if info and inventory and not values.get("exact_layout"):
        if inventory.get("error"):
            info.set_text("Could not list the objects: %s" % inventory["error"])
        else:
            lines = ["Objects found by %s:" % inventory.get("info", "?")]
            for entry in inventory.get("objects", []):
                name = entry.get("label") if entry.get("individual") else entry.get("category")
                lines.append("  %s: %s" % (name, "seen once" if entry.get("unique")
                                            else "seen %d times" % entry.get("count", 0)))
            if (inventory.get("water") or {}).get("present"):
                lines.append("  water")
            info.set_text("\n".join(lines))
            for line in lines:
                unreal.log("Reference World Builder: " + line)
    if values.get("auto_generate"):
        _status(widget, "Analyzed %s. Generating the world ..." % data.get("name", ""))
        generate(widget)
        return
    if layers:
        _status(widget, "Analyzed %d layers of %s. Press Generate world." % (len(layers), data.get("name", "")))
    else:
        _status(widget, "Analyzed %s. Press Generate world." % data.get("name", os.path.basename(path)))


def _show_palette(widget, data):
    palette = data.get("palette") or []
    for index in range(6):
        swatch = _maybe(widget, "PaletteSwatch%d" % index)
        if not swatch:
            continue
        if index < len(palette):
            swatch.set_brush_color(_linear(palette[index]["hex"]))
        else:
            swatch.set_brush_color(unreal.LinearColor(0.09, 0.12, 0.14, 1.0))


def generate(widget):
    unreal.log("Reference World Builder: Generate world started")
    if not _state.get("analysis"):
        path = str(_find(widget, "ReferencePathBox").get_text()).strip().strip('"')
        if not path:
            _status(widget, "Choose and analyze a reference image first.")
            return
        try:
            with busy("Reading " + os.path.basename(path)):
                _state["analysis"] = analyze_reference(path)
            _state["path"] = path
            _state["accent"] = _state["analysis"]["mood"].get("accent", "#FFD27A")
            _show_palette(widget, _state["analysis"])
        except Exception as error:
            _status(widget, "Analysis failed: " + str(error))
            return
    _status(widget, "Clearing the previous world ...")
    rwb_world.forget()
    _refresh_model_rows(widget)
    _clear_generated()
    _status(widget, "Building terrain, water and props ...")
    try:
        with busy("Building your world"):
            _build_world(widget)
    except Cancelled:
        _status(widget, "Build cancelled. Press Generate world to try again.")
    except Exception as error:
        import traceback
        unreal.log_error(traceback.format_exc())
        _status(widget, "Build failed: " + str(error))


def clear_world(widget):
    rwb_world.forget()
    _refresh_model_rows(widget)
    _clear_generated()
    _status(widget, "World cleared. The template environment is visible again.")


def play(widget):
    try:
        _state["reactions"] = _find(widget, "ReactionsBox").is_checked()
        _ensure_tick()
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_request_begin_play()
        _status(widget, "Playing. Walk near a prop to make it react.")
    except Exception as error:
        _status(widget, "Could not start Play: " + str(error))


def stop(widget):
    try:
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_request_end_play()
        _status(widget, "Stopped.")
    except Exception as error:
        _status(widget, "Could not stop Play: " + str(error))


_BUTTON_ACTIONS.update({
    "BrowseButton": browse_reference,
    "AnalyzeButton": analyze_clicked,
    "GenerateButton": generate,
    "ClearButton": clear_world,
    "PlayButton": play,
    "StopButton": stop,
    "DrawButton": open_sketch,
    "SettingButton": cycle_setting,
})


def generate_from_menu():
    widget = _current_widget()
    if not widget:
        widget = open_tool()
    unreal.log("Reference World Builder: Generate world (menu)")
    generate(widget)


# --------------------------------------------------------------------------- #
# widget asset + menu
# --------------------------------------------------------------------------- #
def build_widget_asset():
    if unreal.EditorAssetLibrary.does_asset_exist(ASSET_PATH):
        return unreal.load_asset(ASSET_PATH)
    if unreal.EditorAssetLibrary.does_asset_exist(OLD_EUW_PATH):
        if unreal.EditorAssetLibrary.rename_asset(OLD_EUW_PATH, ASSET_PATH):
            return unreal.load_asset(ASSET_PATH)
    if not unreal.load_asset(SOURCE_EUW_PATH):
        raise RuntimeError("Landmass must be enabled to create the Reference World Builder panel")
    if not unreal.EditorAssetLibrary.duplicate_asset(SOURCE_EUW_PATH, ASSET_PATH):
        raise RuntimeError("Could not create the Reference World Builder panel")
    return unreal.load_asset(ASSET_PATH)


def open_tool():
    global _widget_asset
    asset = build_widget_asset()
    _widget_asset = asset
    subsystem = unreal.get_editor_subsystem(unreal.EditorUtilitySubsystem)
    widget = subsystem.find_utility_widget_from_blueprint(asset)
    if not widget:
        widget = subsystem.spawn_and_register_tab(asset)
    _build_runtime_ui(widget)
    bind_widget(widget)
    _status(widget, "Ready. Choose a reference image to begin.")
    return widget


def reload_tool():
    global _widget_asset
    asset = build_widget_asset()
    _widget_asset = asset
    subsystem = unreal.get_editor_subsystem(unreal.EditorUtilitySubsystem)
    widget = subsystem.find_utility_widget_from_blueprint(asset)
    if not widget:
        widget = subsystem.spawn_and_register_tab(asset)
    _build_runtime_ui(widget, force=True)
    bind_widget(widget)
    _status(widget, "Panel reloaded.")
    return widget


@unreal.uclass()
class ReferenceWorldBuilderMenuEntry(unreal.ToolMenuEntryScript):
    @unreal.ufunction(override=True)
    def get_label(self, context):
        return "Open Reference World Builder"

    @unreal.ufunction(override=True)
    def get_tool_tip(self, context):
        return "Turn a concept image into a walkable Unreal environment."

    @unreal.ufunction(override=True)
    def execute(self, context):
        open_tool()


@unreal.uclass()
class ReferenceWorldBuilderGenerateEntry(unreal.ToolMenuEntryScript):
    @unreal.ufunction(override=True)
    def get_label(self, context):
        return "Reference World Builder: Generate world"

    @unreal.ufunction(override=True)
    def get_tool_tip(self, context):
        return "Build the world from the reference image analyzed in the panel."

    @unreal.ufunction(override=True)
    def execute(self, context):
        generate_from_menu()


def register_menu():
    menus = unreal.ToolMenus.get()
    menus.unregister_owner_by_name(MENU_OWNER)
    menu = menus.extend_menu("LevelEditor.MainMenu.Tools")
    entry = ReferenceWorldBuilderMenuEntry()
    entry.init_entry(
        MENU_OWNER,
        "LevelEditor.MainMenu.Tools",
        "ReferenceWorldBuilder",
        "ReferenceWorldBuilder.Open",
        "Open Reference World Builder",
    )
    menu.add_menu_entry_object(entry)
    generate_entry = ReferenceWorldBuilderGenerateEntry()
    generate_entry.init_entry(
        MENU_OWNER,
        "LevelEditor.MainMenu.Tools",
        "ReferenceWorldBuilder",
        "ReferenceWorldBuilder.Generate",
        "Reference World Builder: Generate world",
    )
    menu.add_menu_entry_object(generate_entry)
    menus.refresh_all_widgets()


_watch_state = {"last": 0.0}


def _watch_for_manually_opened_panel(_delta_seconds):
    try:
        if not _widget_asset or _actions["running"] or _progress["task"] is not None:
            return                      # the panel is mid-job: rewiring its buttons now isn't safe
        now = time.time()
        if now - _watch_state["last"] < 1.0:
            return
        _watch_state["last"] = now
        widget = unreal.get_editor_subsystem(unreal.EditorUtilitySubsystem).find_utility_widget_from_blueprint(_widget_asset)
        if not widget:
            return
        if not widget.find_child_widget_by_name("RWBTitle"):
            _build_runtime_ui(widget)
            bind_widget(widget)
            _status(widget, "Ready. Choose a reference image to begin.")
        else:
            bind_widget(widget)
    except Exception as error:
        unreal.log_warning("Reference World Builder panel watch skipped: {}".format(error))


def _is_commandlet():
    text = " ".join(sys.argv)
    try:
        text += " " + unreal.SystemLibrary.get_command_line()
    except Exception:
        pass
    text = text.lower()
    return "-run=" in text or "-unattended" in text or "-nullrhi" in text


def startup():
    global _watch_handle, _widget_asset
    if _is_commandlet():
        return None
    try:
        register_menu()
        _widget_asset = build_widget_asset()
        if _watch_handle is None:
            _watch_handle = unreal.register_slate_post_tick_callback(_watch_for_manually_opened_panel)
        _ensure_tick()
        return open_tool()
    except Exception as error:
        unreal.log_error("Reference World Builder startup failed: {}".format(error))
        return None
