"""Interactive World Builder Editor Utility Widget runtime."""

import json
import math
import os
import random
import zipfile

import unreal


ASSET_PATH = "/Game/WorldBuilder/EUW_WorldBuilderPanel"
SOURCE_EUW_PATH = "/Landmass/Landscape/ErosionBrush/BP/Widgets/EUW_Landscape_Erosion"
HOST_PANEL = "VerticalBox_SettingsPreset"
GENERATED_TAG, PREVIEW_TAG = "IWB_Generated", "IWB_Preview"
TERRAIN_TAG, TREE_TAG, ROCK_TAG, GRASS_TAG = "IWB_Terrain", "IWB_Trees", "IWB_Rocks", "IWB_Grass"
FLOWER_TAG, MOUNTAIN_TAG = "IWB_Flowers", "IWB_Mountains"
PATH_TAG, LIGHT_TAG, PLAYER_TAG = "IWB_Path", "IWB_Lighting", "IWB_PlayerStart"
TEMPLATE_HIDDEN_TAG = "IWB_TemplateHidden"
PRESET_FILENAME = "InteractiveWorldBuilderPreset.json"
KENNEY_ZIP_FILENAME = "kenney_nature-kit.zip"
KENNEY_DESTINATION = "/Game/WorldBuilder/Imported/Kenney"
# Deliberately small selection: enough visual variety for a world, without
# putting the entire 330-asset kit into this class project.
KENNEY_FILES = (
    "tree_detailed", "tree_oak", "tree_pineRoundA", "tree_pineTallB",
    "rock_largeB", "rock_largeE", "rock_smallA", "rock_smallH",
    "grass", "grass_large", "plant_bushDetailed", "plant_bushSmall",
    "flower_purpleA", "flower_redB", "flower_yellowC",
    "cliff_blockSlope_rock", "cliff_large_rock", "cliff_top_rock",
)
MENU_OWNER = "InteractiveWorldBuilderMenu"
_callbacks = []
_bound_widget = None
_watch_handle = None
_widget_asset = None
_tint_material = None
_roboto_font = None


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
    """Use Unreal's shipped Roboto face; a size-only font renders as missing glyphs in this host EUW."""
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


def _button(owner, name, text, tone="soft"):
    item = _new(unreal.Button, owner, name)
    colors = {
        "forest": unreal.LinearColor(0.08, 0.32, 0.20, 1.0),
        "clay": unreal.LinearColor(0.47, 0.25, 0.14, 1.0),
        "soft": unreal.LinearColor(0.18, 0.23, 0.19, 1.0),
    }
    try:
        item.set_background_color(colors.get(tone, colors["soft"]))
    except Exception:
        pass
    item.add_child(_text(owner, name + "Label", text, True, unreal.LinearColor(0.94, 0.94, 0.88, 1.0), 12))
    return item


def _section(owner, parent, name, title, caption=None):
    """A compact card keeps the long tool readable without making it feel like a form."""
    card = _new(unreal.Border, owner, name + "Card")
    try:
        card.set_brush_color(unreal.LinearColor(0.045, 0.075, 0.058, 1.0))
        card.set_padding(_margin(8, 7, 8, 7))
    except Exception:
        pass
    content = _new(unreal.VerticalBox, owner, name + "Content")
    card.add_child(content)
    _add(content, _text(owner, name + "Title", title, True, unreal.LinearColor(0.68, 0.81, 0.67, 1.0), 12), _margin(3, 1, 3, 1))
    if caption:
        _add(content, _text(owner, name + "Caption", caption, False, unreal.LinearColor(0.47, 0.60, 0.49, 1.0), 10), _margin(3, 0, 3, 5))
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
    # A Fill slot gives the Slider a visible rail instead of a tiny isolated handle.
    _add(row, slider, _margin(0, 5, 8, 5), fill=True)
    readout = _text(owner, key + "Readout", f"{value:.2f}" if decimals else str(round(value)))
    readout.set_min_desired_width(48.0)
    _add(row, readout, _margin(0, 3, 8, 3))
    _add(panel, row)


def _clean_host_graph(asset):
    # The duplicated utility supplies a durable UMG root. Its original logic is not needed.
    try:
        unreal.BlueprintEditorLibrary.reparent_blueprint(asset, unreal.EditorUtilityWidget)
        for graph in unreal.BlueprintEditorLibrary.list_graphs(asset):
            unreal.BlueprintEditorLibrary.remove_graph(asset, graph)
        unreal.BlueprintEditorLibrary.compile_blueprint(asset)
        unreal.EditorAssetLibrary.save_loaded_asset(asset, False)
    except Exception as error:
        unreal.log_warning("World Builder host graph cleanup skipped: {}".format(error))


def build_widget_asset():
    if unreal.EditorAssetLibrary.does_asset_exist(ASSET_PATH):
        asset = unreal.load_asset(ASSET_PATH)
        _clean_host_graph(asset)
        return asset
    if not unreal.load_asset(SOURCE_EUW_PATH):
        raise RuntimeError("Landmass must be enabled to create the World Builder panel")
    if not unreal.EditorAssetLibrary.duplicate_asset(SOURCE_EUW_PATH, ASSET_PATH):
        raise RuntimeError("Could not create the World Builder Editor Utility Widget")
    asset = unreal.load_asset(ASSET_PATH)
    _clean_host_graph(asset)
    return asset


def _find(widget, name):
    item = widget.find_child_widget_by_name(name)
    if not item:
        raise RuntimeError("World Builder UI element is missing: " + name)
    return item


def _build_runtime_ui(widget, force=False):
    if widget.find_child_widget_by_name("IWBTitle") and not force:
        return
    global _callbacks, _bound_widget
    if force:
        # Callbacks held by the former dynamic controls are no longer useful.
        _callbacks = []
        _bound_widget = None
    panel = _find(widget, HOST_PANEL)
    panel.clear_children()
    # This asset started as the Landmass erosion utility. Hide every remaining
    # Erosion control so it reads as one deliberate World Builder tool.
    for name in (
        "HorizontalBox_PresetButtons", "HorizontalBox_MoveDelete", "HorizontalBox_ErosionKeys",
        "Button_AddErosionKey", "Button_Simulate", "Button_Reset", "Text_Iterations", "TextBlock_13",
        "Button_LoadSettings", "Button_SaveToDataAsset", "Button_NewDataAsset",
        "Button_DataAssetCategory", "Button_Delete", "Button_SwapBack", "Button_SwapFwd",
    ):
        old = widget.find_child_widget_by_name(name)
        if old:
            old.set_visibility(unreal.SlateVisibility.COLLAPSED)

    cream = unreal.LinearColor(0.93, 0.89, 0.78, 1.0)
    leaf = unreal.LinearColor(0.53, 0.78, 0.48, 1.0)
    muted = unreal.LinearColor(0.47, 0.61, 0.48, 1.0)
    header = _new(unreal.Border, widget, "IWBHeader")
    try:
        header.set_brush_color(unreal.LinearColor(0.12, 0.27, 0.18, 1.0))
        header.set_padding(_margin(12, 10, 12, 10))
    except Exception:
        pass
    header_content = _new(unreal.VerticalBox, widget, "IWBHeaderContent")
    header.add_child(header_content)
    _add(header_content, _text(widget, "IWBTitle", "WORLD BUILDER", True, cream, 23), _margin(0, 0, 0, 0))
    _add(header_content, _text(widget, "IWBSubtitle", "A small, quiet world — grown from a seed.", False, leaf, 11), _margin(0, 0, 0, 0))
    _add(panel, header, _margin(8, 10, 8, 5))

    world = _section(widget, panel, "World", "WORLD", "Name it, choose a mood, then keep the seed for repeatable worlds.")
    _field_row(widget, world, "WorldName", "World name", "Whispering Pines")
    _field_row(widget, world, "Seed", "Seed", "20261018")
    theme_row = _new(unreal.HorizontalBox, widget, "ThemeRow")
    theme_label = _text(widget, "ThemeLabel", "World mood", size=12)
    theme_label.set_min_desired_width(112.0)
    _add(theme_row, theme_label, _margin(8, 3, 4, 3))
    theme = _new(unreal.ComboBoxString, widget, "ThemeBox")
    for option in ("Forest", "Golden Meadow", "Misty Highlands"):
        theme.add_option(option)
    theme.set_selected_option("Forest")
    _add(theme_row, theme, _margin(0, 3, 8, 3), fill=True)
    _add(world, theme_row)

    land = _section(widget, panel, "Land", "LAND", "Shape the clearing first. Mountains stay at the edge so the path remains playable.")
    _slider_row(widget, land, "Terrain", "World scale", 4000.0, 1600.0, 8000.0)
    mountain_row = _new(unreal.HorizontalBox, widget, "MountainRow")
    mountains = _new(unreal.CheckBox, widget, "MountainsBox")
    mountains.set_is_checked(False)
    _add(mountain_row, mountains, _margin(8, 5, 5, 5))
    _add(mountain_row, _text(widget, "MountainsLabel", "Add low-poly mountains at the edges", size=12), _margin(0, 5, 8, 5), fill=True)
    _add(land, mountain_row)

    life = _section(widget, panel, "Life", "LIFE", "Control each layer separately — the terrain is not rebuilt when you refresh vegetation.")
    _slider_row(widget, life, "Trees", "Trees", 45.0, 8.0, 120.0)
    _slider_row(widget, life, "Flowers", "Flowers", 34.0, 0.0, 140.0)
    _slider_row(widget, life, "Rocks", "Rock clusters", 24.0, 0.0, 80.0)
    _slider_row(widget, life, "Grass", "Grass", 120.0, 0.0, 320.0)
    _add(life, _text(widget, "RuleNote", "Trees thin on high ground; rocks gather in clusters; flowers and grass stay clear of the path.", False, muted, 10), _margin(8, 2, 8, 3))

    atmosphere = _section(widget, panel, "Atmosphere", "ATMOSPHERE", "A soft, warm palette inspired by your references.")
    _slider_row(widget, atmosphere, "Lighting", "Sun strength", 8.0, 1.0, 15.0, True)
    _slider_row(widget, atmosphere, "SkyTint", "Sky warmth", 0.55, 0.0, 1.0, True)
    _slider_row(widget, atmosphere, "GroundTint", "Ground richness", 0.32, 0.0, 1.0, True)
    _slider_row(widget, atmosphere, "LightTint", "Sun warmth", 0.12, 0.0, 1.0, True)
    character_row = _new(unreal.HorizontalBox, widget, "CharacterRow")
    character = _new(unreal.CheckBox, widget, "PlayableCharacterBox")
    character.set_is_checked(True)
    _add(character_row, character, _margin(8, 5, 5, 5))
    _add(character_row, _text(widget, "PlayableCharacterLabel", "Keep the Third Person character ready to explore", size=12), _margin(0, 5, 8, 5), fill=True)
    _add(atmosphere, character_row)

    actions = _section(widget, panel, "Actions", "BUILD", "Generate replaces the current World Builder result. Preview leaves it untouched.")
    primary = _new(unreal.HorizontalBox, widget, "PrimaryButtons")
    _add(primary, _button(widget, "GenerateButton", "Generate World", "forest"), _margin(3, 3, 3, 3), fill=True)
    _add(primary, _button(widget, "PreviewButton", "Preview", "soft"), _margin(3, 3, 3, 3), fill=True)
    _add(actions, primary)
    utility = _new(unreal.HorizontalBox, widget, "UtilityButtons")
    _add(utility, _button(widget, "ImportNatureButton", "Import Nature Kit", "forest"), _margin(3, 2, 3, 2), fill=True)
    _add(utility, _button(widget, "UpdateVegetationButton", "Refresh Plants", "soft"), _margin(3, 2, 3, 2), fill=True)
    _add(utility, _button(widget, "StartOverButton", "Start Over", "clay"), _margin(3, 2, 3, 2), fill=True)
    _add(actions, utility)
    presets = _new(unreal.HorizontalBox, widget, "PresetButtons")
    for name, label in (("SavePresetButton", "Save Preset"), ("LoadPresetButton", "Load Preset"), ("ResetButton", "Reset Controls"), ("DeleteButton", "Delete World")):
        _add(presets, _button(widget, name, label), _margin(3, 2, 3, 2), fill=True)
    _add(actions, presets)
    _add(panel, _text(widget, "StatusText", "Ready to grow a forest.", False, cream, 11), _margin(14, 8, 14, 12))
    widget.invalidate_layout_and_volatility()
    widget.force_layout_prepass()


def _status(widget, message):
    _find(widget, "StatusText").set_text(message)
    unreal.log("Interactive World Builder: " + message)


def _refresh_readouts(widget):
    fractional = {"Lighting", "SkyTint", "GroundTint", "LightTint"}
    for key in ("Terrain", "Trees", "Flowers", "Rocks", "Grass", "Lighting", "SkyTint", "GroundTint", "LightTint"):
        value = _find(widget, key + "Slider").get_value()
        _find(widget, key + "Readout").set_text(f"{value:.2f}" if key in fractional else str(round(value)))


def _values(widget):
    _refresh_readouts(widget)
    try:
        seed = int(str(_find(widget, "SeedBox").get_text()).strip())
    except ValueError:
        seed = 20261018
    return {
        "world_name": str(_find(widget, "WorldNameBox").get_text()).strip() or "Untitled Forest",
        "seed": seed, "theme": str(_find(widget, "ThemeBox").get_selected_option()),
        "terrain": _find(widget, "TerrainSlider").get_value(),
        "trees": round(_find(widget, "TreesSlider").get_value()),
        "flowers": round(_find(widget, "FlowersSlider").get_value()),
        "rocks": round(_find(widget, "RocksSlider").get_value()),
        "grass": round(_find(widget, "GrassSlider").get_value()),
        "lighting": _find(widget, "LightingSlider").get_value(),
        "sky_tint": _find(widget, "SkyTintSlider").get_value(),
        "ground_tint": _find(widget, "GroundTintSlider").get_value(),
        "light_tint": _find(widget, "LightTintSlider").get_value(),
        "mountains": _find(widget, "MountainsBox").is_checked(),
        "playable_character": _find(widget, "PlayableCharacterBox").is_checked(),
    }


def _preset_path():
    return os.path.join(unreal.Paths.project_saved_dir(), PRESET_FILENAME)


def _has_tag(actor, tag):
    return tag in [str(value) for value in actor.get_editor_property("tags")]


def _clear_world(preview_only=False, layers=None):
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        remove = preview_only and _has_tag(actor, PREVIEW_TAG)
        if not preview_only and _has_tag(actor, GENERATED_TAG):
            remove = layers is None or any(_has_tag(actor, layer) for layer in layers)
        if remove:
            unreal.EditorLevelLibrary.destroy_actor(actor)


def _hide_template_environment():
    """Hide the Third Person template geometry while keeping its light and player start alive."""
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        if any(_has_tag(actor, tag) for tag in (GENERATED_TAG, PREVIEW_TAG, TEMPLATE_HIDDEN_TAG)):
            continue
        if isinstance(actor, unreal.StaticMeshActor):
            tags = list(actor.get_editor_property("tags"))
            tags.append(unreal.Name(TEMPLATE_HIDDEN_TAG))
            actor.set_editor_property("tags", tags)
            actor.set_is_temporarily_hidden_in_editor(True)
            actor.set_actor_hidden_in_game(True)


def _restore_template_environment():
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        if _has_tag(actor, TEMPLATE_HIDDEN_TAG):
            actor.set_editor_property("tags", [tag for tag in actor.get_editor_property("tags") if str(tag) != TEMPLATE_HIDDEN_TAG])
            actor.set_is_temporarily_hidden_in_editor(False)
            actor.set_actor_hidden_in_game(False)


def _tag(actor, generated_tag, layer, label):
    actor.set_editor_property("tags", [unreal.Name(generated_tag), unreal.Name(layer)])
    actor.set_actor_label(label)
    return actor


def _get_tint_material():
    """Create one real parameterized material instead of hoping BasicShapeMaterial exposes a color parameter."""
    global _tint_material
    if _tint_material:
        return _tint_material
    material_path = "/Game/WorldBuilder/Materials/M_IWB_Tint"
    material = unreal.load_asset(material_path)
    if not material:
        tools = unreal.AssetToolsHelpers.get_asset_tools()
        material = tools.create_asset(
            "M_IWB_Tint",
            "/Game/WorldBuilder/Materials",
            unreal.Material,
            unreal.MaterialFactoryNew(),
        )
        if not material:
            raise RuntimeError("Could not create the World Builder tint material")
        tint = unreal.MaterialEditingLibrary.create_material_expression(
            material, unreal.MaterialExpressionVectorParameter, -380, 0
        )
        tint.set_editor_property("parameter_name", "Tint")
        tint.set_editor_property("default_value", unreal.LinearColor(0.12, 0.42, 0.12, 1.0))
        roughness = unreal.MaterialEditingLibrary.create_material_expression(
            material, unreal.MaterialExpressionConstant, -380, 150
        )
        roughness.set_editor_property("r", 0.82)
        unreal.MaterialEditingLibrary.connect_material_property(tint, "", unreal.MaterialProperty.MP_BASE_COLOR)
        unreal.MaterialEditingLibrary.connect_material_property(roughness, "", unreal.MaterialProperty.MP_ROUGHNESS)
        unreal.MaterialEditingLibrary.recompile_material(material)
        unreal.EditorAssetLibrary.save_loaded_asset(material)
    _tint_material = material
    return material


def _spawn_mesh(mesh, location, scale, rotation, generated_tag, layer, label, tint=None):
    actor = unreal.EditorLevelLibrary.spawn_actor_from_object(mesh, location, rotation)
    actor.set_actor_scale3d(scale)
    _tag(actor, generated_tag, layer, label)
    if tint:
        component = actor.get_component_by_class(unreal.StaticMeshComponent)
        if component:
            component.set_material(0, _get_tint_material())
            material = component.create_dynamic_material_instance(0)
            if material:
                material.set_vector_parameter_value("Tint", tint)
    return actor


def _kenney_mesh(name):
    """Load one of the small, intentionally chosen imported Nature Kit assets."""
    return unreal.load_asset(KENNEY_DESTINATION + "/" + name)


def _kenney_meshes(names):
    # Some FBX files expose a material with the same name as their mesh.
    # Never pass such an asset to SpawnActor; only real StaticMesh assets belong
    # in the generator.
    return [mesh for mesh in (_kenney_mesh(name) for name in names) if isinstance(mesh, unreal.StaticMesh)]


def _kenney_zip_path():
    return os.path.join(os.path.expanduser("~"), "Downloads", KENNEY_ZIP_FILENAME)


def _kenney_is_imported():
    # One representative asset is sufficient because all files are imported
    # together in one batch. This also makes a second click instant.
    return unreal.EditorAssetLibrary.does_asset_exist(KENNEY_DESTINATION + "/tree_detailed")


def _import_kenney_assets():
    """Extract and import only the models the generator uses, from the user's zip."""
    if _kenney_is_imported():
        return True, "Nature Kit is already ready."
    archive_path = _kenney_zip_path()
    if not os.path.isfile(archive_path):
        return False, "Nature Kit zip not found in Downloads."
    source_dir = os.path.join(unreal.Paths.project_intermediate_dir(), "WorldBuilderNatureSource")
    os.makedirs(source_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for name in KENNEY_FILES:
                member = "Models/FBX format/{}.fbx".format(name)
                destination = os.path.join(source_dir, name + ".fbx")
                with archive.open(member) as source, open(destination, "wb") as target:
                    target.write(source.read())
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        return False, "Could not unpack Nature Kit: {}".format(error)

    tasks = []
    for name in KENNEY_FILES:
        options = unreal.FbxImportUI()
        options.set_editor_property("import_mesh", True)
        options.set_editor_property("import_as_skeletal", False)
        options.set_editor_property("import_materials", True)
        options.set_editor_property("import_textures", True)
        task = unreal.AssetImportTask()
        task.set_editor_property("filename", os.path.join(source_dir, name + ".fbx"))
        task.set_editor_property("destination_path", KENNEY_DESTINATION)
        task.set_editor_property("automated", True)
        task.set_editor_property("replace_existing", True)
        task.set_editor_property("save", True)
        task.set_editor_property("options", options)
        tasks.append(task)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(tasks)
    unreal.EditorAssetLibrary.save_directory(KENNEY_DESTINATION, only_if_is_dirty=False, recursive=True)
    if _kenney_is_imported():
        return True, "Nature Kit imported: trees, flowers, grass, rocks and cliffs are ready."
    return False, "Unreal did not finish importing the Nature Kit. Check the Output Log."


def import_nature_kit(widget):
    _status(widget, "Importing selected Nature Kit models…")
    success, message = _import_kenney_assets()
    _status(widget, message if success else "Import failed: " + message)


def _pcg_tree_meshes():
    """Use the real, bundled PCG sample tree models before falling back to primitives."""
    return [mesh for mesh in (
        unreal.load_asset("/PCG/SampleContent/SimpleForest/Meshes/PCG_Tree_01"),
        unreal.load_asset("/PCG/SampleContent/SimpleForest/Meshes/PCG_Tree_02"),
        unreal.load_asset("/PCG/SampleContent/SimpleForest/Meshes/PCG_Tree_03"),
    ) if mesh]


def _spawn_grounded_mesh(mesh, location_xy, height, rotation, generated_tag, layer, label):
    """Scale a mesh by its actual bounds and put its base on the terrain."""
    # A malformed FBX import can expose a material under a mesh name. Skipping
    # it is always better than leaving the user with a half-generated world.
    if not isinstance(mesh, unreal.StaticMesh):
        unreal.log_warning("World Builder skipped a non-mesh asset in {}".format(label))
        return None
    bounds = mesh.get_bounds()
    mesh_height = max(1.0, bounds.box_extent.z * 2.0)
    scale = height / mesh_height
    # Asset pivots differ; place the lowest point exactly on the generated terrain.
    ground_offset = -(bounds.origin.z - bounds.box_extent.z) * scale
    return _spawn_mesh(
        mesh,
        unreal.Vector(location_xy.x, location_xy.y, ground_offset),
        unreal.Vector(scale, scale, scale),
        rotation,
        generated_tag,
        layer,
        label,
    )


def _spawn_stylized_tree(mesh, location_xy, height, rotation, generated_tag):
    return _spawn_grounded_mesh(mesh, location_xy, height, rotation, generated_tag, TREE_TAG, "IWB_StylizedTree")


def _elevation(x, y, terrain):
    frequency = 6.25 / terrain
    return 0.5 + 0.5 * math.sin(x * frequency) * math.cos(y * frequency * 1.18)


def _slope(x, y, terrain):
    frequency = 6.25 / terrain
    return abs(math.cos(x * frequency) * math.sin(y * frequency * 1.18))


def _palette(data):
    """Three close cousins of the same low-poly world, rather than unrelated themes."""
    ground_richness = data["ground_tint"]
    theme = data["theme"]
    if theme == "Golden Meadow":
        return {
            "ground": unreal.LinearColor(0.25 + ground_richness * 0.20, 0.30 + ground_richness * 0.28, 0.10, 1.0),
            "leaf": unreal.LinearColor(0.25, 0.39, 0.08, 1.0),
            "grass": unreal.LinearColor(0.37, 0.48, 0.10, 1.0),
            "mountain": unreal.LinearColor(0.42, 0.23, 0.13, 1.0),
            "flower_a": unreal.LinearColor(0.98, 0.52, 0.35, 1.0),
            "flower_b": unreal.LinearColor(0.98, 0.83, 0.35, 1.0),
        }
    if theme == "Misty Highlands":
        return {
            "ground": unreal.LinearColor(0.09, 0.24 + ground_richness * 0.20, 0.18, 1.0),
            "leaf": unreal.LinearColor(0.05, 0.25, 0.15, 1.0),
            "grass": unreal.LinearColor(0.10, 0.32, 0.20, 1.0),
            "mountain": unreal.LinearColor(0.23, 0.29, 0.31, 1.0),
            "flower_a": unreal.LinearColor(0.65, 0.72, 0.95, 1.0),
            "flower_b": unreal.LinearColor(0.72, 0.52, 0.82, 1.0),
        }
    return {
        "ground": unreal.LinearColor(0.055 + ground_richness * 0.18, 0.17 + ground_richness * 0.34, 0.04 + ground_richness * 0.10, 1.0),
        "leaf": unreal.LinearColor(0.025, 0.29, 0.06, 1.0),
        "grass": unreal.LinearColor(0.06, 0.37, 0.06, 1.0),
        "mountain": unreal.LinearColor(0.32, 0.19, 0.12, 1.0),
        "flower_a": unreal.LinearColor(0.98, 0.57, 0.65, 1.0),
        "flower_b": unreal.LinearColor(0.96, 0.84, 0.56, 1.0),
    }


def _build_base(data, tag, origin_x):
    cube = unreal.load_asset("/Engine/BasicShapes/Cube")
    terrain = data["terrain"]
    _spawn_mesh(cube, unreal.Vector(origin_x, 0, -55), unreal.Vector(terrain / 100.0, terrain / 100.0, 0.55), unreal.Rotator(), tag, TERRAIN_TAG, "IWB_Terrain", _palette(data)["ground"])


def _build_mountains(data, tag, origin_x):
    if not data["mountains"]:
        return 0
    cone = unreal.load_asset("/Engine/BasicShapes/Cone")
    # Keep the imported cliff kit for manual use. Distant, simple hills are
    # intentionally safer here: the raw cliffs have incompatible proportions
    # when auto-scaled into a tiny playable world.
    cliff_meshes = []
    rng, terrain, palette = random.Random(data["seed"] + 415), data["terrain"], _palette(data)
    half, total = terrain * 0.5, 0
    # Fewer, smaller cliffs create a distant rim rather than a pile of giant
    # overlapping blocks over the playable clearing.
    for _ in range(6):
        angle = rng.uniform(0.0, math.tau)
        distance = rng.uniform(half * 0.82, half * 0.94)
        x, y = math.cos(angle) * distance, math.sin(angle) * distance
        height = rng.uniform(330.0, 560.0)
        width = rng.uniform(420.0, 720.0)
        rotation = unreal.Rotator(0.0, rng.uniform(0, 360), 0.0)
        if cliff_meshes:
            _spawn_grounded_mesh(
                cliff_meshes[rng.randrange(len(cliff_meshes))],
                unreal.Vector(origin_x + x, y, 0), height, rotation, tag, MOUNTAIN_TAG, "IWB_Cliff",
            )
        else:
            _spawn_mesh(
                cone,
                unreal.Vector(origin_x + x, y, height * 0.43),
                unreal.Vector(width / 100.0, width / 100.0, height / 100.0),
                rotation,
                tag,
                MOUNTAIN_TAG,
                "IWB_Mountain",
                palette["mountain"],
            )
        total += 1
    return total


def _build_vegetation(data, tag, origin_x):
    rng, terrain = random.Random(data["seed"]), data["terrain"]
    palette = _palette(data)
    half, path_half = terrain * 0.5, max(160.0, min(360.0, terrain * 0.045))
    cube = unreal.load_asset("/Engine/BasicShapes/Cube")
    cone = unreal.load_asset("/Engine/BasicShapes/Cone")
    cylinder = unreal.load_asset("/Engine/BasicShapes/Cylinder")
    sphere = unreal.load_asset("/Engine/BasicShapes/Sphere")
    tree_meshes = _kenney_meshes(("tree_detailed", "tree_oak", "tree_pineRoundA", "tree_pineTallB")) or _pcg_tree_meshes()
    rock_meshes = _kenney_meshes(("rock_largeB", "rock_largeE", "rock_smallA", "rock_smallH"))
    boulder = unreal.load_asset("/PCG/SampleContent/SimpleForest/Meshes/PCG_Boulder_02")
    grass_meshes = _kenney_meshes(("grass_large",))
    flower_meshes = _kenney_meshes(("flower_purpleA", "flower_redB", "flower_yellowC"))
    if not all((cube, cone, cylinder, sphere)):
        raise RuntimeError("Unreal Engine basic-shape meshes are unavailable")

    trees = 0
    for _ in range(data["trees"] * 5):
        if trees >= data["trees"]:
            break
        x, y = rng.uniform(-half * 0.92, half * 0.92), rng.uniform(-half * 0.92, half * 0.92)
        if _elevation(x, y, terrain) > 0.76 or _slope(x, y, terrain) > 0.86 or abs(y) < path_half:
            continue
        height = rng.uniform(360.0, 580.0)
        if tree_meshes:
            _spawn_stylized_tree(tree_meshes[rng.randrange(len(tree_meshes))], unreal.Vector(origin_x + x, y, 0), height, unreal.Rotator(0, rng.uniform(0, 360), 0), tag)
        else:
            _spawn_mesh(cylinder, unreal.Vector(origin_x + x, y, height * 0.5), unreal.Vector(rng.uniform(0.12, 0.18), rng.uniform(0.12, 0.18), height / 100.0), unreal.Rotator(0, rng.uniform(0, 360), 0), tag, TREE_TAG, "IWB_TreeTrunk", unreal.LinearColor(0.13, 0.055, 0.018, 1.0))
            _spawn_mesh(cone, unreal.Vector(origin_x + x, y, height + 110.0), unreal.Vector(rng.uniform(0.62, 1.0), rng.uniform(0.62, 1.0), rng.uniform(1.2, 1.8)), unreal.Rotator(0, rng.uniform(0, 360), 0), tag, TREE_TAG, "IWB_TreeCrown", palette["leaf"])
        trees += 1

    rocks = 0
    while rocks < data["rocks"]:
        center_x, center_y = rng.uniform(-half * 0.86, half * 0.86), rng.uniform(-half * 0.86, half * 0.86)
        for _ in range(min(rng.randint(2, 5), data["rocks"] - rocks)):
            rock_mesh = rock_meshes[rng.randrange(len(rock_meshes))] if rock_meshes else (boulder or sphere)
            _spawn_mesh(rock_mesh, unreal.Vector(origin_x + center_x + rng.uniform(-260, 260), center_y + rng.uniform(-260, 260), rng.uniform(25, 70)), unreal.Vector(rng.uniform(0.35, 1.1), rng.uniform(0.30, 0.90), rng.uniform(0.26, 0.70)), unreal.Rotator(rng.uniform(0, 35), rng.uniform(0, 360), rng.uniform(0, 35)), tag, ROCK_TAG, "IWB_Rock", None if (rock_meshes or boulder) else unreal.LinearColor(0.18, 0.22, 0.20, 1.0))
            rocks += 1

    grass = 0
    for _ in range(data["grass"] * 4):
        if grass >= data["grass"]:
            break
        x, y = rng.uniform(-half * 0.94, half * 0.94), rng.uniform(-half * 0.94, half * 0.94)
        if abs(y) < path_half or _elevation(x, y, terrain) > 0.88:
            continue
        if grass_meshes:
            _spawn_grounded_mesh(grass_meshes[rng.randrange(len(grass_meshes))], unreal.Vector(origin_x + x, y, 0), rng.uniform(65.0, 140.0), unreal.Rotator(0, rng.uniform(0, 360), 0), tag, GRASS_TAG, "IWB_Grass")
        else:
            _spawn_mesh(cone, unreal.Vector(origin_x + x, y, 32), unreal.Vector(rng.uniform(0.035, 0.065), rng.uniform(0.035, 0.065), rng.uniform(0.32, 0.70)), unreal.Rotator(0, rng.uniform(0, 360), 0), tag, GRASS_TAG, "IWB_Grass", palette["grass"])
        grass += 1
    flowers = 0
    for _ in range(data["flowers"] * 4):
        if flowers >= data["flowers"]:
            break
        x, y = rng.uniform(-half * 0.88, half * 0.88), rng.uniform(-half * 0.88, half * 0.88)
        if abs(y) < path_half * 1.2 or _elevation(x, y, terrain) > 0.82:
            continue
        if flower_meshes:
            _spawn_grounded_mesh(flower_meshes[rng.randrange(len(flower_meshes))], unreal.Vector(origin_x + x, y, 0), rng.uniform(45.0, 85.0), unreal.Rotator(0, rng.uniform(0, 360), 0), tag, FLOWER_TAG, "IWB_Flower")
        else:
            stem_height = rng.uniform(20.0, 45.0)
            _spawn_mesh(cylinder, unreal.Vector(origin_x + x, y, stem_height * 0.5), unreal.Vector(0.012, 0.012, stem_height / 100.0), unreal.Rotator(), tag, FLOWER_TAG, "IWB_FlowerStem", palette["grass"])
            petal_color = palette["flower_a"] if flowers % 2 else palette["flower_b"]
            petal_size = rng.uniform(0.065, 0.10)
            for dx, dy in ((9, 0), (-9, 0), (0, 9), (0, -9)):
                _spawn_mesh(sphere, unreal.Vector(origin_x + x + dx, y + dy, stem_height + 10.0), unreal.Vector(petal_size, petal_size, petal_size * 0.42), unreal.Rotator(), tag, FLOWER_TAG, "IWB_FlowerPetal", petal_color)
            _spawn_mesh(sphere, unreal.Vector(origin_x + x, y, stem_height + 12.0), unreal.Vector(0.045, 0.045, 0.025), unreal.Rotator(), tag, FLOWER_TAG, "IWB_FlowerCentre", unreal.LinearColor(0.98, 0.72, 0.16, 1.0))
        flowers += 1
    _spawn_mesh(cube, unreal.Vector(origin_x, 0, 6), unreal.Vector(terrain / 100.0, (path_half * 1.4) / 100.0, 0.06), unreal.Rotator(), tag, PATH_TAG, "IWB_Path", unreal.LinearColor(0.18, 0.11, 0.055, 1.0))
    return trees, rocks, grass, flowers


def _build_lighting_and_player(data, tag, origin_x):
    sky_color = unreal.LinearColor(0.18 + data["sky_tint"] * 0.20, 0.32 + data["sky_tint"] * 0.40, 0.55 + data["sky_tint"] * 0.40, 1.0)
    light_color = unreal.LinearColor(1.0 - data["light_tint"] * 0.22, 0.86 + data["light_tint"] * 0.04, 0.64 + data["light_tint"] * 0.34, 1.0)
    # Third Person already owns a sun and a skylight. Reuse those instead of
    # adding a second pair, which caused the "multiple directional lights" warning.
    actors = unreal.EditorLevelLibrary.get_all_level_actors()
    light = next((actor for actor in actors if isinstance(actor, unreal.DirectionalLight)), None)
    if not light:
        light = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(origin_x, 0, 1100), unreal.Rotator(-52, -28, 0))
        _tag(light, tag, LIGHT_TAG, "IWB_ForestLight")
    light_component = light.get_component_by_class(unreal.DirectionalLightComponent)
    if light_component:
        # UE directional lights use a small lux value here, not 10,000× the UI value.
        light_component.set_editor_property("intensity", 1.5 + data["lighting"] * 0.70)
        light_component.set_light_color(light_color)

    sky = next((actor for actor in actors if isinstance(actor, unreal.SkyLight)), None)
    if not sky:
        sky = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(origin_x, 0, 1050), unreal.Rotator())
        _tag(sky, tag, LIGHT_TAG, "IWB_ForestSky")
    sky_component = sky.get_component_by_class(unreal.SkyLightComponent)
    if sky_component:
        sky_component.set_editor_property("intensity", 0.45 + data["lighting"] * 0.07)
        sky_component.set_light_color(sky_color)
        sky_component.recapture_sky()
    if data["playable_character"] and not any(isinstance(actor, unreal.PlayerStart) for actor in actors):
        start = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PlayerStart, unreal.Vector(origin_x - data["terrain"] * 0.275, 0, 120), unreal.Rotator())
        _tag(start, tag, PLAYER_TAG, "IWB_ThirdPersonStart")


def _build_world(widget, preview=False, vegetation_only=False):
    data = _values(widget)
    tag, origin_x = (PREVIEW_TAG, data["terrain"] * 0.68) if preview else (GENERATED_TAG, 0.0)
    if not vegetation_only:
        _build_base(data, tag, origin_x)
        _build_mountains(data, tag, origin_x)
    trees, rocks, grass, flowers = _build_vegetation(data, tag, origin_x)
    if not preview and not vegetation_only:
        _build_lighting_and_player(data, tag, origin_x)
    _status(widget, "{} '{}' (seed {}): {} trees, {} flowers, {} rocks, {} grass.".format("Preview" if preview else "Generated", data["world_name"], data["seed"], trees, flowers, rocks, grass))


def generate(widget):
    if not _kenney_is_imported() and os.path.isfile(_kenney_zip_path()):
        _status(widget, "First build: importing the selected Nature Kit models…")
        imported, message = _import_kenney_assets()
        if not imported:
            _status(widget, "Nature Kit was skipped: " + message)
    _clear_world()
    _clear_world(preview_only=True)
    _hide_template_environment()
    try:
        _build_world(widget)
    except Exception as error:
        _clear_world()
        _status(widget, "Generation stopped safely: {}".format(error))
        unreal.log_error("World Builder generation stopped: {}".format(error))
        return
    generated = [actor for actor in unreal.EditorLevelLibrary.get_all_level_actors() if _has_tag(actor, GENERATED_TAG)]
    unreal.get_editor_subsystem(unreal.EditorActorSubsystem).set_selected_level_actors(generated)
    _status(widget, "World generated. The terrain is selected — press F in the viewport to frame it.")


def preview(widget):
    _clear_world(preview_only=True)
    _build_world(widget, preview=True)


def update_vegetation(widget):
    _clear_world(layers=(TREE_TAG, FLOWER_TAG, ROCK_TAG, GRASS_TAG, PATH_TAG))
    _build_world(widget, vegetation_only=True)


def start_over(widget):
    _clear_world()
    _clear_world(preview_only=True)
    reset(widget)
    _status(widget, "Fresh start. The generated world is cleared and the controls are reset.")


def save_preset(widget):
    with open(_preset_path(), "w", encoding="utf-8") as handle:
        json.dump(_values(widget), handle, indent=2)
    _status(widget, "Preset saved to Saved/InteractiveWorldBuilderPreset.json.")


def load_preset(widget):
    if not os.path.exists(_preset_path()):
        _status(widget, "No saved preset yet.")
        return
    with open(_preset_path(), "r", encoding="utf-8") as handle:
        data = json.load(handle)
    _find(widget, "WorldNameBox").set_text(data.get("world_name", "Whispering Pines"))
    _find(widget, "SeedBox").set_text(str(data.get("seed", 20261018)))
    legacy_themes = {"Mountain (next pass)": "Misty Highlands", "Meadow (next pass)": "Golden Meadow", "Beach (next pass)": "Golden Meadow"}
    _find(widget, "ThemeBox").set_selected_option(legacy_themes.get(data.get("theme", "Forest"), data.get("theme", "Forest")))
    for key, source in (("Terrain", "terrain"), ("Trees", "trees"), ("Flowers", "flowers"), ("Rocks", "rocks"), ("Grass", "grass"), ("Lighting", "lighting"), ("SkyTint", "sky_tint"), ("GroundTint", "ground_tint"), ("LightTint", "light_tint")):
        _find(widget, key + "Slider").set_value(data.get(source, _find(widget, key + "Slider").get_value()))
    _find(widget, "MountainsBox").set_is_checked(data.get("mountains", True))
    _find(widget, "PlayableCharacterBox").set_is_checked(data.get("playable_character", True))
    _refresh_readouts(widget)
    _status(widget, "Preset loaded. Preview checks it without replacing the generated world.")


def reset(widget):
    _find(widget, "WorldNameBox").set_text("Whispering Pines")
    _find(widget, "SeedBox").set_text("20261018")
    _find(widget, "ThemeBox").set_selected_option("Forest")
    defaults = {"Terrain": 4000.0, "Trees": 45.0, "Flowers": 34.0, "Rocks": 24.0, "Grass": 120.0, "Lighting": 8.0, "SkyTint": 0.55, "GroundTint": 0.32, "LightTint": 0.12}
    for key, value in defaults.items():
        _find(widget, key + "Slider").set_value(value)
    _find(widget, "MountainsBox").set_is_checked(True)
    _find(widget, "PlayableCharacterBox").set_is_checked(True)
    _refresh_readouts(widget)
    _status(widget, "Controls reset. Existing generated actors were not changed.")


def delete_world(widget):
    _clear_world()
    _clear_world(preview_only=True)
    _restore_template_environment()
    _status(widget, "Generated actors removed. The original Third Person level is visible again.")


def _make_click_callback(action, widget):
    def callback():
        action(widget)
    return callback


def _make_slider_callback(widget):
    def callback(_value):
        _refresh_readouts(widget)
    return callback


def bind_widget(widget):
    global _callbacks, _bound_widget
    if _bound_widget and _bound_widget == widget:
        return
    _callbacks = []
    actions = {"GenerateButton": generate, "PreviewButton": preview, "ImportNatureButton": import_nature_kit, "UpdateVegetationButton": update_vegetation, "StartOverButton": start_over, "SavePresetButton": save_preset, "LoadPresetButton": load_preset, "ResetButton": reset, "DeleteButton": delete_world}
    for name, action in actions.items():
        callback = _make_click_callback(action, widget)
        _find(widget, name).on_clicked.add_callable(callback)
        _callbacks.append(callback)
    for key in ("Terrain", "Trees", "Flowers", "Rocks", "Grass", "Lighting", "SkyTint", "GroundTint", "LightTint"):
        callback = _make_slider_callback(widget)
        _find(widget, key + "Slider").on_value_changed.add_callable(callback)
        _callbacks.append(callback)
    _refresh_readouts(widget)
    _bound_widget = widget


def open_tool():
    asset = build_widget_asset()
    subsystem = unreal.get_editor_subsystem(unreal.EditorUtilitySubsystem)
    widget = subsystem.find_utility_widget_from_blueprint(asset)
    if not widget:
        widget = subsystem.spawn_and_register_tab(asset)
    _build_runtime_ui(widget)
    bind_widget(widget)
    _status(widget, "World Builder v3.1 ready — clean ground placement enabled.")
    return widget


def reload_tool():
    """Refresh the live panel after a Python edit, without restarting Unreal."""
    asset = build_widget_asset()
    subsystem = unreal.get_editor_subsystem(unreal.EditorUtilitySubsystem)
    widget = subsystem.find_utility_widget_from_blueprint(asset)
    if not widget:
        widget = subsystem.spawn_and_register_tab(asset)
    _build_runtime_ui(widget, force=True)
    bind_widget(widget)
    _status(widget, "World Builder v3 loaded — safe generation is ready.")
    return widget


@unreal.uclass()
class InteractiveWorldBuilderMenuEntry(unreal.ToolMenuEntryScript):
    @unreal.ufunction(override=True)
    def get_label(self, context):
        return "Open Interactive World Builder"

    @unreal.ufunction(override=True)
    def get_tool_tip(self, context):
        return "Open the seeded forest world-building panel."

    @unreal.ufunction(override=True)
    def execute(self, context):
        open_tool()


def register_menu():
    menus = unreal.ToolMenus.get()
    menus.unregister_owner_by_name(MENU_OWNER)
    menu = menus.extend_menu("LevelEditor.MainMenu.Tools")
    entry = InteractiveWorldBuilderMenuEntry()
    entry.init_entry(MENU_OWNER, "LevelEditor.MainMenu.Tools", "WorldBuilder", "InteractiveWorldBuilder.Open", "Open Interactive World Builder")
    menu.add_menu_entry_object(entry)
    menus.refresh_all_widgets()


def _watch_for_manually_opened_panel(_delta_seconds):
    """Convert a manually-run host EUW into the World Builder panel."""
    try:
        if not _widget_asset:
            return
        widget = unreal.get_editor_subsystem(unreal.EditorUtilitySubsystem).find_utility_widget_from_blueprint(_widget_asset)
        if widget and not widget.find_child_widget_by_name("IWBTitle"):
            _build_runtime_ui(widget)
            bind_widget(widget)
            _status(widget, "World Builder v3.1 ready — clean ground placement enabled.")
    except Exception as error:
        unreal.log_warning("World Builder panel watch skipped: {}".format(error))


def startup():
    try:
        global _watch_handle, _widget_asset
        register_menu()
        _widget_asset = build_widget_asset()
        if _watch_handle is None:
            _watch_handle = unreal.register_slate_post_tick_callback(_watch_for_manually_opened_panel)
        return open_tool()
    except Exception as error:
        unreal.log_error("Interactive World Builder startup failed: {}".format(error))
        return None
