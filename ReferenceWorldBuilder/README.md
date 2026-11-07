# Reference World Builder

**PFA (HSG-5232-A) - Research Project 1**
**Unreal Engine 5.8 · Editor Utility Widget · Python**

Reference World Builder turns a concept image or Photoshop file into a small,
walkable Unreal environment. The picture's colours set the mood and lighting,
its brightness becomes terrain height, and its colour regions decide what grows
where. In Play mode, props near the character glow and gently pulse.

## How the image drives the world

| From the image | Becomes in Unreal |
| --- | --- |
| Dominant palette (k-means) | Sun colour, sky light, fog tint, water colour, glow accent |
| Luminance (brightness) | Terrain height - bright areas are hills, dark areas are lowlands |
| Darkest ~22% of the ground | Water level |
| Green hues | Trees and grass |
| Warm, saturated hues (reds/oranges/pinks) | Flowers |
| Dark, low-saturation areas | Rocks |
| Top ~24% of the image | Treated as sky and left out of the terrain |
| Bottom of the image | Foreground - the player starts here, facing into the picture |

## Running it

1. Open `ReferenceWorldBuilder.uproject` in Unreal 5.8. The panel opens by itself
   (or use **Tools > Open Reference World Builder**).
2. **Browse...** for a PNG, JPG, PSD or PSB (or paste a path) and press
   **Analyze reference**. The palette swatches fill in.
3. Adjust **World size**, **Terrain height**, **Prop density** and **Seed**, then
   press **Generate world**.
4. Press **Play in editor** and walk around. Save the level (Ctrl+S) to keep a result.
5. **Clear** removes everything the tool made (all generated actors are tagged
   `RWB_Generated` and live in the `ReferenceWorldBuilder` outliner folder).

## Default mode: your painting, in 3D

Every image or PSD you submit (Browse / Analyze) generates a world straight away (turn off
**Generate the world right after analyzing** to wait). Stand at the start position and the
world lines up with the picture.

1. **Recognition.** For a flat image, OWLv2 finds objects by name, SlimSAM cuts out each one's
   real outline, and CLIP double-checks the label. For a PSD, each layer's name says what it is
   and its pixels are the outline; layers named like "Layer 3" are recognised from their pixels.
2. **Placement.** The painting's camera is solved from its horizon, so every object lands where
   it was painted: lower in the picture = closer, the same pixel height further back = bigger.
   The base of each outline is where it meets the ground. If the depth map shows the base is
   hidden behind something, the depth map is used instead. Small things above the horizon
   (butterflies, hanging lanterns) float where they were painted. **FG / MG / BG** tags override
   the distance.
3. **New 3D objects.** Each object becomes a new mesh built from its outline and painted with
   its own piece of the picture, then refined in Blender. Library models, simple shapes and
   painted cut-outs are now only fallbacks for objects without an outline.
4. **Terrain.** Water follows the painted shoreline, anything painted standing keeps dry ground
   under it (a tower on an island stays on its island), and the ground is calmer inside the
   painted view.
5. **Extra copies** of repeated things, and of trees even if only one was recognised (amount set by
   **Prop density**), only go **outside the painted view**, so they fill the map without disturbing
   the picture. Turn them off with
   **Add extra copies of repeated things outside the painted view**. Characters and creatures
   are never duplicated.

### Cities, rooms, deserts, snow and coasts

Before anything else the tool works out **what kind of place** the picture shows (CLIP compares the whole
picture with descriptions of each kind). The panel shows it after analysing, for example
`Setting: city (city 98%, interior 1%, desert 0%)`. Each kind of place changes the rest of the build:

| Setting | Also looks for | Ground can be | Terrain | Grass / extra trees | Walls |
| --- | --- | --- | --- | --- | --- |
| Nature | (the usual trees, rocks, water...) | grass, meadow, dirt, sand, rock, water | rolling | yes / yes | - |
| City | buildings, storefronts, signs, billboards, street lamps, poles, cars, bicycles | road, pavement, grass, water | flat | no / no | street |
| Interior | furniture, lamps, doors, windows, rugs | floorboards, tiles, carpet | flat | no / no | room |
| Desert | cacti, palms, tents, ruins | sand, dirt, rock, water | dunes (2.2x) | no / no | - |
| Snow | snowy pines, igloos | snow, ice, rock, water | hilly (1.8x) | no / yes | - |
| Coast | palms, sailboats, piers, lighthouses | sand, grass, rock, water | gentle | yes / no | - |

**Streets are lined with individual buildings.** When a city picture looks straight down a street (a level
camera: building corners upright, facades down both sides), each side is cut into separate buildings where the
roofline jumps, the facade colour changes, or the detector boxed a building. Every building becomes its own
block (`RWB_Building_L01`, `R02`...) with its own height from its painted roofline, a small setback, a flat roof and
ends in its own colour, and a front painted from the picture. Cities that aren't a level street view (looking up,
tilted, a skyline) are "open": each detected building becomes its own **block** model, a squared-off outline with
flat walls and a stepped roofline (Blender keeps it square instead of smoothing it).

**Signs and billboards are boards you can read.** Every sign the detector finds becomes its own model
(`RWB_Sign_...`): a thin board with a dark frame whose face is a sharp crop of the sign from your full-resolution
picture, lit from inside so the lettering stays readable at night. Tall signs stick out from their wall facing down
the street and are printed on both sides; wide signs at the sides lie flat on their facade; signs in the middle hang
across the street facing you. Each board covers exactly the spot it was painted, seen from the start.

**Rooms (and streets with no buildings found) are walled in by the picture itself.** Facades are too big to cut out as objects, so the
painting is projected onto two side walls and an end wall from the start position: from there the walls look
exactly like the picture, and as you walk down the street the shop fronts, signs and windows stay where they
were painted. The street is as wide as the painted people and things are spread (at least about 13 m), things
painted inside it stay inside it, and the walls stop you walking into the buildings.

**Night pictures** (dark overall, dark sky) get a set sun and a dim sky; every glowing, saturated patch
(neon signs, lit windows) becomes a coloured light on the wall where it was painted.

If the guess is wrong, click **Setting: Auto** in the BUILD section to choose the kind of place yourself,
then press **Analyze reference** again.

### How each object is modelled

| Shape | Used for | How |
| --- | --- | --- |
| Inflate | trees, bushes, rocks, flowers, anything unknown | outline puffed out by distance to its edge, so trunks are fat and branches thin |
| Lathe | towers, mushrooms, lanterns, pillars, pines, furniture | each row of the outline becomes a ring |
| Slab | walls, gates, signs, people, houses, castles, buildings, vehicles | outline extruded with a soft bevel; houses and castles are deep, signs are thin |
| Ground | boats, lily pads, bridges | outline projected through the camera onto the ground, then domed |

### Grass and water that work like grass and water

The tool reads what the painted ground is made of (grass, meadow, dirt, sand, rock, water) by grouping
the paint below the horizon by colour and asking CLIP what each group looks like. Muted paint counts
by its hue; near-black shadow and framing takes on whatever surrounds it. The panel reports the
result, such as `Ground: grass 84%, meadow 16%`.

- **The floor you painted is pasted onto the ground.** For a flat picture SlimSAM outlines the floor (prompted
  along the bottom of the picture, grown by colour, small gaps such as road markings closed), and the
  picture's own floor pixels are projected through the painting's camera onto the terrain. Seen from the
  start, the ground is the painting. They fade out where the picture gets too coarse (roughly 25-40 m away),
  and where objects stand on the floor (their outlines, not their boxes) the surface's tile fills in.
- **Beyond that**, the ground is painted with a seamless tile stitched from each surface's own paint, placed using the
  painting's camera, so each spot gets the surface painted there. Slopes and shorelines go bare.
- **What each surface is** is judged by CLIP seeing it in its surroundings (a lit sidewalk next to a kerb reads
  as a sidewalk), plus a little by its colour. In streets and rooms only the floor counts as ground, so
  shop fronts and walls below the horizon don't.
- **Grass** grows on grassy ground (turn off **Grow grass on grassy ground** to skip it).
- **Water is only made when it is really water.** Every candidate (a pond the detector found, a bluish part of
  the ground) is checked by CLIP looking at it with its surroundings and at the whole picture: a lake in a
  forest reads 81% water, blue-lit pavement or a neon-lit road 10-20%, so those stay dry. Grey paint (a
  sketch) is never water, and streets and rooms need more proof. The panel says what it decided, e.g.
  `Water: none (a bluish part of the ground looked wet, but reads as dry ground in its surroundings)`.
  Untick **Water where the picture has water** to never make any.
- **Water** is never a floor, in any build mode. If the picture has water, you walk into it, down to the
  basin below, and wade at about half speed at depth.
  Swimming isn't built: you walk on the lake bed.

### Low poly or as painted

**Model style** runs from 0 (generic low poly) to 1 (as painted). Lower settings use a coarser outline
and fewer triangles; below 0.6 models use flat facets, and below 0.2 they become simple convex forms.
The paint stays either way. Press **Generate world** after moving it.

### Your brushstrokes on every side

The tool reads the painting's brushwork (direction, width, length, shown in the panel) and pulls it
out as a seamless map stitched from real patches of your paint. Each model is then baked: surfaces
facing the painter show the painting itself; surfaces it never showed (backs, sides) get that part
of the object's broad colour with your strokes over it, at the scale they have in the painting. An
object with enough paint uses its own marks; small ones use the whole painting's. The **Brushstrokes**
slider sets how strong the marks are on the unseen surfaces (0 = plain colour); changing it rebakes
without rebuilding any geometry.

**What to expect:** these are concept-stage forms. The silhouette comes from one view, so depth is
inferred, and a detail you painted (a window) isn't invented on the back: the back gets colour and
brushwork, not new features. Add your own finished model to `../ModelLibrary` to use it for the
automatic extra copies.

### Choosing how many of each model

After generating, the **MODELS** section lists every generated model with its picture:

- **- / + / +5** change that model's count straight away, with no rebuild. Copies appear around the
  original at slightly different sizes and turned different ways. 0 removes the model.
- Click a model's **name** to select it and its copies in the level.
- **Keep added copies out of the painted view** sends new copies elsewhere on the map instead.
- Counts are remembered per picture, so generating again restores them.
- If there's no room left the panel says how many fit. Boats and lily pads look for water first.

### Fixing a model by describing it

Each model in the **MODELS** list has a **What is it?** box. Type what the object really is
("birdcage", "spotted toadstool", "wooden canoe") and press **Find model**. The tool searches:

- **Poly Haven** - about 500 high-quality models, all CC0 (free for anything, no credit needed).
- **Objaverse** - a large research collection of Sketchfab models in 1,156 categories (mushrooms,
  birdcages, lanterns, canoes, butterflies...). Only CC0 and CC-BY models are used.
- **Your `../ModelLibrary` folder** - any .glb / .gltf / .fbx / .obj, matched by file name.

The best matches appear as **pictures under the row**. CLIP ranks them by how well each preview fits
your words and the shape you painted, so pick whichever looks right. The chosen model is turned to face
the painting, leaned toward the painted proportions, given your **Model style**, and painted with your
picture's colours and brushstrokes. Tick **Found models keep their own colours** to keep its original look.
**Use painted** goes back to the model made from your painting. Choices are remembered when you generate again.

- The first search takes about 30-70 seconds (it downloads small indexes and previews); later ones are faster.
- Gazebos and castles aren't categories in Objaverse, so describe their parts or a near thing instead
  ("pavilion", "tower", "birdcage").
- **Credits:** CC-BY models must be credited if you share work that uses them. Every model you use is
  listed with its author and licence in `ModelLibrary/_online/CREDITS.txt`.

### Drawing your own shapes

**Draw a new shape...** opens the Shape Sketch window beside Unreal, with your painting as the canvas.
**Where you draw is where the object goes in the world**, using the same camera as everything else.

1. Choose a tool: **Outline** (draw around a shape; it fills itself), **Brush**, **Eraser**.
2. Choose a **3D shape**: Puffy, Round, Flat front or Lying down.
3. Choose its paint: **Under shape** uses what you drew over. **Pick texture** lets you drag a box over
   any part of the painting, and the shape is filled with new paint stitched from that area's strokes.
4. **Send to Unreal**. It appears within a few seconds and joins the Models list, so you can make more
   of it. Keep drawing: the window stays open. Drawn shapes come back when you generate again.

Keys: L outline, B brush, E eraser, T pick texture, Z undo, Enter send, Esc close. Draw above the horizon
to make something float. Generate the world first: drawn shapes go into it.

### Improving recognition

- **PSDs give the best results**: a named layer is never mislabelled and its outline is exact.
- Teach the detector new things in `../rwb_layer_keywords.json`, under `"detector"`:
  `"a spinning wheel": "furniture"`. The category decides the 3D shape and real-world size.
- The panel lists what was found. A wrong label on a flat image only changes the shape it is
  built with, not where it goes.

### Blender

Found automatically (`Program Files\Blender Foundation`, `PATH`, or the `BLENDER_EXE` environment
variable). It makes each mesh watertight (voxel remesh), smooth and game-sized (decimated to
600-9,000 triangles by on-screen size), unwraps it for the bake and exports an FBX. Without
Blender the raw meshes are imported with the painting projected from the front, so nothing
breaks, but they are heavier and rougher and **don't get brushstrokes on their backs and sides**.
Blender does no recognition: seeing the objects is the vision models' job.

### Timing and downloads

The first analysis downloads about 350 MB of models once (OWLv2 155 MB, SlimSAM 40 MB, CLIP 153 MB;
offline, put the files in `../models/owlv2`, `slimsam`, `clip`). Then, on a CPU with no NVIDIA GPU:
a first analyze + generate takes about 3-4 minutes. Results are
cached, so generating again takes about 1 minute, and a drawn shape takes a few seconds. **Unreal's window does not respond during these
steps; it hasn't crashed.**

**Recreate the painting's exact layout instead** switches to the earlier modes (PSD layer layout /
single-image depth reconstruction).

## Layered PSDs: every layer becomes the thing it's named after

When you analyze a `.psd` / `.psb`, Photoshop exports each visible layer and
group separately. The **layer name decides what the object is**; the **layer's
pixels decide where it stands, how big it is, its proportions and its colours**;
and the **layer's artwork is painted onto the object** (blend it with
**Art blend**: 0 = model colours only, 1 = your painting).

| Name the layer... | You get |
| --- | --- |
| Tree, Oak, Forest, Pine, Fir | Tree models - one per painted tree, a wide tree line is split into a row |
| Bush, Hedge, Thorns, Briar, Thorn wall | Bushes (thorn walls become a row of dark bushes) |
| Rock, Boulder, Cliff | Rock / cliff models |
| Castle, Tower, House, Cottage, Wall, Fence, Gate, Bridge, Pillar, Lantern | Buildings assembled from towers, roofs and walls (lanterns also light up) |
| Aurora, Princess, Knight, Fairy, Owl, Horse, Spindle, Statue | A painted standee (cut-out of your artwork) at its place in the scene |
| Mountains, Hills | Raise the terrain where they're painted |
| Lake, River, Moat, Water | Carve the terrain and add water there |
| Ground, Meadow, Field, Path, Road | Colour the terrain from the painted layer |
| Sky, Clouds, Background | A painted backdrop behind the world + sky colour |
| Fog, Mist / Light, Glow | Fog colour / accent light colour |
| Grass, Flowers, Roses | Scattered plants where they're painted, tinted from the layer |
| Sketch, Lineart, Shadows, Highlights | Skipped (paint-only layers) |
| Layer 3, Group 1 (names that say nothing) | Recognised from their pixels. A solid compact shape is kept even if unrecognised; washes and shading are skipped. Groups are opened, never skipped |

Rules of thumb:

- **The last recognised word wins**: "Castle tower" -> tower, "Tree shadow" -> skipped.
- **Force a type** with brackets or a colon: `[rock] the big thing`, `tower: west`.
- **Groups**: a group with a recognised name is built as one object; an
  unrecognised group is opened and its layers are used individually.
- **Unknown names** become painted standees, and the panel lists every layer
  and what it became, so you can rename layers and analyze again.
- Add your own words to `../rwb_layer_keywords.json` (next to the project).
- **Distance tags: FG / MG / BG.** Add them anywhere in a layer or group name
  to say how far away it is, on top of what it is:
  - `FG` / `Foreground` / `Front` / `Near` - right in front of the player
  - `MG` / `Midground` / `Middle ground` / `Mid` - the middle of the world
  - `BG` / `Background` / `Back` / `Far` / `Distant` - far off, near the horizon

  `Trees BG` is still trees, just placed far away; `Thorns FG` is a thorn wall
  right in front of you; `Mountains BG` raises only the distant terrain. A
  **group** named `FG`, `MG` or `BG` passes its tag to every layer inside it,
  and a tag on a layer beats its group's tag. Inside a band, objects keep the
  front-to-back order they have in the painting. Works with `FGTrees`,
  `Trees_BG`, `bg trees`, etc. A layer named only `BG` / `Background` is still
  used as the sky backdrop.
- Untagged objects sit where the **bottom** of the painted shape touches the
  ground: lower in the picture = closer to the player. The player starts at the bottom
  edge, looking into the painting.
- Sizes blend what you painted with sensible real sizes (a person is about 1.8 m, a
  castle about 30 m). **Object size** scales everything.

The first analysis of a PSD takes a while (Photoshop exports every layer); after
that it is cached in `Saved/RWBLayers/` until the PSD changes. Layer artwork is
imported to `Content/ReferenceWorldBuilder/Layers/`.

## Single images: depth + perspective

For a flat PNG/JPG (or a PSD with **PSD: build objects from layer names** turned
off), **Single image: rebuild depth + perspective** is on by default:

1. **Depth** - *Depth Anything V2 (small)* estimates how far away every pixel is.
   It runs on the CPU through `onnxruntime` (already installed). The model
   (~99 MB) downloads from Hugging Face the first time you analyze, then stays
   cached. Offline? Put the file at `../models/depth_anything_v2_small.onnx`
   (from `onnx-community/depth-anything-v2-small`, file `onnx/model.onnx`).
2. **Camera** - the sky and the horizon line are found, and the painting's
   camera is solved (**Painting lens (FOV)**, default 60 degrees), so everything
   lands at its real distance: things near the horizon are far away, not
   halfway.
3. **Terrain** follows the depth map, and **the painting is projected onto it**
   from the painting's own camera. Stand at the start and the ground looks like
   your picture. Spots the camera never saw (behind trees and hills) get
   matching colours. **Art blend** sets how much of the painting shows.
4. **Objects** - wherever the depth shows something standing up (trees,
   bushes, rocks, cliffs), a model is placed at that spot with its real size,
   its colours and its own piece of the painting.
5. **Backdrop** - everything beyond the terrain (sky, distant mountains, far
   castles) becomes a painted backdrop whose bottom edge lines up with the
   terrain's far edge from the start position.
6. A **colour grade** (tint, saturation, vignette) matches the painting's mood,
   and the player starts where the painting's camera was, looking in.

**World size** sets how deep the terrain goes (x2, so 4000 = 80 m),
**Terrain height** exaggerates or flattens the relief, and **Object size**
scales the eye height (and with it everything). After analyzing, the panel
says whether depth was used and where the horizon is. Without the depth
model, the tool still uses perspective and projection on a flat ground, with
plants and rocks guessed from the colours.

## Requirements

- Unreal Engine 5.8 with Python Editor Script Plugin, Editor Scripting
  Utilities and Landmass enabled (already set in the `.uproject`).
- System Python on PATH with `pygame`, `numpy`, `onnxruntime`, `tokenizers`, `huggingface_hub`.
- Blender (optional, recommended) for refining the generated models.
- Photoshop, only for `.psd` / `.psb` files (they are flattened through it first).
- No C++ or Visual Studio needed.

## Files

- `Content/Python/world_builder.py` - the Unreal tool: panel, terrain, water, props, lighting, Play-mode reactions.
- `Content/Python/init_unreal.py` - opens the panel when the project loads.
- `../reference_analyze.py` - reads the image/PSD and outputs palette, mood and a colour/brightness grid as JSON.
- `../reference_pick.py` - native file picker used by **Browse...**.
- `../reference_layers.py` - PSD layer export, name recognition, per-layer shape/colour analysis.
- `../rwb_layer_keywords.json` - the editable layer-name vocabulary.
- `Content/Python/rwb_layers.py` - builds Unreal objects, terrain and materials from the layers.
- `../reference_scene.py` - single-image depth, horizon/camera solving and 3D reconstruction.
- `Content/Python/rwb_scene.py` - builds the single-image world (projected terrain, objects, backdrop, grade).
- `../reference_objects.py` - recognition: OWLv2 + SlimSAM outlines + CLIP labels.
- `../reference_inventory.py` - what is in the picture (flat image or PSD), water shape, colours.
- `../reference_mesh.py` - builds a new mesh from each outline.
- `../reference_strokes.py` - reads the brushwork, stitches stroke maps, bakes the textures.
- `../reference_draw.py` - the Shape Sketch drawing window.
- `../reference_catalog.py` - searches Poly Haven, Objaverse and ModelLibrary for a described model.
- `../reference_surfaces.py` - reads the ground's surfaces and paints the terrain from them.
- `../reference_setting.py` - what kind of place the picture is (nature, city, interior...), night and neon lights.
- `../reference_facades.py` - is it a street? where each building along it starts and ends.
- `../reference_water.py` - checks every water candidate before it becomes water you can wade into.
- `Content/Python/rwb_street.py` - streets and rooms: building blocks, walls, sign boards, sign lights.
- `../rwb_progress.py` - lets helper scripts move the panel's progress bar.
- `../blender_refine.py` - runs inside Blender to finish each mesh.
- `Content/Python/rwb_layout.py` - the painting's camera: where everything goes (no `unreal`, testable).
- `Content/Python/rwb_generated.py` - imports the new meshes and paints them.
- `Content/Python/rwb_library.py` - imports and indexes the model library.
- `Content/Python/rwb_world.py` - the default builder: terrain, painted objects, extra copies.
- `../test_reference_world.py` - tests: `python test_reference_world.py [--live]`.
- `../ModelLibrary/` - drop model packs (zip / FBX / GLB) here.
- `../palette_sheet.py` - existing palette extraction reused by the analyzer.

## How it works (technical)

Unreal's bundled Python can't load pygame or show a file dialog, so the panel
shells out to the system Python for those two jobs and reads JSON back.
Terrain is a `ProceduralMeshComponent` with vertex colours sampled from the
image, using a generated vertex-colour material. Props are Kenney low-poly
nature meshes. A Slate post-tick callback watches Play-in-Editor and scales /
overlays props within the reaction radius, with a point light that follows the
player.

## Known limits

- One terrain size (40 x 40 vertices); very detailed images are simplified.
- You wade through water on the lake bed; there is no swimming.
- Reactions are visual only (glow, pulse, light) - no pickups or scoring yet.
