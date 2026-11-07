Reference World Builder - model library
=======================================

Drop 3D model packs here: .zip files, or loose .fbx / .glb / .gltf / .obj files.
The next time you press Generate world, new files are imported into Unreal
(Content/ReferenceWorldBuilder/Library) and matched to the objects in your
image BY FILE NAME, using the same words as rwb_layer_keywords.json
(mushroom_red -> mushroom, canoe -> boat, tree_pineTall -> pine, ...).

Good free (CC0) packs whose files are named after what they are:
  - Kenney Nature Kit   https://kenney.nl/assets/nature-kit   (trees, rocks, mushrooms, canoe, lily pads, stumps, fences...)
  - Kenney Fantasy Town Kit / Castle Kit (houses, towers, walls, lanterns, fountains)

A kenney_*.zip left in your Downloads folder is picked up too.
Files whose names contain a word from "library_ignore" (ground, path, tile...) are skipped.
