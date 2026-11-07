"""Build new 3D objects from painted outlines (Reference World Builder).

Every object in a painting has a cut-out (a SlimSAM outline or a PSD layer).  This
turns each one into a closed mesh whose silhouette, from the painting's camera, is
the shape that was painted, with the painting itself as its texture:

* **inflate** - organic things (trees, bushes, rocks): the silhouette is puffed out by
  its distance to the edge, so thick parts become thick and branches stay thin.
* **lathe** - round things (towers, mushrooms, lanterns, pillars): each row of the
  silhouette becomes a ring, so the object is truly round.
* **slab** - flat-fronted things (walls, gates, signs, people): the silhouette
  extruded with a soft bevel.
* **ground** - things lying down (boats, lily pads, bridges): the outline is projected
  through the camera onto the ground, then domed, because their height in the picture
  is really depth.
* **composite** - buildings: a solid core (round if the outline is symmetric) with thin limbs
  (arches, roots, branches) built separately and joined on, so one spur can't warp the whole.
* **cluster** - foliage and indistinct shapes: a mass of soft overlapping forms inside the
  outline.  Loose paint reads as loose paint instead of one lumpy solid.

The geometry runs on plain numpy.  If Blender is installed, ``blender_refine.py`` then makes each
mesh watertight, smooth and game-sized and unwraps it, and ``reference_strokes.bake`` paints the
whole surface: the painting where it faces the painter, your brushstrokes everywhere else.
Without Blender the raw mesh is exported as an OBJ with the painting projected from the front.

CLI (used by Unreal):
    python reference_mesh.py --objects objects.json --out DIR [--no-blender]
Prints the path of a manifest JSON listing each finished mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402

import rwb_progress  # noqa: E402

STAGE_GEOMETRY = 0.2       # share of this script's progress bar for each stage (measured)
STAGE_BLENDER = 0.65
GRID_LONG = 112          # silhouette resolution (long side, in cells)
GRID_LOW = 18            # ... with Model style at Low poly
TEXTURE_LONG = 1024      # art texture resolution (long side)
RING_SIDES = 28          # sides of a lathe ring
MESH_VERSION = "rwb-mesh-9"
HERE = Path(__file__).resolve().parent

LATHE = {"tower", "mushroom", "lantern", "pillar", "pine", "furniture"}
SLAB = {"wall", "fence", "gate", "prop_card", "character", "creature", "floating", "house", "castle",
        "building", "sign", "vehicle"}
GROUND = {"boat", "lily", "bridge"}
ARCHITECTURE = {"tower", "castle", "house", "building"}
BLOCKS = {"building"}         # flat walls, square corners and a stepped roofline: modelled as "block"
ROOF_STEP = 0.06              # roofline changes smaller than this share of the height are the same roof
FOLIAGE = {"bush", "thorns", "flower", "grass"}
# made things keep their own builder even with a rough outline: a lumpy sign is still flat
MADE = {"sign", "vehicle", "building", "character", "pillar", "lantern", "gate", "wall", "fence", "prop_card"}
ROUND_MIN_SYMMETRY = 0.68     # below this, spinning an outline makes a vase out of a lamp post
UNSURE_QUALITY = 0.68         # the outline model's own confidence
UNSURE_ROUGHNESS = 14.0       # a tangled outline (see reference_objects.roughness)
# depth of a slab as a share of its width; buildings are deep, signs are thin
SLAB_DEPTH = {"house": 0.45, "castle": 0.5, "wall": 0.14, "fence": 0.06, "gate": 0.14, "prop_card": 0.06,
              "character": 0.16, "creature": 0.22, "floating": 0.08, "building": 0.5, "sign": 0.05,
              "vehicle": 0.45}
# how far an inflated shape puffs out, relative to a perfectly round cross-section
INFLATE_DEPTH = {"tree": 0.85, "bush": 0.9, "rock": 0.8, "cliff": 0.7, "log": 0.9, "thorns": 0.5,
                 "flower": 0.55, "grass": 0.35}


def method_for(category, aspect):
    """Which builder suits a category.  ``aspect`` is height / width of the outline."""
    if category in GROUND:
        return "ground"
    if category in LATHE:
        # a "round" thing far wider than tall isn't one: treat it as a mound instead
        return "lathe" if aspect > 0.35 else "inflate"
    if category in SLAB:
        return "slab"
    return "inflate"


def symmetry(mask):
    """How well a shape matches its own mirror image about its central axis, 0..1."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.0
    axis = float(np.median(xs))
    mirrored = np.round(2.0 * axis - xs).astype(np.int64)
    inside = (mirrored >= 0) & (mirrored < mask.shape[1])
    matched = int(mask[ys[inside], mirrored[inside]].sum())
    return matched / float(len(xs))


def choose_method(obj, grid, aspect):
    """The builder for this object, from its outline as well as its label."""
    category = obj.get("category", "unknown")
    if obj.get("method"):
        return obj["method"]
    if category in GROUND or category == "drawn":
        return method_for(category, aspect)
    if category in BLOCKS:
        return "block"
    if category in ARCHITECTURE:
        return "composite"
    unsure = (obj.get("quality", 1.0) < UNSURE_QUALITY or obj.get("roughness", 0.0) > UNSURE_ROUGHNESS)
    if category in FOLIAGE or (unsure and category not in ({"tree", "pine", "log"} | MADE)):
        return "cluster"
    wanted = method_for(category, aspect)
    if wanted == "lathe" and symmetry(grid) < ROUND_MIN_SYMMETRY:
        return "slab"                       # labelled round, but its outline isn't
    return wanted


# --------------------------------------------------------------------------- #
# silhouettes
# --------------------------------------------------------------------------- #
def load_rgba(path):
    import pygame
    surf = pygame.image.load(str(path))
    w, h = surf.get_size()
    rgba = pygame.Surface((w, h), pygame.SRCALPHA, 32)
    rgba.blit(surf, (0, 0))
    return np.frombuffer(pygame.image.tobytes(rgba, "RGBA"), np.uint8).reshape(h, w, 4).copy()


def silhouette_grid(alpha, long_side=GRID_LONG, threshold=96):
    """Bool grid of the outline, rows top to bottom.  Thin parts survive the downsample."""
    H, W = alpha.shape
    scale = min(1.0, long_side / float(max(H, W)))
    gh, gw = max(2, int(round(H * scale))), max(2, int(round(W * scale)))
    rows = np.minimum((np.arange(H) * gh) // H, gh - 1)
    cols = np.minimum((np.arange(W) * gw) // W, gw - 1)
    grid = np.zeros((gh, gw), bool)
    on = alpha >= threshold
    # any painted pixel in a cell switches it on, so a 2 px fishing rod isn't lost
    ys, xs = np.nonzero(on)
    grid[rows[ys], cols[xs]] = True
    return grid


def distance_to_edge(mask):
    """Chamfer distance (in cells) from each on-cell to the nearest off-cell or border."""
    h, w = mask.shape
    d = np.where(mask, 1e6, 0.0)
    diag = math.sqrt(2.0)
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            best = 1.0 if (y == 0 or x == 0 or y == h - 1 or x == w - 1) else d[y, x]
            if y > 0:
                best = min(best, d[y - 1, x] + 1.0)
                if x > 0:
                    best = min(best, d[y - 1, x - 1] + diag)
                if x < w - 1:
                    best = min(best, d[y - 1, x + 1] + diag)
            if x > 0:
                best = min(best, d[y, x - 1] + 1.0)
            d[y, x] = best
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if not mask[y, x]:
                continue
            best = d[y, x]
            if y < h - 1:
                best = min(best, d[y + 1, x] + 1.0)
                if x > 0:
                    best = min(best, d[y + 1, x - 1] + diag)
                if x < w - 1:
                    best = min(best, d[y + 1, x + 1] + diag)
            if x < w - 1:
                best = min(best, d[y, x + 1] + 1.0)
            d[y, x] = best
    return d


def _max_filter(a, radius):
    """Square maximum filter (numpy only)."""
    out = a.copy()
    for _ in range(max(0, int(radius))):
        pad = np.pad(out, 1, mode="edge")
        out = np.max(np.stack([pad[1:-1, 1:-1], pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]]), axis=0)
    return out


def _box_blur(a, passes=2):
    out = a.astype(np.float64)
    for _ in range(passes):
        pad = np.pad(out, 1, mode="edge")
        out = sum(pad[1 + dy:1 + dy + out.shape[0], 1 + dx:1 + dx + out.shape[1]]
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    return out


# --------------------------------------------------------------------------- #
# closed surfaces from a grid
# --------------------------------------------------------------------------- #
def _shell(mask, front, back):
    """Close a two-sided surface over a mask.

    ``front`` / ``back`` map (row, col) -> xyz.  Returns (verts, faces) with the front
    surface's normal facing the viewer; faces are vertex-index quads.
    """
    h, w = mask.shape
    index = -np.ones((h, w), np.int64)
    rows, cols = np.nonzero(mask)
    index[rows, cols] = np.arange(len(rows))
    n = len(rows)
    verts = [front(r, c) for r, c in zip(rows, cols)] + [back(r, c) for r, c in zip(rows, cols)]
    faces = []
    edges = {}
    for r in range(h - 1):
        for c in range(w - 1):
            if not (mask[r, c] and mask[r, c + 1] and mask[r + 1, c] and mask[r + 1, c + 1]):
                continue
            tl, tr = int(index[r, c]), int(index[r, c + 1])
            bl, br = int(index[r + 1, c]), int(index[r + 1, c + 1])
            # rows grow downward, so tl -> bl -> br -> tr has its normal toward -y
            faces.append((tl, bl, br, tr))
            faces.append((tl + n, tr + n, br + n, bl + n))
            for a, b in ((tl, bl), (bl, br), (br, tr), (tr, tl)):
                edges[(a, b)] = edges.get((a, b), 0) + 1
    # edges with no opposite are the rim: stitch front to back there
    for (a, b) in list(edges):
        if (b, a) not in edges:
            faces.append((b, a, a + n, b + n))
    return np.array(verts, np.float64), faces


def _thicken_thin(mask):
    """Quads need 2x2 blocks: grow one-cell-wide lines so they aren't dropped from the mesh."""
    h, w = mask.shape
    block = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
    covered = np.zeros_like(mask)
    covered[:-1, :-1] |= block
    covered[1:, :-1] |= block
    covered[:-1, 1:] |= block
    covered[1:, 1:] |= block
    lonely = mask & ~covered
    if not lonely.any():
        return mask
    out = mask.copy()
    rows, cols = np.nonzero(lonely)
    out[rows, np.minimum(cols + 1, w - 1)] = True
    out[np.minimum(rows + 1, h - 1), cols] = True
    out[np.minimum(rows + 1, h - 1), np.minimum(cols + 1, w - 1)] = True
    return out


def _pad(mask):
    return np.pad(mask, 1, mode="constant", constant_values=False)


def inflate(mask, depth=0.85, relief=None):
    """Puffed-out silhouette.  x = right, z = up, -y = toward the viewer, in cells.

    Right-handed like Blender: Unreal's importer flips Y, so the front (-y here) ends up
    facing +Y in Unreal, where the player stands, and the picture isn't mirrored.
    """
    grid = _thicken_thin(_pad(mask))
    dist = distance_to_edge(grid)
    peak = max(1.0, float(dist.max()))
    # compare with the local thickness, so branches stay thin next to a fat trunk
    local = np.maximum(_max_filter(dist, max(2, int(peak * 0.35))), dist)
    round_profile = np.sqrt(np.maximum(0.0, dist * (2.0 * local - dist)))
    bulge = _box_blur(round_profile, 1) * depth
    h, w = grid.shape
    front_h = bulge.copy()
    if relief is not None:
        # what the depth map says is nearer pushes out; relief is -1..1 at grid size
        pad_relief = np.pad(relief, 1, mode="edge")
        front_h = bulge * (1.0 + 0.45 * pad_relief)
    return _shell(grid,
                  lambda r, c: (float(c), -float(front_h[r, c]) - 0.05, float(h - 1 - r)),
                  lambda r, c: (float(c), float(bulge[r, c]) + 0.05, float(h - 1 - r)))


def slab(mask, depth_ratio=0.15):
    """Silhouette extruded to a set depth, with a soft bevel so it isn't a cookie cutter."""
    grid = _thicken_thin(_pad(mask))
    h, w = grid.shape
    columns = np.nonzero(grid.any(0))[0]
    width = float(columns.max() - columns.min() + 1)
    half = max(0.6, width * depth_ratio * 0.5)
    dist = distance_to_edge(grid)
    profile = np.minimum(half, half * 0.45 + np.maximum(0.0, dist - 1.0) * 0.9)
    return _shell(grid,
                  lambda r, c: (float(c), -float(profile[r, c]), float(h - 1 - r)),
                  lambda r, c: (float(c), float(profile[r, c]), float(h - 1 - r)))


def block_outline(mask, step=ROOF_STEP):
    """A building's outline squared off: flat ground line, straight sides, a roofline in level steps."""
    h, w = mask.shape
    cols = np.nonzero(mask.any(0))[0]
    out = np.zeros_like(mask, bool)
    if len(cols) < 2:
        return mask.copy()
    c0, c1 = int(cols.min()), int(cols.max())
    tops = np.array([int(np.argmax(mask[:, c])) if mask[:, c].any() else h for c in range(c0, c1 + 1)], np.float64)
    bottom = int(max(np.nonzero(mask[:, c])[0].max() for c in cols))
    # ignore spikes (an aerial, a sign over the roof) narrower than a twelfth of the width
    k = max(1, (c1 - c0 + 1) // 24)
    padded = np.pad(tops, k, mode="edge")
    tops = np.array([np.median(padded[i:i + 2 * k + 1]) for i in range(len(tops))])
    height = max(1.0, bottom - tops.min())
    steps = []
    start = 0
    for i in range(1, len(tops) + 1):
        if i == len(tops) or abs(tops[i] - np.median(tops[start:i])) > step * height:
            steps.append((start, i, int(round(np.median(tops[start:i])))))
            start = i
    for a, b, top in steps:
        out[min(top, bottom):bottom + 1, c0 + a:c0 + b] = True
    return out


def block(mask, depth_ratio=0.5):
    """A building: its squared-off outline extruded straight back, with sharp corners (no bevel)."""
    grid = _pad(block_outline(mask))
    h, w = grid.shape
    columns = np.nonzero(grid.any(0))[0]
    width = float(columns.max() - columns.min() + 1)
    half = max(1.0, width * depth_ratio * 0.5)
    return _shell(grid,
                  lambda r, c: (float(c), -half, float(h - 1 - r)),
                  lambda r, c: (float(c), half, float(h - 1 - r)))


def lathe(mask, sides=RING_SIDES):
    """Surface of revolution from each row's width; the axis follows the shape's lean."""
    h, _w = mask.shape
    rows = [r for r in range(h) if mask[r].any()]
    left = np.array([np.nonzero(mask[r])[0].min() for r in rows], np.float64)
    right = np.array([np.nonzero(mask[r])[0].max() + 1 for r in rows], np.float64)
    radius = _smooth_1d(np.maximum(0.5, (right - left) / 2.0), 1)
    centre = _smooth_1d((left + right) / 2.0, 3)
    heights = np.array([float(h - r) for r in rows])
    verts = []
    for i in range(len(rows)):
        for k in range(sides):
            a = 2.0 * math.pi * k / sides
            verts.append((centre[i] + math.cos(a) * radius[i], math.sin(a) * radius[i], heights[i]))
    n_rings = len(rows)
    faces = []
    for i in range(n_rings - 1):
        for k in range(sides):
            a = i * sides + k
            b = i * sides + (k + 1) % sides
            c = (i + 1) * sides + (k + 1) % sides
            d = (i + 1) * sides + k
            # row i is above row i+1
            faces.append((a, d, c, b))
    top = len(verts)
    verts.append((centre[0], 0.0, heights[0] + radius[0] * 0.12))
    bottom = len(verts)
    verts.append((centre[-1], 0.0, heights[-1]))
    last = (n_rings - 1) * sides
    for k in range(sides):
        faces.append((top, k, (k + 1) % sides))
        faces.append((bottom, last + (k + 1) % sides, last + k))
    return np.array(verts, np.float64), faces


def _join(meshes):
    """One vertex list + one face list from several (verts, faces) pairs."""
    all_verts, all_faces, offset = [], [], 0
    for verts, faces in meshes:
        all_verts.append(verts)
        all_faces.extend(tuple(i + offset for i in face) for face in faces)
        offset += len(verts)
    return np.concatenate(all_verts), all_faces


def split_core(mask):
    """(the shape's solid body, [thin limbs]) by removing everything too narrow to be body."""
    import reference_scene
    dist = distance_to_edge(mask)
    radius = max(2, int(float(dist.max()) * 0.45))
    seeds = dist >= radius
    if not seeds.any():
        return mask, []
    opened = _max_filter(seeds.astype(np.float64), radius) > 0.0
    opened &= mask
    parts = reference_scene._components(opened)
    largest = max(parts, key=len)
    core = np.zeros_like(mask)
    rows, cols = zip(*largest)
    core[list(rows), list(cols)] = True
    around = _max_filter(core.astype(np.float64), 1) > 0.0
    limbs = []
    for cells in reference_scene._components(mask & ~around):
        if len(cells) < max(6, mask.sum() * 0.004):
            continue
        limb = np.zeros_like(mask)
        rows, cols = zip(*cells)
        limb[list(rows), list(cols)] = True
        # overlap the body a little, so the remesh welds them instead of leaving a gap
        limbs.append((_max_filter(limb.astype(np.float64), 2) > 0.0) & mask)
    return core, limbs


def composite(mask, depth_ratio=0.45):
    """A building: solid core, round when the outline allows, with limbs joined on."""
    core, limbs = split_core(mask)
    if symmetry(core) >= ROUND_MIN_SYMMETRY:
        meshes = [lathe(core)]
    else:
        meshes = [_centred(slab(core, depth_ratio))]
    meshes.extend(_centred(inflate(limb, 0.6)) for limb in limbs)
    return _join(meshes)


def _centred(mesh):
    """Line a padded-grid mesh up with lathe()'s coordinates."""
    verts, faces = mesh
    return verts + np.array([-0.5, 0.0, 0.0]), faces


def _ball(cx, cy, cz, radius, rows=8, sides=12):
    """A closed sphere, faces outward."""
    verts = [(cx, cy, cz + radius)]
    for i in range(1, rows):
        polar = math.pi * i / rows
        for k in range(sides):
            a = 2.0 * math.pi * k / sides
            verts.append((cx + radius * math.sin(polar) * math.cos(a),
                          cy + radius * math.sin(polar) * math.sin(a), cz + radius * math.cos(polar)))
    verts.append((cx, cy, cz - radius))
    faces = []
    for k in range(sides):
        faces.append((0, 1 + k, 1 + (k + 1) % sides))
        last = 1 + (rows - 2) * sides
        faces.append((len(verts) - 1, last + (k + 1) % sides, last + k))
    for i in range(rows - 2):
        for k in range(sides):
            a, b = 1 + i * sides + k, 1 + i * sides + (k + 1) % sides
            c, d = a + sides, b + sides
            faces.append((a, c, d, b))
    return np.array(verts, np.float64), faces


def cluster(mask, seed=1, max_balls=60):
    """Soft overlapping forms packed inside the outline, on a thin base that keeps its edge."""
    grid = _thicken_thin(_pad(mask))
    dist = distance_to_edge(grid)
    peak = max(1.0, float(dist.max()))
    h, w = grid.shape
    rng = np.random.default_rng(seed)
    remaining = dist.copy()
    meshes = [inflate(mask, 0.3)]
    ys, xs = np.mgrid[0:h, 0:w]
    for _ in range(max_balls):
        spot = int(np.argmax(remaining))
        row, col = spot // w, spot % w
        radius = float(dist[row, col])
        if remaining[row, col] < max(1.2, peak * 0.14):
            break
        # forms sit at different depths: that's what makes a mass read as volume, not a pillow
        depth = float(rng.uniform(-0.55, 0.55)) * radius
        meshes.append(_ball(float(col), depth, float(h - 1 - row), radius * 1.05))
        remaining[(xs - col) ** 2 + (ys - row) ** 2 < (radius * 0.85) ** 2] = 0.0
    return _join(meshes)


def _smooth_1d(values, passes):
    out = values.astype(np.float64)
    for _ in range(passes):
        pad = np.pad(out, 1, mode="edge")
        out = (pad[:-2] + 2.0 * pad[1:-1] + pad[2:]) / 4.0
    return out


def ground_shape(mask, bbox, camera):
    """Outline projected through the camera to the ground, then domed.  Units: camera heights.

    ``bbox`` = the outline in work pixels; camera = {W, H, horizon, focal}.  x = right,
    y = distance from the camera, z = up (right-handed; Unreal's import flips Y so distance
    runs into the picture).  Returns (verts, faces).
    """
    grid = _thicken_thin(_pad(mask))
    gh, gw = grid.shape
    x0, y0, x1, y1 = bbox
    focal, horizon, W = camera["focal"], camera["horizon"], camera["W"]
    # the two padding cells sit just outside the outline
    px_x = x0 + (np.arange(gw) - 0.5) * (x1 - x0) / max(1.0, gw - 2.0)
    px_y = y0 + (np.arange(gh) - 0.5) * (y1 - y0) / max(1.0, gh - 2.0)
    below = np.maximum(px_y - horizon, max(2.0, 0.01 * camera["H"]))
    dist = focal / below                                    # distance per row, in camera heights
    cols = np.nonzero(grid.any(0))[0]
    rows = np.nonzero(grid.any(1))[0]
    footprint_w = (px_x[cols.max()] - px_x[cols.min()]) * float(np.median(dist[rows])) / focal
    edge = distance_to_edge(grid)
    peak = max(1.0, float(edge.max()))
    dome = 1.0 - (1.0 - np.minimum(1.0, edge / peak)) ** 2
    rise = min(0.9, max(0.04, footprint_w * 0.16))         # a shallow form, not a loaf
    ground_x = lambda r, c: (px_x[c] - W / 2.0) * dist[r] / focal
    return _shell(grid,
                  lambda r, c: (ground_x(r, c), float(dist[r]), float(dome[r, c] * rise)),
                  lambda r, c: (ground_x(r, c), float(dist[r]), 0.0))


# --------------------------------------------------------------------------- #
# mapping and export
# --------------------------------------------------------------------------- #
def art_uv(verts, method, projection):
    """UV of each vertex: where it came from in the object's picture."""
    if method == "ground":
        focal, horizon, W = projection["focal"], projection["horizon"], projection["W"]
        x0, y0, x1, y1 = projection["bbox"]
        dist = np.maximum(verts[:, 1], 1e-3)
        px = W / 2.0 + verts[:, 0] * focal / dist
        py = horizon + focal / dist
        return np.stack([(px - x0) / max(1.0, x1 - x0), (py - y0) / max(1.0, y1 - y0)], -1)
    return np.stack([(verts[:, 0] - projection["x_min"]) / max(1e-6, projection["x_size"]),
                     1.0 - (verts[:, 2] - projection["z_min"]) / max(1e-6, projection["z_size"])], -1)


def centre_base(verts, method):
    """Move the pivot to the middle of the base and return the shifted verts + the offset."""
    if method == "ground":
        offset = np.array([(verts[:, 0].min() + verts[:, 0].max()) / 2.0,
                           (verts[:, 1].min() + verts[:, 1].max()) / 2.0, 0.0])
    else:
        offset = np.array([(verts[:, 0].min() + verts[:, 0].max()) / 2.0, 0.0, verts[:, 2].min()])
    return verts - offset, offset


def save_art(rgba, path, long_side=TEXTURE_LONG):
    """Opaque texture: colours are bled past the outline so edges never show the background."""
    import pygame
    h, w = rgba.shape[:2]
    scale = min(1.0, long_side / float(max(h, w)))
    if scale < 1.0:
        nw, nh = max(2, int(w * scale)), max(2, int(h * scale))
        surf = pygame.image.frombuffer(rgba.tobytes(), (w, h), "RGBA")
        surf = pygame.transform.smoothscale(surf, (nw, nh))
        rgba = np.frombuffer(pygame.image.tobytes(surf, "RGBA"), np.uint8).reshape(nh, nw, 4).copy()
        h, w = nh, nw
    known = rgba[..., 3] >= 128
    out = rgba[..., :3].astype(np.float64)
    if known.any():
        out[~known] = 0.0
        filled = known.copy()
        for _ in range(48):
            if filled.all():
                break
            total = np.zeros_like(out)
            count = np.zeros(filled.shape)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                shift_f = np.zeros_like(filled)
                shift_c = np.zeros_like(out)
                ys = slice(max(dy, 0), h + min(dy, 0))
                yd = slice(max(-dy, 0), h + min(-dy, 0))
                xs = slice(max(dx, 0), w + min(dx, 0))
                xd = slice(max(-dx, 0), w + min(-dx, 0))
                shift_f[ys, xs] = filled[yd, xd]
                shift_c[ys, xs] = out[yd, xd]
                total += shift_c * shift_f[..., None]
                count += shift_f
            new = (~filled) & (count > 0)
            out[new] = total[new] / count[new][:, None]
            filled |= new
        out[~filled] = rgba[..., :3][known].mean(0)
    surf = pygame.image.frombuffer(np.clip(out, 0, 255).astype(np.uint8).tobytes(), (w, h), "RGB")
    pygame.image.save(surf, str(path))
    return str(path)


def save_thumb(rgba, path, size=96):
    """A small picture of the object for the panel's model list."""
    import pygame
    h, w = rgba.shape[:2]
    scale = size / float(max(h, w))
    surf = pygame.image.frombuffer(rgba.tobytes(), (w, h), "RGBA")
    surf = pygame.transform.smoothscale(surf, (max(1, int(w * scale)), max(1, int(h * scale))))
    pygame.image.save(surf, str(path))
    return str(path)


def bake_entry(entry, strokes, strength):
    """Paint a refined mesh's whole surface.  Returns the texture path."""
    import pygame
    import reference_strokes as rs
    data = np.load(entry["surface"])
    surface = {key: data[key] for key in ("positions", "normals", "tris", "tri_uvs")}
    rgba = load_rgba(entry["cutout"])
    art_h, art_w = rgba.shape[:2]
    art_surf = pygame.image.load(entry["front_art"])
    front = pygame.Surface(art_surf.get_size(), 0, 24)
    front.blit(art_surf, (0, 0))
    fw, fh = front.get_size()
    art = np.frombuffer(pygame.image.tobytes(front, "RGB"), np.uint8).reshape(fh, fw, 3)
    method, projection = entry["method"], dict(entry["projection"])
    if projection.get("auto"):
        # a found model: the painting spans whatever the fitted model spans from the front
        pos = surface["positions"]
        projection.update({"x_min": float(pos[:, 0].min()), "x_size": float(np.ptp(pos[:, 0])),
                           "z_min": float(pos[:, 2].min()), "z_size": float(np.ptp(pos[:, 2]))})
        entry["size"] = [float(np.ptp(pos[:, i])) for i in range(3)]
    offset = np.array(projection.get("offset", (0.0, 0.0, 0.0)))

    def uv_of(positions):
        return art_uv(positions + offset, method, projection)

    # the object's own marks if it has enough paint for them, else the whole painting's
    stats = (strokes or {}).get("stats") or {}
    inside = rs.blur((rgba[..., 3] >= 128).astype(np.float32), 3) > 0.98
    detail = None
    if inside.sum() > 96 * 96:
        detail = rs.stroke_map(rgba[..., :3], stats or None, inside, size=256, seed=len(entry["name"]))
    if detail is None and (strokes or {}).get("map") and Path(strokes["map"]).is_file():
        detail = np.load(strokes["map"]).astype(np.float32)
    per_unit = art_w / max(1e-6, float(entry["size"][0]))
    size = 256 if max(art_w, art_h) < 200 else 512 if max(art_w, art_h) < 600 else 1024
    rgb, direct = rs.bake(surface, art, detail, uv_of, method, per_unit, strength, size,
                          wrap_art=bool(entry.get("wrap_art")))
    path = Path(entry["front_art"]).with_name(entry["name"] + "_wrap.png")
    pygame.image.save(pygame.image.frombuffer(rgb.tobytes(), (size, size), "RGB"), str(path))
    entry["painted_share"] = round(direct, 3)
    return str(path)


def _bake_job(job):
    """One bake, safe to run in a worker process.  Returns (path, painted share, error)."""
    entry, strokes, strength = job
    try:
        path = bake_entry(entry, strokes, strength)
        return path, entry.get("painted_share"), None
    except Exception as error:
        return None, None, str(error)[:200]


def bake_all(entries, strokes, strength, workers=1, span=(0.0, 1.0)):
    """Bake every refined mesh.  A failure falls back to that mesh's front projection."""
    todo = [e for e in entries if e.get("surface") and Path(e["surface"]).is_file() and not e.get("own_look")]

    def tick(finished):
        rwb_progress.report(span[0] + (span[1] - span[0]) * finished / float(max(1, len(todo))),
                            "Wrapping brushstrokes around %d of %d models" % (finished, len(todo)))

    results = _in_parallel(_bake_job, [(e, strokes, strength) for e in todo], workers, tick)
    done = failed = 0
    for entry, (path, share, error) in zip(todo, results):
        if error:
            entry["art"], entry["mesh"], entry["refined"] = entry["front_art"], entry["raw_mesh"], False
            entry["bake_error"] = error
            failed += 1
        else:
            entry["art"], entry["painted_share"] = path, share
            done += 1
    return done, failed


def write_obj(verts, faces, uvs, path):
    with open(path, "w", encoding="utf-8") as out:
        out.write("# Reference World Builder generated mesh\n")
        for v in verts:
            out.write("v %.5f %.5f %.5f\n" % (v[0], v[1], v[2]))
        for uv in uvs:
            out.write("vt %.5f %.5f\n" % (uv[0], 1.0 - uv[1]))
        for face in faces:
            out.write("f " + " ".join("%d/%d" % (i + 1, i + 1) for i in face) + "\n")
    return str(path)


def resolution(detail, low, high):
    """Between low (detail 0) and high (detail 1), even-feeling across the slider."""
    detail = max(0.0, min(1.0, float(detail)))
    return low * (high / float(low)) ** detail


def adopt_object(obj, out_dir, camera, index):
    """An entry that starts from a found model (reference_catalog) instead of the painted outline.

    Blender fits it to the painted proportions; it is then painted like any other model, unless
    ``own_look`` asks to keep the model's own colours.
    """
    import reference_catalog
    rgba = load_rgba(obj["cutout"])
    h, w = rgba.shape[:2]
    source = reference_catalog.download(obj["source"])
    reference_catalog.credit(obj["source"])
    name = obj.get("name") or "obj_%02d" % index
    art = save_art(rgba, Path(out_dir) / (name + "_art.png"))
    thumb = save_thumb(rgba, Path(out_dir) / (name + "_thumb.png"))
    return {
        "id": obj.get("id", name), "name": name, "label": obj.get("label", obj.get("category")),
        "category": obj.get("category", "unknown"), "method": "adopt", "source_file": source,
        "source_name": obj["source"].get("name"), "own_look": bool(obj.get("own_look")),
        "mesh": source, "raw_mesh": source, "art": art, "front_art": art, "thumb": thumb,
        "cutout": obj["cutout"], "cut_size": [w, h], "projection": {"auto": True},
        "target_aspect": h / float(max(1, w)), "fit": float(obj.get("fit", 0.6)),
        "wrap_art": False, "detail": float(obj.get("detail", 1.0)),
        "size": [float(w), float(w) * 0.6, float(h)],
        "budget": int(max(1500, min(14000, 1500 + 60000 * float(obj.get("area", 0.05))))),
        "tris_raw": 0,
    }


def build_object(obj, out_dir, camera, index):
    """Raw mesh + art for one object.  Returns the manifest entry (or None if unusable)."""
    if obj.get("source"):
        return adopt_object(obj, out_dir, camera, index)
    rgba = load_rgba(obj["cutout"])
    h, w = rgba.shape[:2]
    if h < 3 or w < 3 or (rgba[..., 3] >= 96).sum() < 20:
        return None
    detail = float(obj.get("detail", 1.0))
    grid = silhouette_grid(rgba[..., 3], int(resolution(detail, GRID_LOW, GRID_LONG)))
    if grid.sum() < 8:
        return None
    category = obj.get("category", "unknown")
    method = choose_method(obj, grid, h / float(max(1, w)))
    if method == "ground" and (not obj.get("bbox") or obj["bbox"][1] < camera["horizon"] + 0.03 * camera["H"]):
        method = "slab"                 # it crosses the horizon, so it is standing up
    relief = None
    if obj.get("relief") is not None:
        relief = np.asarray(obj["relief"], np.float64)
        if relief.shape != grid.shape:
            import reference_scene
            relief = reference_scene._resize(relief, grid.shape[1], grid.shape[0]).astype(np.float64)
    if method == "ground":
        bbox = obj["bbox"]
        verts, faces = ground_shape(grid, bbox, camera)
        projection = {"focal": camera["focal"], "horizon": camera["horizon"], "W": camera["W"], "bbox": list(bbox)}
    elif method == "lathe":
        verts, faces = lathe(grid)
    elif method == "composite":
        verts, faces = composite(grid, SLAB_DEPTH.get(category, 0.45))
    elif method == "block":
        verts, faces = block(grid, SLAB_DEPTH.get(category, 0.5))
    elif method == "cluster":
        verts, faces = cluster(grid, seed=index + 1)
    elif method == "slab":
        verts, faces = slab(grid, SLAB_DEPTH.get(category, 0.15))
    else:
        verts, faces = inflate(grid, INFLATE_DEPTH.get(category, 0.8), relief)
    if method != "ground":
        # standing objects use the silhouette's full extent as the picture's extent
        extent_x = (float(verts[:, 0].min()), float(verts[:, 0].max()))
        extent_z = (float(verts[:, 2].min()), float(verts[:, 2].max()))
        projection = {"x_min": extent_x[0], "x_size": extent_x[1] - extent_x[0],
                      "z_min": extent_z[0], "z_size": extent_z[1] - extent_z[0]}
    uvs = art_uv(verts, method, projection)
    verts, offset = centre_base(verts, method)
    # art_uv is recomputed in Blender on the new surface, so express the projection in moved coords
    if method == "ground":
        projection["offset"] = [float(v) for v in offset]
    else:
        projection["x_min"] -= float(offset[0])
        projection["z_min"] -= float(offset[2])
    name = obj.get("name") or "obj_%02d" % index
    art = save_art(rgba, Path(out_dir) / (name + "_art.png"))
    thumb = save_thumb(rgba, Path(out_dir) / (name + "_thumb.png"))
    raw = write_obj(verts, faces, np.clip(uvs, 0.0, 1.0), Path(out_dir) / (name + ".obj"))
    return {
        "id": obj.get("id", name), "name": name, "label": obj.get("label", category), "category": category,
        "method": method, "mesh": raw, "raw_mesh": raw, "art": art, "front_art": art, "thumb": thumb,
        "cutout": obj["cutout"], "cut_size": [w, h], "projection": projection,
        "wrap_art": category in ARCHITECTURE, "detail": detail,
        "size": [float(np.ptp(verts[:, i])) for i in range(3)],
        "budget": int(max(600, min(14000 if category in ARCHITECTURE else 9000,
                                   600 + 60000 * float(obj.get("area", 0.05))))),
        "tris_raw": sum(len(f) - 2 for f in faces),
    }


# --------------------------------------------------------------------------- #
# Blender
# --------------------------------------------------------------------------- #
def find_blender():
    """Path to blender.exe, newest first, or None."""
    env = os.environ.get("BLENDER_EXE")
    if env and Path(env).is_file():
        return env
    found = shutil.which("blender")
    if found:
        return found
    candidates = []
    for base in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"), os.environ.get("LOCALAPPDATA")):
        if base:
            candidates += list(Path(base).glob("Blender Foundation/Blender */blender.exe"))
    candidates += list(Path("/Applications").glob("Blender*.app/Contents/MacOS/Blender"))
    candidates += [Path(p) for p in ("/usr/bin/blender", "/snap/bin/blender")]
    existing = [c for c in candidates if c.is_file()]
    if not existing:
        return None
    return str(sorted(existing, key=lambda p: str(p), reverse=True)[0])


def worker_count():
    """Half the logical cores, at most 4: Blender's remesh and numpy use several threads each."""
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def _batches(entries, count):
    """Split models between Blender processes, so each gets a similar amount of triangles."""
    groups = [[] for _ in range(count)]
    loads = [0] * count
    for entry in sorted(entries, key=lambda e: -e.get("budget", 1000)):
        i = loads.index(min(loads))
        groups[i].append(entry)
        loads[i] += entry.get("budget", 1000)
    return [g for g in groups if g]


def refine_with_blender(entries, out_dir, blender=None, timeout=900, workers=1, span=(0.0, 1.0)):
    """Run blender_refine.py over the raw meshes, ``workers`` Blenders at once.  Returns (done, message)."""
    blender = blender or find_blender()
    if not blender:
        return False, "Blender not found: using the raw meshes (install Blender for watertight, smoother models)"
    out_dir = Path(out_dir)
    script = HERE / "blender_refine.py"
    creation = 0x08000000 if os.name == "nt" else 0
    running = []
    try:
        for number, group in enumerate(_batches(entries, max(1, workers))):
            jobs = out_dir / ("blender_jobs_%d.json" % number)
            jobs.write_text(json.dumps(group), encoding="utf-8")
            log = open(out_dir / ("blender_log_%d.txt" % number), "wb")
            running.append((subprocess.Popen(
                [blender, "--background", "--factory-startup", "--python-exit-code", "1",
                 "--python", str(script), "--", str(jobs)],
                stdout=log, stderr=subprocess.STDOUT, creationflags=creation), log, number))
    except OSError as error:
        for process, log, _number in running:
            process.kill()
            log.close()
        return False, "Blender could not run (%s): using the raw meshes" % str(error).splitlines()[0][:120]
    started = time.time()
    total = len(entries)
    try:
        while any(process.poll() is None for process, _log, _number in running):
            if time.time() - started > timeout:
                raise TimeoutError()
            finished = sum(1 for e in entries if Path(e["raw_mesh"]).with_suffix(".fbx").is_file())
            rwb_progress.report(span[0] + (span[1] - span[0]) * finished / float(max(1, total)),
                                "Blender finished %d of %d models" % (finished, total))
            time.sleep(0.25)
    except TimeoutError:
        for process, _log, _number in running:
            process.kill()
        return False, "Blender took too long: using the raw meshes"
    finally:
        for _process, log, _number in running:
            log.close()
    refined, failures = 0, []
    done = {}
    for process, _log, number in running:
        report = out_dir / ("blender_report_%d.json" % number)
        if process.returncode == 0 and report.is_file():
            done.update(json.loads(report.read_text(encoding="utf-8")))
        else:
            text = (out_dir / ("blender_log_%d.txt" % number)).read_text(encoding="utf-8", errors="replace")
            tail = [ln for ln in text.splitlines() if ln.strip()][-1:]
            failures.append(tail[0][:160] if tail else "no output")
    for entry in entries:
        item = done.get(entry["name"])
        if entry["method"] == "adopt" and not (item and item.get("mesh")):
            entry["adopt_error"] = (item or {}).get("error", "Blender could not read it")
        if item and item.get("mesh") and Path(item["mesh"]).is_file():
            entry["mesh"] = item["mesh"]
            entry["surface"] = item.get("surface")
            entry["tris"] = item.get("tris")
            if item.get("size"):
                entry["size"] = item["size"]
            entry["refined"] = True
            refined += 1
    if not refined and failures:
        return False, "Blender failed (%s): using the raw meshes" % failures[0]
    return True, "Blender refined %d of %d models" % (refined, len(entries))


def _in_parallel(function, jobs, workers, on_done=None):
    """function(job) over jobs, in order, using processes when that's worth it."""
    results = [None] * len(jobs)
    if workers <= 1 or len(jobs) < 3:
        for i, job in enumerate(jobs):
            results[i] = function(job)
            if on_done:
                on_done(i + 1)
        return results
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, job): i for i, job in enumerate(jobs)}
        for finished, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if on_done:
                on_done(finished)
    return results


def _finish(manifest, manifest_path, strokes, strength, workers=1):
    """Bake (or rebake after a strength change, without redoing any geometry)."""
    bake_key = json.dumps([round(float(strength), 3), (strokes or {}).get("map")])
    if manifest.get("bake_key") != bake_key:
        done, failed = bake_all(manifest["models"], strokes, strength, workers, span=(STAGE_BLENDER, 1.0))
        manifest["bake_key"] = bake_key
        manifest["bake_message"] = "brushstrokes wrapped around %d models%s" % (
            done, " (%d kept the front projection only)" % failed if failed else "") if done or failed else ""
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _build_job(job):
    """One model's raw mesh, safe to run in a worker process.  Returns (entry, error)."""
    obj, out_dir, camera, index = job
    try:
        return build_object(obj, out_dir, camera, index), None
    except Exception as error:
        return None, "%s: %s" % (obj.get("label", index), str(error)[:120])


def build_all(objects, out_dir, camera, use_blender=True, strokes=None, strength=1.0, workers=1):
    """Everything for a list of objects, cached by their cut-outs.  Returns the manifest dict.

    workers > 1 uses extra processes, so the script calling this needs an ``if __name__`` guard.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = hashlib.md5()
    stamp.update(json.dumps([MESH_VERSION, camera, use_blender, bool(find_blender()) if use_blender else False],
                            sort_keys=True).encode())
    for obj in objects:
        st = os.stat(obj["cutout"])
        stamp.update(("%s|%s|%s|%s|%s|%s|%s|%s|%d|%d" % (obj["cutout"], obj.get("category"), obj.get("method"),
                                                 obj.get("quality"), obj.get("roughness"), obj.get("detail", 1.0),
                                                 json.dumps(obj.get("source"), sort_keys=True), obj.get("own_look"),
                                                 st.st_size, int(st.st_mtime))).encode())
    key = stamp.hexdigest()
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached.get("key") == key and all(Path(e["raw_mesh"]).is_file() for e in cached["models"]):
                return _finish(cached, manifest_path, strokes, strength, workers)
        except Exception:
            pass
    for old in list(out_dir.glob("obj_*")) + list(out_dir.glob("drawn_*")) + list(out_dir.glob("blender_*")):
        old.unlink()
    entries, skipped = [], []

    def built(finished):
        rwb_progress.report(STAGE_GEOMETRY * finished / float(max(1, len(objects))),
                            "Shaping %d of %d models" % (finished, len(objects)))

    jobs = [(obj, str(out_dir), camera, index) for index, obj in enumerate(objects)]
    for obj, (entry, error) in zip(objects, _in_parallel(_build_job, jobs, workers, built)):
        if error:
            skipped.append(error)
        elif entry:
            entries.append(entry)
        else:
            skipped.append("%s: outline too small" % obj.get("label", "model"))
    if any(e["method"] == "adopt" for e in entries) and not (use_blender and find_blender()):
        skipped.extend("%s: a found model needs Blender to fit it" % e["label"] for e in entries if e["method"] == "adopt")
        entries = [e for e in entries if e["method"] != "adopt"]
    message = "Blender step turned off"
    if use_blender and entries:
        _ok, message = refine_with_blender(entries, out_dir, workers=workers, span=(STAGE_GEOMETRY, STAGE_BLENDER))
    for entry in [e for e in entries if e.get("adopt_error")]:
        skipped.append("%s: couldn't use that model (%s)" % (entry.get("source_name"), entry["adopt_error"][:100]))
    entries = [e for e in entries if not e.get("adopt_error")]
    manifest = {"key": key, "models": entries, "skipped": skipped, "message": message}
    return _finish(manifest, manifest_path, strokes, strength, workers)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build 3D objects from painted outlines")
    parser.add_argument("--objects", required=True, help="JSON: {camera: {...}, objects: [...]}")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-blender", action="store_true")
    args = parser.parse_args(argv)
    try:
        import pygame
        pygame.display.init()
        data = json.loads(Path(args.objects).read_text(encoding="utf-8"))
        manifest = build_all(data["objects"], args.out, data["camera"], use_blender=not args.no_blender,
                             strokes=data.get("strokes"), strength=float(data.get("stroke_strength", 1.0)),
                             workers=worker_count())
    except Exception as error:
        print("MESH_ERROR: %s" % error, file=sys.stderr)
        return 1
    print(json.dumps({"manifest": str(Path(args.out) / "manifest.json"), "models": len(manifest["models"]),
                      "message": manifest["message"], "bake": manifest.get("bake_message", "")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
