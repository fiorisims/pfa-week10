"""What is the ground made of?  (Reference World Builder)

Reads the painted ground (everything below the horizon that isn't an object) as a few kinds of
surface, so the world can treat each one properly: grass grows on grass, water is water, dirt stays
bare.

1. **Group** - paint below the horizon is clustered by colour, so each tone of ground is one group.
2. **Name** - CLIP looks at each group and says what it looks like, with colour as a second opinion
   (a pure blue patch is hardly dirt).  If CLIP isn't available, colour decides alone.
3. **Texture** - each surface gets a seamless tile stitched from its own real paint.
4. **Bake** - ``bake_ground`` paints a top-down map of the terrain with those tiles, using the
   painting's camera: inside the painted view each spot gets the surface painted there.

CLI (used by Unreal, because numpy isn't available inside it):
    python reference_surfaces.py --job job.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402

GROUPS = 6
GRID = 72
TILE = 512
TILE_CM = 520.0            # one repeat of a surface's paint on the ground
GROUND_MARGIN = 0.07       # of the picture height, below the horizon
SHADOW_BELOW = 0.22        # brightness: darker groups are shadow and framing, not surface
MIN_FLOOR = 0.04           # a floor outline smaller than this share of the picture isn't trusted
MIN_SHARE = 0.03           # a group smaller than this share of the ground is detail, not a surface
BAKE_SIZE = 2048
FLOOR_SHARP_CM = 30.0      # the painted floor is pasted as-is while one of its pixels covers less ground than this
FLOOR_GONE_CM = 140.0      # ... and fades into the surface's tile by here (far away it is too coarse)
# surface -> what CLIP is asked, and the look that supports it
SURFACES = {
    "grass": "a painting of green grass",
    "meadow": "a painting of a flowery meadow",
    "dirt": "a painting of bare brown dirt",
    "sand": "a painting of pale sand",
    "rock": "a painting of grey rock",
    "water": "a painting of blue water",
    # towns and rooms (reference_setting decides which kinds a picture may have)
    "road": "a painting of an asphalt road",
    "pavement": "a painting of a stone pavement sidewalk",
    "snow": "a painting of white snow",
    "ice": "a painting of frozen blue ice",
    "floorboards": "a painting of a wooden floor",
    "tiles": "a painting of a tiled floor",
    "carpet": "a painting of a carpet",
}
NATURE_SURFACES = ["grass", "meadow", "dirt", "sand", "rock", "water"]
PLANT_SURFACES = {"grass", "meadow"}


def hsv(rgb):
    x = np.asarray(rgb, np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    high, low = x.max(-1), x.min(-1)
    d = high - low
    hue = np.zeros_like(high)
    nz = d > 1e-6
    rm = nz & (high == r)
    gm = nz & (high == g) & ~rm
    bm = nz & ~rm & ~gm
    hue[rm] = (60.0 * ((g - b)[rm] / d[rm])) % 360.0
    hue[gm] = 60.0 * ((b - r)[gm] / d[gm]) + 120.0
    hue[bm] = 60.0 * ((r - g)[bm] / d[bm]) + 240.0
    return hue, np.where(high > 0, d / np.maximum(high, 1e-6), 0.0), high


def group_colours(pixels, groups=GROUPS, rounds=10, seed=1):
    """k-means on an array of colours.  Returns (centres, label of each pixel)."""
    pixels = np.asarray(pixels, np.float32)
    rng = np.random.default_rng(seed)
    sample = pixels[rng.integers(0, len(pixels), min(len(pixels), 6000))]
    groups = min(groups, len(np.unique(sample.round(), axis=0)))
    centres = [sample.mean(0)]
    for _ in range(1, groups):
        gap = np.min([((sample - c) ** 2).sum(-1) for c in centres], axis=0)
        centres.append(sample[int(np.argmax(gap))])
    centres = np.array(centres)
    for _ in range(rounds):
        labels = ((sample[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
        for i in range(len(centres)):
            if (labels == i).any():
                centres[i] = sample[labels == i].mean(0)
    return centres, ((pixels[:, None] - centres[None]) ** 2).sum(-1).argmin(1)


def colour_prior(colour):
    """How much each surface's usual look fits this colour, 0..1."""
    hue, sat, val = [float(v[0]) for v in hsv(np.asarray(colour, np.float32)[None])]

    def near(centre, width):
        gap = abs((hue - centre + 180.0) % 360.0 - 180.0)
        return max(0.0, 1.0 - gap / width)

    # paintings run muted: a dark teal with 10% saturation is still green paint, not grey stone.
    # Only paint with almost no colour at all reads as neutral.
    tinted = min(1.0, sat / 0.08)
    vivid = min(1.0, sat / 0.25)
    bright = max(0.0, min(1.0, (val - 0.25) / 0.3))
    return {
        "grass": 0.15 + 0.85 * near(125.0, 70.0) * tinted,
        "meadow": 0.12 + 0.5 * near(290.0, 60.0) * tinted,
        "dirt": 0.1 + 0.9 * near(30.0, 28.0) * vivid * (1.0 - max(0.0, val - 0.75)),
        "sand": 0.05 + 0.95 * near(45.0, 25.0) * min(1.0, val / 0.7) * (1.0 - min(1.0, sat / 0.6)),
        "rock": 0.1 + 0.8 * (1.0 - tinted) * min(1.0, val / 0.45),
        # water is a clear, fairly bright blue: a dark bluish shadow on the ground isn't
        "water": 0.02 + 0.98 * near(205.0, 40.0) * min(1.0, sat / 0.35) * bright,
        # a street is grey whatever lights it, so its prior barely cares about colour: the paint of
        # a street at night is pink and purple with neon, and CLIP decides
        # (only a slight lean by brightness: under neon or at night a sidewalk is as dark as a road)
        "road": 0.42 + 0.1 * (1.0 - min(1.0, val / 0.85)),
        "pavement": 0.42 + 0.1 * min(1.0, val / 0.7),
        "snow": 0.05 + 0.95 * max(0.0, min(1.0, (val - 0.6) / 0.25)) * (1.0 - min(1.0, sat / 0.3)),
        "ice": 0.05 + 0.8 * near(195.0, 35.0) * max(0.0, min(1.0, (val - 0.5) / 0.3)),
        "floorboards": 0.1 + 0.9 * near(30.0, 30.0) * tinted,
        "tiles": 0.3 + 0.4 * (1.0 - vivid) * min(1.0, val / 0.6),
        "carpet": 0.25 + 0.4 * vivid,
    }


def clip_opinions(rgb, masks, names):
    """Each masked group against each surface's description, or None without CLIP."""
    try:
        import reference_objects as ro
        phrases = [SURFACES[n] for n in names]
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(ro._file(ro.CLIP_REPO, ro.CLIP_TOKENIZER, "clip")))
        tok.no_padding()
        tok.no_truncation()
        ids = np.zeros((len(phrases), 77), np.int64)
        for i, phrase in enumerate(phrases):
            encoded = list(tok.encode(phrase).ids)[:77]
            ids[i, :len(encoded)] = encoded
        text = ro._session(ro._file(ro.CLIP_REPO, ro.CLIP_TEXT, "clip")).run(None, {"input_ids": ids})[0]
        text = text / np.linalg.norm(text, axis=-1, keepdims=True)
        vision = ro._session(ro._file(ro.CLIP_REPO, ro.CLIP_VISION, "clip"))
        rows = []
        for mask in masks:
            x = (ro._on_grey(rgb, mask) - ro.CLIP_MEAN) / ro.CLIP_STD
            emb = vision.run(None, {"pixel_values": x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
            emb = emb / np.linalg.norm(emb)
            sims = emb @ text.T * 100.0
            probs = np.exp(sims - sims.max())
            rows.append(probs / probs.sum())
        return np.stack(rows)
    except Exception:
        return None


def _largest_part(mask, step=4):
    """The biggest connected piece of a (scattered) colour group, at full size."""
    import reference_scene
    small = mask[::step, ::step]
    parts = reference_scene._components(small) if small.any() else []
    if not parts:
        return mask
    rows, cols = zip(*max(parts, key=len))
    out = np.zeros_like(mask)
    for r, c in zip(rows, cols):
        out[r * step:(r + 1) * step, c * step:(c + 1) * step] = True
    return out & mask if (out & mask).any() else out


def context_opinions(rgb, masks, names):
    """Each group seen **in its surroundings** (its biggest piece plus what is around it), against each
    surface's description, or None without CLIP.  A pink-lit sidewalk next to a kerb reads as a sidewalk;
    cut out onto grey it reads as anything.
    """
    try:
        import reference_water
        clip = reference_water._model()
        text = clip.texts([SURFACES[n] for n in names])
        rows = []
        for mask in masks:
            crop = reference_water.crop_around(rgb, _largest_part(mask), 0.08)
            sims = text @ clip.image(crop) * 100.0
            probs = np.exp(sims - sims.max())
            rows.append(probs / probs.sum())
        return np.stack(rows)
    except Exception:
        return None


def name_groups(colours, opinions, names):
    """The surface each colour group is, weighing what it looks like against its colour."""
    out = []
    for i, colour in enumerate(colours):
        prior = colour_prior(colour)
        scores = np.array([prior[n] for n in names])
        if opinions is not None:
            scores = scores * (0.1 + np.asarray(opinions[i]))
        out.append(names[int(np.argmax(scores))])
    return out


def _fill_nearest(labels):
    """Unknown cells (-1; objects hide the ground there) take the nearest known surface."""
    out = labels.copy()
    while (out < 0).any():
        known = out >= 0
        if not known.any():
            return np.zeros_like(out)
        grown = out.copy()
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            moved = np.roll(out, shift, axis)
            valid = np.roll(known, shift, axis)
            if axis == 0:
                valid[0 if shift == 1 else -1] = False
            else:
                valid[:, 0 if shift == 1 else -1] = False
            take = (grown < 0) & valid
            grown[take] = moved[take]
        out = grown
    return out


def read_surfaces(rgb, texture, horizon, cache_dir, water_mask=None, object_mask=None, seed=1, allowed=None,
                  floor=None, group_on_floor=True, floor_hidden=None):
    """Classify the painted ground.  Returns a dict for the inventory (JSON-friendly), or None.

    allowed: the kinds of ground this kind of place can have (default: the outdoor ones).
    floor: the floor's outline (reference_objects.floor_mask); without it, everything a little below
    the horizon counts as ground.
    """
    import reference_scene
    import reference_strokes as rs
    H, W = rgb.shape[:2]
    cache_dir = Path(cache_dir).resolve()
    ground = np.zeros((H, W), bool)
    # a little below the horizon: right at it, tree trunks and backdrop get read as ground
    ground[int(max(0, horizon + GROUND_MARGIN * H)):] = True
    use_floor = floor is not None and floor.shape == (H, W) and floor.mean() >= MIN_FLOOR
    if use_floor and group_on_floor:
        ground = floor.copy()                       # the floor itself: walls and shop fronts aren't ground
    if object_mask is not None:
        ground &= ~object_mask
    dry = ground if water_mask is None else ground & ~water_mask
    if dry.sum() < 200:
        return None
    centres, labels = group_colours(rgb[dry], seed=seed)
    label_map = -np.ones((H, W), np.int64)
    label_map[dry] = labels
    shares = np.bincount(labels, minlength=len(centres)) / float(len(labels))
    values = hsv(centres)[2]
    # near-black is shadow and framing, not a kind of ground: it takes on whatever surrounds it
    shadows = {i for i in range(len(centres)) if values[i] < SHADOW_BELOW}
    groups = [i for i in range(len(centres)) if shares[i] >= MIN_SHARE and i not in shadows]         or [i for i in range(len(centres)) if i not in shadows] or [int(np.argmax(shares))]
    shadows -= set(groups)
    names = [n for n in (allowed or NATURE_SURFACES) if n in SURFACES] or list(NATURE_SURFACES)
    masks = [label_map == i for i in groups]
    alone = clip_opinions(rgb, masks, names)
    around = context_opinions(rgb, masks, names)
    # what it looks like in place counts most; its paint on its own breaks ties
    opinions = alone if around is None else around if alone is None else 0.2 * alone + 0.8 * around
    named = name_groups(centres[groups], opinions, names)
    # small groups join the closest big one: they're highlights and shadows of the same ground
    remap = {}
    for i in range(len(centres)):
        if i in shadows:
            continue
        if i in groups:
            remap[i] = named[groups.index(i)]
        else:
            nearest = min(groups, key=lambda g: float(((centres[g] - centres[i]) ** 2).sum()))
            remap[i] = named[groups.index(nearest)]
    kinds = []
    for name in remap.values():
        if name not in kinds:
            kinds.append(name)
    # "water" found here is real water the detector missed, if it is plainly water
    extra_water = None
    if "water" in kinds:
        wet = np.zeros((H, W), bool)
        for i, name in remap.items():
            if name == "water":
                wet |= label_map == i
        if wet.mean() >= 0.01:
            extra_water = wet
        kinds = [k for k in kinds if k != "water"] or [next(n for n in names if n != "water")]
        for i, name in list(remap.items()):
            if name == "water":
                remap[i] = min(kinds, key=lambda k: min(
                    (float(((centres[g] - centres[i]) ** 2).sum()) for g in groups if remap[g] == k and g != i),
                    default=1e12))
    grid = -np.ones((H, W), np.int64)
    for i, name in remap.items():
        grid[label_map == i] = kinds.index(name)
    if extra_water is not None:
        grid[extra_water] = -1
    # the ground under an object is whatever surrounds it
    step_y, step_x = H / float(GRID), W / float(GRID)
    cells = -np.ones((GRID, GRID), np.int64)
    for gy in range(GRID):
        for gx in range(GRID):
            cell = grid[int(gy * step_y):max(int(gy * step_y) + 1, int((gy + 1) * step_y)),
                        int(gx * step_x):max(int(gx * step_x) + 1, int((gx + 1) * step_x))]
            known = cell[cell >= 0]
            if known.size:
                cells[gy, gx] = int(np.bincount(known).argmax())
    cells = _fill_nearest(cells)
    # a tile of each surface's real paint
    TH, TW = texture.shape[:2]
    stats = rs.read_strokes(texture)
    patch = rs.patch_size(stats)
    full = np.zeros((TH, TW), np.int64) - 1
    ys = np.minimum((np.arange(TH) * H) // TH, H - 1)
    xs = np.minimum((np.arange(TW) * W) // TW, W - 1)
    full[:] = grid[ys][:, xs]
    surfaces = []
    for index, name in enumerate(kinds):
        # paint from the surface's main colour group only: a smaller group joined to it is often a
        # stray trunk or highlight, and its marks would be repeated all over the ground
        members = [i for i, n in remap.items() if n == name]
        main = max(members, key=lambda i: shares[i])
        region = np.zeros((TH, TW), bool)
        region[:] = (label_map == main)[ys][:, xs]
        region = rs.blur(region.astype(np.float32), 5) > 0.95
        tile = rs.quilt(texture.astype(np.float32), TILE, patch, region, seed=index + 1) if region.sum() > patch * patch else None
        chosen = (grid == index)
        colour = rgb[chosen].mean(0) if chosen.any() else np.array([110.0, 120.0, 90.0])
        if tile is None:
            tile = np.tile(colour[None, None], (TILE, TILE, 1)).astype(np.float32)
        path = cache_dir / ("surface_%s.npy" % name)
        np.save(path, tile.astype(np.float32))
        surfaces.append({"name": name, "tile": str(path), "color": [int(v) for v in colour],
                         "share": round(float(chosen.sum()) / max(1.0, float((grid >= 0).sum())), 3)})
    # the painted floor itself, to paste onto the terrain where it can be seen (bake_ground)
    # what to paste: the floor, minus what really stands on it (outlines, not boxes) and the water
    paste = dry
    if use_floor:
        hide = floor_hidden if floor_hidden is not None else object_mask
        paste = floor.copy() if hide is None else floor & ~hide
        if water_mask is not None:
            paste &= ~water_mask
    floor = None
    try:
        visible = np.zeros((TH, TW), bool)
        visible[:] = paste[ys][:, xs]
        visible = rs.blur(visible.astype(np.float32), 2) > 0.6      # a little off the edges of objects
        np.save(cache_dir / "floor_picture.npy", texture[..., :3].astype(np.uint8))
        np.save(cache_dir / "floor_mask.npy", visible)
        floor = {"picture": str(cache_dir / "floor_picture.npy"), "mask": str(cache_dir / "floor_mask.npy"),
                 "share": round(float(visible.mean()), 3)}
    except Exception:
        floor = None
    return {"surfaces": surfaces, "cells": cells.tolist(), "labels": {str(i): n for i, n in remap.items()},
            "extra_water": extra_water, "used_clip": opinions is not None, "floor": floor}


def dominant(result):
    return max(range(len(result["surfaces"])), key=lambda i: result["surfaces"][i]["share"])


def line(result):
    """One sentence for the panel."""
    parts = ["%s %d%%" % (s["name"], round(100 * s["share"])) for s in sorted(result["surfaces"], key=lambda s: -s["share"])]
    return "Ground: " + ", ".join(parts)


# --------------------------------------------------------------------------- #
# painting the terrain
# --------------------------------------------------------------------------- #
def sample_cells(cells, u, v):
    """Surface index at picture positions (arrays), or -1 outside the picture."""
    cells = np.asarray(cells)
    n_rows, n_cols = cells.shape
    inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    out = np.full(np.shape(u), -1, np.int64)
    out[inside] = cells[np.minimum((v[inside] * n_rows).astype(int), n_rows - 1),
                        np.minimum((u[inside] * n_cols).astype(int), n_cols - 1)]
    return out


def picture_position(x, y, camera):
    """Where each ground position shows up in the picture (the same camera as rwb_layout)."""
    distance = camera["player_y"] - y
    ahead = distance > 1.0
    safe = np.where(ahead, distance, 1.0)
    u = 0.5 + x * camera["focal"] / (safe * camera["W"])
    v = (camera["horizon"] + camera["eye"] * camera["focal"] / safe) / camera["H"]
    return np.where(ahead, u, -1.0), np.where(ahead, v, -1.0), ahead


def surface_at(cells, shares, x, y, camera, seed=1):
    """Surface index for ground positions; beyond the painted view the ground's main surface."""
    import reference_strokes as rs
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    u, v, _ahead = picture_position(x, y, camera)
    out = sample_cells(cells, u, v)
    main = int(np.argmax(shares))
    return np.where(out >= 0, out, main)


def floor_weight(distance, camera, picture_h):
    """How much of the painted floor to paste at a distance: 1 where it is sharp, 0 where too coarse."""
    rows = picture_h / float(camera["H"])                         # floor picture rows per camera row
    footprint = distance ** 2 / (camera["eye"] * camera["focal"] * rows)   # ground cm under one picture row
    return np.clip(1.0 - (footprint - FLOOR_SHARP_CM) / (FLOOR_GONE_CM - FLOOR_SHARP_CM), 0.0, 1.0)


def paste_floor(out, xs, ys, camera, picture, mask):
    """The painted floor, projected through the painting's camera onto the ground it shows.

    Where objects hide the floor, beyond the picture and far away the surface tiles stay.
    """
    import reference_strokes as rs
    PH, PW = picture.shape[:2]
    u, v, ahead = picture_position(xs, ys, camera)
    inside = ahead & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    col = np.clip((u * PW).astype(np.int64), 0, PW - 1)
    row = np.clip((v * PH).astype(np.int64), 0, PH - 1)
    seen = inside & mask[row, col]
    weight = np.where(seen, floor_weight(np.maximum(1.0, camera["player_y"] - ys), camera, PH), 0.0).astype(np.float32)
    weight = np.minimum(weight, rs.blur(weight, max(2, out.shape[0] // 256)))     # soft edges, never bleeding out
    painted = picture[row, col].astype(np.float32)
    return out * (1.0 - weight[..., None]) + painted * weight[..., None]


def bake_ground(job):
    """A top-down texture of the terrain.  Returns the output path."""
    import pygame
    import reference_strokes as rs
    camera, size = job["camera"], float(job["world_size"])
    surfaces = job["surfaces"]
    tiles = [np.load(s["tile"]).astype(np.float32) for s in surfaces]
    shares = [s["share"] for s in surfaces]
    bake = int(job.get("bake_size", BAKE_SIZE))
    centre = (np.arange(bake) + 0.5) / bake * size - size / 2.0
    xs, ys = np.meshgrid(centre, centre)
    index = surface_at(job["cells"], shares, xs, ys, camera)
    # blurred weights, so one surface fades into the next instead of meeting in a blocky line
    px = xs / TILE_CM * TILE
    py = ys / TILE_CM * TILE
    total = np.zeros((bake, bake, 3), np.float32)
    weight_sum = np.zeros((bake, bake), np.float32)
    for i, tile in enumerate(tiles):
        weight = rs.blur((index == i).astype(np.float32), max(4, bake // 128))
        if weight.max() <= 0.0:
            continue
        total += rs.sample(tile, px, py, wrap=True) * weight[..., None]
        weight_sum += weight
    out = total / np.maximum(weight_sum, 1e-4)[..., None]
    floor = job.get("floor")
    if floor and Path(floor.get("picture", "")).is_file() and Path(floor.get("mask", "")).is_file():
        out = paste_floor(out, xs, ys, camera, np.load(floor["picture"]), np.load(floor["mask"]))
    # bare ground at the water's edge and on steep slopes: both shed plants
    heights = np.asarray(job.get("heights") or [[0.0]], np.float32)
    if heights.size > 1 and job.get("rock_color"):
        import reference_scene
        full = reference_scene._resize(heights, bake, bake)
        gy, gx = np.gradient(full, size / bake)
        slope = np.clip((np.hypot(gx, gy) - 0.45) / 0.5, 0.0, 1.0)[..., None]
        out = out * (1.0 - slope) + np.array(job["rock_color"], np.float32) * slope * 0.9 + out * slope * 0.1
        if job.get("water_z") is not None:
            shore = np.clip(1.0 - (full - float(job["water_z"])) / 45.0, 0.0, 1.0)[..., None]
            out = out * (1.0 - 0.55 * shore) + np.array(job.get("shore_color", (120, 105, 80)), np.float32) * 0.55 * shore
    rgb = np.clip(out, 0, 255).astype(np.uint8)
    pygame.image.save(pygame.image.frombuffer(rgb.tobytes(), (bake, bake), "RGB"), str(job["out"]))
    return str(job["out"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paint the terrain from the picture's surfaces")
    parser.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    try:
        import pygame
        pygame.display.init()
        path = bake_ground(json.loads(Path(args.job).read_text(encoding="utf-8")))
    except Exception as error:
        print("SURFACE_ERROR: %s" % error, file=sys.stderr)
        return 1
    print(json.dumps({"out": path}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
