"""What is in the picture, and how many of each?  (Reference World Builder)

Produces the *inventory* the Unreal side builds from:

* **PSD**: every named layer is an object type (see ``reference_layers``);
  separate painted pieces on a layer are counted.
* **Flat image**: ``reference_objects`` finds each object (OWLv2), outlines it
  (SlimSAM) and double-checks its label (CLIP).  If those can't run, a colour-based
  fallback finds trees, bushes, flowers, rocks and water.
* **PSD layers with names like "Layer 3"** are recognised from their pixels too.

Every instance records where it is in the picture, so the builder can place it there
and build a new 3D object from its cut-out.  ``unique`` / repeated only decides whether
extra copies may be added outside the painted view; characters are never duplicated.

Also reports water and hills, and the painting's ground / sky / accent colours
so the world keeps the colour scheme.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402

import rwb_progress  # noqa: E402
import reference_layers as rl  # noqa: E402
import reference_objects  # noqa: E402

ROOT = Path(__file__).resolve().parent
INVENTORY_VERSION = "rwb-inventory-2"
WATER_GRID = 72           # terrain follows the shore much better than the old 24 x 24 grid
MAX_LAYER_LOOKS = 24      # PSD layers recognised from pixels per analysis (about 3 s each)
GENERIC_MIN_SCORE = 0.15  # a "Layer 3" must look like something, or it is paint rather than an object
BACKDROP_AREA = 0.45      # a layer covering this much of the canvas is a background
SOLID_FILL = 0.4          # a layer filling this much of its own box is one solid shape
WATER_HUE_RANGE = 26.0    # degrees either side of the water's hue
MOUNTAIN_MIN_SCORE = 0.3  # weak "mountain" hits are usually foliage; a real one is unmistakable

INDIVIDUALS = {"character", "creature", "prop_card", "unknown"}       # never duplicated
MASSES = {"flower", "grass"}                                          # always "many"
AREA_KINDS = {"water", "mountain", "path", "ground", "sky", "fog", "light"}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _resize(a, w, h):
    import reference_scene
    return reference_scene._resize(a, w, h)


def _hsv_arrays(rgb):
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    high = x.max(-1)
    low = x.min(-1)
    d = high - low
    hue = np.zeros_like(high)
    nz = d > 1e-6
    rm = nz & (high == r)
    gm = nz & (high == g) & ~rm
    bm = nz & ~rm & ~gm
    hue[rm] = (60.0 * ((g - b)[rm] / d[rm])) % 360.0
    hue[gm] = 60.0 * ((b - r)[gm] / d[gm]) + 120.0
    hue[bm] = 60.0 * ((r - g)[bm] / d[bm]) + 240.0
    sat = np.where(high > 0, d / np.maximum(high, 1e-6), 0.0)
    return hue, sat, high


def _profile(rgb, box, inner=0.6):
    """Colour profile of a box (its central part, to avoid the background)."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = max(1, (x1 - x0) * inner / 2), max(1, (y1 - y0) * inner / 2)
    ix0, ix1 = int(max(0, cx - hw)), int(min(W, cx + hw + 1))
    iy0, iy1 = int(max(0, cy - hh)), int(min(H, cy + hh + 1))
    if ix1 - ix0 < 1 or iy1 - iy0 < 1:
        ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(W, max(x0 + 1, x1)), min(H, max(y0 + 1, y1))
    ys, xs = np.mgrid[iy0:iy1, ix0:ix1]
    ys, xs = ys.ravel(), xs.ravel()
    step = max(1, len(xs) // 1500)
    pts = [(int(xs[i]), int(ys[i]), tuple(int(v) for v in rgb[ys[i], xs[i]])) for i in range(0, len(xs), step)]
    return rl._colour_profile(pts, (ix0, iy0, ix1, iy1))


# --------------------------------------------------------------------------- #
# colour fallback (no detector)
# --------------------------------------------------------------------------- #
def detect_by_colour(rgb, horizon):
    H, W = rgb.shape[:2]
    s = min(1.0, 256.0 / max(H, W))
    w, h = max(8, int(W * s)), max(8, int(H * s))
    small = _resize(rgb.astype(np.float32), w, h)
    hue, sat, val = _hsv_arrays(np.clip(small, 0, 255))
    rows = np.arange(h)[:, None] * np.ones((1, w))
    below = rows >= horizon * s
    classes = {
        "green": (hue >= 60) & (hue <= 175) & (sat > 0.18) & (val > 0.15),
        "flower": (sat > 0.45) & (val > 0.45) & ((hue < 45) | (hue > 270)),
        "water": (hue >= 175) & (hue <= 250) & (sat > 0.25) & (val > 0.3) & below,
        "rock": (sat < 0.12) & (val > 0.3) & (val < 0.75) & below,
    }
    dets = []
    for name, mask in classes.items():
        if name == "water":
            continue
        comps = rl._components(bytearray(mask.astype(np.uint8).ravel().tobytes()), w, h)
        for members in comps:
            area = len(members) / float(w * h)
            if name == "flower" and not (0.0002 < area < 0.01):
                continue
            if name in ("green", "rock") and area < (0.006 if name == "green" else 0.003):
                continue
            x0, y0, x1, y1 = rl._box(members, w)
            bw, bh = x1 - x0, y1 - y0
            if name == "green":
                category = "tree" if bh > bw * 0.9 and area > 0.02 else "bush"
            else:
                category = name
            dets.append({"score": 0.0, "category": category,
                         "box": [x0 / s, y0 / s, x1 / s, y1 / s]})
    return dets, classes["water"], (w, h)


# --------------------------------------------------------------------------- #
# depth placement helpers
# --------------------------------------------------------------------------- #
DEPTH_BANDS = {"fg": (0.05, 0.33), "mg": (0.33, 0.66), "bg": (0.66, 0.95)}


def _depth_t(bottom_y, H, horizon):
    """0 = right in front of the viewer ... 1 = at the horizon."""
    if bottom_y <= horizon:
        return 1.0
    return float(max(0.0, min(1.0, (H - bottom_y) / max(1.0, H - horizon))))


def _ground_palette(rgb, horizon, boxes, water_mask=None):
    H, W = rgb.shape[:2]
    keep = np.zeros((H, W), bool)
    keep[int(min(H - 1, max(0, horizon))):, :] = True
    for b in boxes:
        x0, y0, x1, y1 = [int(v) for v in b]
        keep[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
    if water_mask is not None:
        keep &= ~water_mask
    px = rgb[keep]
    lum = 0.2126 * px[:, 0] + 0.7152 * px[:, 1] + 0.0722 * px[:, 2] if len(px) else np.zeros(0)
    if (lum > 40).sum() > 200:
        px = px[lum > 40]                       # ignore near-black framing / shadows
    if len(px) < 50:
        px = rgb[int(H * 0.6):].reshape(-1, 3)
    step = max(1, len(px) // 3000)
    clusters = rl._kmeans([tuple(int(v) for v in c) for c in px[::step]], 4)
    cols = [rl._hex(c) for _n, c in clusters] or ["#5F7A4A"]
    rock = min((c for _n, c in clusters), key=lambda c: rl._sat(c) * 2 + rl._lum(c) / 255.0, default=(110, 105, 100))
    return {"colors": cols[:3], "rock": rl._hex(rock)}


# --------------------------------------------------------------------------- #
# inventory
# --------------------------------------------------------------------------- #
def refine_water(rgb, areas, horizon):
    """Water shape from the detector's pond masks, grown by hue to the real shoreline.

    A box prompt finds a patch of a pond.  Painted water runs from pale to deep but keeps its hue,
    so hue finds the rest, where exact colour would stop at the first change of tone.  Masks whose
    hue disagrees with the most confident one are dropped: they are usually the bank.
    Returns a bool mask or None.
    """
    H, W = rgb.shape[:2]
    masks = [(a["score"], np.load(a["mask"])) for a in areas if a.get("category") == "water"
             and "waterfall" not in a.get("label", "")]
    masks = [(score, mask) for score, mask in masks if mask.shape == (H, W) and mask[int(max(0, horizon)):].any()]
    if not masks:
        return None
    masks.sort(key=lambda item: -item[0])
    below = np.zeros((H, W), bool)
    below[int(max(0, horizon)):] = True
    hue, sat, val = _hsv_arrays(rgb)

    def hue_gap(reference):
        gap = np.abs(hue - reference)
        return np.minimum(gap, 360.0 - gap)

    def typical(part):
        # a circular mean, so reds either side of 0 degrees don't average to cyan
        angle = np.radians(hue[part])
        return float(np.degrees(math.atan2(float(np.sin(angle).mean()), float(np.cos(angle).mean()))) % 360.0)

    first = masks[0][1] & below
    reference = typical(first)
    wet = (hue_gap(reference) < WATER_HUE_RANGE) & (sat > 0.16) & (val > 0.28)
    union = np.zeros((H, W), bool)
    for _score, mask in masks:
        part = mask & below
        if part.sum() and abs(((typical(part) - reference + 180.0) % 360.0) - 180.0) < WATER_HUE_RANGE:
            union |= part & wet
    # only matching paint connected to a detected pond: a blue flower isn't water
    import reference_scene
    step = max(1, int(max(H, W) / 256))
    candidates = (wet & below)[::step, ::step]
    seeds = union[::step, ::step]
    out = np.zeros((H, W), bool)
    for cells in reference_scene._components(candidates):
        if len(cells) < 6:
            continue
        rows, cols = zip(*cells)
        if seeds[list(rows), list(cols)].any():
            grown = np.zeros_like(candidates)
            grown[list(rows), list(cols)] = True
            out |= np.repeat(np.repeat(grown, step, 0), step, 1)[:H, :W]
    return reference_objects.smooth_outline(out & below, 2) if out.any() else None


def water_tile(texture, water_grid, path):
    """A seamless swatch stitched from the painted water itself, or None."""
    import pygame
    import reference_scene
    import reference_strokes as rs
    TH, TW = texture.shape[:2]
    grid = np.asarray(water_grid, np.float32)
    # columns/rows of the coverage grid cover the picture; rows above the water are simply empty
    valid = reference_scene._resize(grid, TW, TH) > 0.8
    if valid.sum() < 2500:
        return None
    colour = texture.astype(np.float32)
    # paint only, not whatever edges the water: a little inside the shoreline
    valid = rs.blur(valid.astype(np.float32), 6) > 0.97
    stats = rs.read_strokes(texture, valid)
    tile = rs.quilt(colour, 512, rs.patch_size(stats), valid, seed=3)
    if tile is None:
        return None
    out = np.clip(tile, 0, 255).astype(np.uint8)
    pygame.image.save(pygame.image.frombuffer(out.tobytes(), (512, 512), "RGB"), str(path))
    return str(path)


def _grid_from_mask(mask, size=24):
    h, w = mask.shape
    out = [[0.0] * size for _ in range(size)]
    for gy in range(size):
        for gx in range(size):
            cell = mask[int(gy * h / size):max(int(gy * h / size) + 1, int((gy + 1) * h / size)),
                        int(gx * w / size):max(int(gx * w / size) + 1, int((gx + 1) * w / size))]
            out[gy][gx] = round(float(cell.mean()), 3)
    return out


def _merge_grids(grids):
    grids = [g for g in grids if g]
    if not grids:
        return None
    size = len(grids[0])
    return [[max(g[y][x] for g in grids if len(g) == size) for x in range(size)] for y in range(size)]


def _objects_mask(objects, H, W):
    """Everything standing on the ground, from the objects' boxes: that isn't surface."""
    hidden = np.zeros((H, W), bool)
    for entry in objects.values():
        for inst in entry["instances"]:
            box = inst.get("bbox_px") or inst.get("box")
            if box:
                x0, y0, x1, y1 = [int(v) for v in box]
                hidden[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    return hidden


def _load_depth(scene, shape):
    path = Path(scene.get("cache", "")) / "depth.npy" if scene.get("cache") and scene.get("depth") else None
    if path and path.is_file():
        disp = np.load(path)
        if disp.shape == shape:
            return disp
    return None


def _texture_rgb(scene, rgb):
    """The painting at full texture resolution (for sharp cut-outs), or the work copy."""
    path = scene.get("texture")
    if path and Path(path).is_file():
        import reference_scene
        return reference_scene._to_rgb(reference_scene._surface(path))
    return rgb


def _placement_from_box(box, W, H, disp, fit, horizon):
    """Placement data for something known only by its box (a PSD layer, a colour blob)."""
    x0, y0, x1, y1 = box
    item = {"contact_u": (x0 + x1) / 2.0 / W, "contact_v": y1 / float(H), "top_v": y0 / float(H),
            "centre_u": (x0 + x1) / 2.0 / W, "width_frac": (x1 - x0) / float(W), "u": (x0 + x1) / 2.0 / W,
            "area": (x1 - x0) * (y1 - y0) / float(W * H), "bbox_px": [x0, y0, x1, y1], "depth_eye": None}
    if disp is not None and fit:
        # the middle of the box: its edges are often whatever is behind the object
        cx0, cx1 = int(x0 + (x1 - x0) * 0.3), int(x1 - (x1 - x0) * 0.3)
        cy0, cy1 = int(y0 + (y1 - y0) * 0.3), int(y1 - (y1 - y0) * 0.3)
        region = np.zeros(disp.shape, bool)
        region[max(0, cy0):max(cy0 + 1, cy1), max(0, cx0):max(cx0 + 1, cx1)] = True
        focal = (W / 2.0) / math.tan(math.radians(60.0) / 2.0)
        if region.any():
            item["depth_eye"] = reference_objects.relative_depth(disp, region, fit, horizon, focal)
    return item


def _layer_picture(layer):
    """A layer's art on neutral grey, for the recogniser."""
    import reference_scene
    pieces = layer.get("instances") or []
    if not pieces or not pieces[0].get("texture") or not Path(pieces[0]["texture"]).is_file():
        return None
    import pygame
    art = reference_scene._surface(pieces[0]["texture"])
    canvas = pygame.Surface(art.get_size())
    canvas.fill((128, 128, 128))
    canvas.blit(art, (0, 0))
    return reference_scene._to_rgb(canvas)


def _solid_shape(layer):
    """A single compact painted piece: an object, even if we can't say which.  Washes are not."""
    x0, y0, x1, y1 = layer.get("bbox") or (0, 0, 0, 0)
    pieces = layer.get("instances") or []
    canvas = float(layer.get("canvas_area") or 0.0)
    if len(pieces) != 1 or not canvas or x1 <= x0 or y1 <= y0:
        return False
    box_share = (x1 - x0) * (y1 - y0) / canvas
    return box_share < 0.25 and float(layer.get("area") or 0.0) / max(1e-6, box_share) > SOLID_FILL


def recognise_unnamed_layers(layers, keywords_path):
    """Give layers whose names say nothing a category from what they look like.

    A meaningfully named but unknown layer that can't be recognised stays a painted object.
    A "Layer 3" that doesn't look like anything is dropped: it is probably shading.
    Returns (layers, number recognised).
    """
    pending = [l for l in layers if l.get("category") == "unknown" and l.get("kind") in ("card", "object")]
    if not pending:
        return layers, 0
    pending.sort(key=lambda l: (not l.get("generic"), -float(l.get("area") or 0.0)))
    table = reference_objects.load_queries(keywords_path)
    session = None
    looked = 0
    for number, layer in enumerate(pending):
        hit = None
        if number < MAX_LAYER_LOOKS and float(layer.get("area") or 0.0) < BACKDROP_AREA:
            picture = _layer_picture(layer)
            if picture is not None:
                try:
                    if session is None:
                        session = reference_objects._session(
                            reference_objects._file(reference_objects.OWL_REPO, reference_objects.OWL_MODEL, "owlv2"))
                    hit = reference_objects.classify_crop(picture, table, session=session)
                except Exception:
                    hit = None
        if hit and (hit[2] >= GENERIC_MIN_SCORE or not layer.get("generic")):
            layer["word"], layer["category"] = hit[0], hit[1]
            layer["kind"] = "object"
            layer["reason"] = "looks like %s (%.2f)" % (hit[0], hit[2])
            looked += 1
        elif layer.get("generic") and not _solid_shape(layer):
            layer["kind"], layer["category"] = "ignore", "ignore"
            layer["reason"] = "generic name, and doesn't look like an object"
            layer["instances"] = []
    return layers, looked


def build(result, image_path, cache_dir, keywords_path):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    scene = result.get("scene") or {}
    rgb_path = Path(scene.get("cache", "")) / "rgb.npy" if scene.get("cache") else None
    if rgb_path and rgb_path.is_file():
        rgb = np.load(rgb_path)
    else:
        import reference_scene
        rgb = reference_scene._to_rgb(reference_scene._scaled(reference_scene._surface(image_path), 768))
    H, W = rgb.shape[:2]
    horizon = float(scene.get("horizon", 0.35 * H)) * (H / float(scene.get("height", H) or H))
    disp = _load_depth(scene, (H, W))
    objects = {}
    water_grids, hill_grids, missing = [], [], []
    atmosphere = {}
    source, info = "", ""

    def add(category, kind, word, inst, label=None, individual=False, depth=None):
        key = (category, label) if individual else category
        entry = objects.setdefault(key, {"category": category, "kind": kind, "word": word or category,
                                         "label": label or category, "instances": [], "depth": depth,
                                         "individual": individual})
        entry["instances"].append(inst)

    layers = result.get("layers")
    if layers is not None:
        source, info = "layers", "PSD layer names"
        sx, sy = W / float(result["width"]), H / float(result["height"])
        layers, looked = recognise_unnamed_layers(layers, keywords_path)
        if looked:
            info += " + %d recognised from their pixels" % looked
        for layer in layers:
            kind, category = layer.get("kind"), layer.get("category")
            if kind == "water":
                water_grids.append(layer.get("coverage"))
                atmosphere.setdefault("water", (layer.get("colors") or {}).get("avg"))
            elif kind == "hill":
                hill_grids.append(layer.get("coverage"))
            elif kind == "sky":
                atmosphere.setdefault("sky", (layer.get("colors") or {}).get("top"))
            elif kind == "fog":
                atmosphere.setdefault("fog", (layer.get("colors") or {}).get("avg"))
            elif kind == "light":
                atmosphere.setdefault("accent", (layer.get("colors") or {}).get("accent"))
            elif kind == "ground":
                atmosphere.setdefault("ground", (layer.get("colors") or {}).get("palette"))
            elif kind in ("object", "card", "scatter"):
                individual = kind == "card" or category in INDIVIDUALS
                insts = layer.get("instances") or []
                if kind == "scatter" or not insts:
                    box = layer.get("bbox") or [0, 0, 1, 1]
                    insts = [{"bbox": box, "colors": layer.get("colors") or {}}] * (3 if kind == "scatter" else 1)
                for inst in insts:
                    b = inst["bbox"]
                    box = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
                    band = DEPTH_BANDS.get(layer.get("depth"))
                    item = _placement_from_box(box, W, H, disp, scene.get("fit"), horizon)
                    item.update({
                        "t": _depth_t(box[3], H, horizon), "box": [round(v, 1) for v in box],
                        "aspect": (box[3] - box[1]) / max(1.0, box[2] - box[0]),
                        "colors": inst.get("colors") or layer.get("colors") or {},
                        "label": layer.get("name"), "texture": inst.get("texture"),
                        "cutout": inst.get("texture") if kind in ("object", "card") else None,
                    })
                    if band:
                        item["band_t"] = band[0] + (band[1] - band[0]) * item["t"]
                    add(category, "object" if kind == "scatter" else kind, layer.get("word"), item,
                        label=layer["name"] if individual else None, individual=individual, depth=layer.get("depth"))
                if kind == "scatter":
                    objects[category]["mass"] = True
                    objects[category]["coverage"] = layer.get("coverage")
        boxes = [o["box"] for e in objects.values() for o in e["instances"]]
        water_mask = None
    if layers is None or not objects:                  # a PSD with nothing usable in its layers: read the art
        layer_info = info
        dets, info = None, ""
        recognised = None
        try:
            texture = _texture_rgb(scene, rgb)
            focal = (W / 2.0) / math.tan(math.radians(60.0) / 2.0)
            recognised = reference_objects.recognize(
                rgb, texture, cache_dir, scene.get("stamp", str(image_path)), keywords_path,
                disp, scene.get("fit"), horizon, focal)
        except Exception as error:
            info = str(error).splitlines()[0][:160] if str(error) else type(error).__name__
        water_mask = None
        if recognised is not None:
            found, areas, info = recognised
            source = "detector"
            if layers is not None:
                info = "%s (nothing usable in the layers)" % info
            water_mask = refine_water(rgb, areas, horizon)
            for area in areas:
                if area["category"] == "mountain" and area["score"] >= MOUNTAIN_MIN_SCORE:
                    mask = np.load(area["mask"])
                    band = mask[max(0, int(horizon - 0.1 * H)):int(horizon + 0.1 * H)]
                    if band.any():                     # canopy high in the frame isn't a mountain
                        hill_grids.append(_grid_from_mask(mask, WATER_GRID))
            for number, item in enumerate(found):
                box = item["box"]
                add(item["category"], "object", item["category"], {
                    **item, "id": "o%02d" % number, "u": (box[0] + box[2]) / 2.0 / W,
                    "t": _depth_t(item["contact_v"] * H, H, horizon),
                    "aspect": (item["top_v"] - item["contact_v"]) * -H / max(1.0, item["width_frac"] * W),
                    "bbox_px": item["bbox"], "colors": _profile(rgb, item["bbox"]),
                }, individual=item["category"] in INDIVIDUALS, label=item["label"])
            if water_mask is not None and water_mask.mean() > 0.01:
                water_grids.append(_grid_from_mask(water_mask, WATER_GRID))
                atmosphere.setdefault("water", rl._hex(rgb[water_mask].mean(0)))
            boxes = [o["box"] for e in objects.values() for o in e["instances"]]
        if recognised is None:
            reason = info
            dets, water_mask, (ww, wh) = detect_by_colour(rgb, horizon)
            water_mask = _resize(water_mask.astype(np.float32), W, H) > 0.5
            # trees etc.: things the depth map shows standing up (if depth worked)
            if scene.get("depth") and scene.get("cache"):
                try:
                    import reference_scene
                    built = reference_scene.build(scene["cache"])
                    dets = [d for d in dets if d["category"] not in ("tree", "bush")]
                    for layer in built.get("layers", []):
                        for inst in layer["instances"]:
                            dets.append({"score": 0.0, "category": layer["category"], "box": inst["bbox"]})
                except Exception:
                    pass
            source, info = "colours", "colour fallback (%s)" % reason
        for d in dets or []:
            category = d["category"]
            box = d["box"]
            if category == "water":
                m = np.zeros((H, W), bool)
                x0, y0, x1, y1 = [int(v) for v in box]
                m[max(int(horizon), y0):y1, x0:x1] = True
                water_mask = m if water_mask is None else (water_mask | m)
                continue
            if category == "mountain":
                m = np.zeros((H, W), bool)
                x0, y0, x1, y1 = [int(v) for v in box]
                m[y0:y1, x0:x1] = True
                hill_grids.append(_grid_from_mask(m))
                continue
            add(category, "object", category,
                {**_placement_from_box(box, W, H, disp, scene.get("fit"), horizon),
                 "t": _depth_t(box[3], H, horizon), "box": [round(v, 1) for v in box],
                 "aspect": (box[3] - box[1]) / max(1.0, box[2] - box[0]),
                 "colors": _profile(rgb, box), "score": round(d.get("score", 0.0), 3), "label": category})
        if water_mask is not None and water_mask.mean() > 0.01:
            if not water_grids:
                water_grids.append(_grid_from_mask(water_mask))
                atmosphere.setdefault("water", rl._hex(rgb[water_mask].mean(0)))
        boxes = [o["box"] for e in objects.values() for o in e["instances"]]

    out_objects = []
    for entry in objects.values():
        n = len(entry["instances"])
        repeated = bool(entry.get("mass")) or entry["category"] in MASSES and source != "layers" or n >= 2
        if entry["individual"]:
            repeated = False
        entry["count"] = n
        entry["unique"] = not repeated
        out_objects.append(entry)
    out_objects.sort(key=lambda e: (not e["unique"], e["category"]))

    # what is the ground made of?  (never fatal: the terrain falls back to plain colours)
    surfaces = None
    rwb_progress.report(0.86, "Reading the ground: grass, dirt, water")
    try:
        import reference_surfaces
        hidden = _objects_mask(objects, H, W)
        surfaces = reference_surfaces.read_surfaces(rgb, _texture_rgb(scene, rgb), horizon, cache_dir,
                                                    water_mask, hidden)
        if surfaces and surfaces.get("extra_water") is not None and not water_grids:
            # plainly water, though the detector had no word for it
            water_mask = surfaces["extra_water"]
            water_grids.append(_grid_from_mask(water_mask, WATER_GRID))
            atmosphere.setdefault("water", rl._hex(rgb[water_mask].mean(0)))
        if surfaces:
            surfaces.pop("extra_water", None)
            surfaces["line"] = reference_surfaces.line(surfaces)
    except Exception as error:
        surfaces = {"error": str(error)[:200]}
    water = _merge_grids(water_grids)
    tile_path = None
    if water:
        try:
            tile_path = water_tile(_texture_rgb(scene, rgb), water, Path(cache_dir).resolve() / "water_tile.png")
        except Exception:
            tile_path = None                          # the water just stays one colour
    rwb_progress.report(0.9, "Reading your brushstrokes")
    strokes = {}
    try:
        import reference_strokes
        strokes = reference_strokes.prepare(_texture_rgb(scene, rgb), cache_dir, scene.get("stamp", str(image_path)))
    except Exception as error:
        strokes = {"error": str(error)[:200]}
    ground = _ground_palette(rgb, horizon, boxes, water_mask)
    if atmosphere.get("ground"):
        ground["colors"] = list(atmosphere["ground"])[:3]
    top = rgb[: max(1, int(min(horizon, H * 0.3)))].reshape(-1, 3)
    sky = atmosphere.get("sky") or rl._hex(top.mean(0))
    return {
        "version": INVENTORY_VERSION, "source": source, "info": info,
        "objects": out_objects,
        "water": {"present": bool(water) and max(max(r) for r in water) > 0.25, "coverage": water,
                  "color": atmosphere.get("water") or "#3E7FA6", "tile": tile_path},
        "hills": {"coverage": _merge_grids(hill_grids)},
        "ground": ground, "sky": sky, "fog": atmosphere.get("fog"), "accent": atmosphere.get("accent"),
        "horizon": horizon / H, "missing": sorted(set(missing)),
        "picture": {"W": W, "H": H, "horizon": horizon},
        "strokes": strokes,
        "surfaces": surfaces,
    }


def summary(inventory):
    lines = ["Objects found by %s:" % inventory.get("info", "?")]
    for e in inventory.get("objects", []):
        rule = "seen once" if e["unique"] else "seen %d times" % e["count"]
        name = e["label"] if e.get("individual") else e["category"]
        lines.append("  %s: %s" % (name, rule))
    if inventory.get("water", {}).get("present"):
        lines.append("  water: pond / river basin")
    if inventory.get("hills", {}).get("coverage"):
        lines.append("  hills / mountains")
    return lines
