"""Foreground and background: what stands in front of what.  (Reference World Builder)

A thing's distance comes from where it meets the ground (lower in the picture = nearer).  That breaks when
its foot is hidden: a building behind a bus, a tree behind a fence, a tower behind the roofs in front.  Its
lowest *visible* point is then higher in the picture than its real foot, and it would be placed too far
away and too big.  So, for every pair of outlines that touch:

* **who is in front**: along the edge where they meet, the depth map says which side is nearer (the
  occluder's paint runs on uninterrupted; the hidden one is cut off by it);
* **hidden feet**: an outline whose bottom edge rests on another, nearer outline doesn't show its foot.
  It must stand behind that one, and no further than its visible bottom says: rwb_layout places it
  between the two (at the depth map's distance when that falls between them).

Each object also gets a layer, near / middle / far, for the panel.  ``plate`` makes the **background
plate**: the picture with the free-standing things painted out, for projecting onto the walls and
buildings behind them (otherwise a person would be painted on the wall *and* stand in front of it).
Plain numpy.
"""

from __future__ import annotations

import math

import numpy as np

TOUCH = 3                 # pixels: outlines this close touch
MIN_EDGE = 8              # pixels of shared edge needed to judge who is in front
NEARER = 1.03             # disparity this much bigger along the shared edge = in front
HIDDEN_FEET = 0.4         # share of an outline's bottom columns resting on something nearer
LAYERS = ((8.0, "near"), (25.0, "middle"), (float("inf"), "far"))   # eye heights (8 = about 13 m)
PLATE_GROW = 0.006        # of the picture's long side, around what is painted out
STANDING_ON_WALLS = {"building", "sign"}   # these are the walls (or hang on them): never painted out


def _grow(mask, r):
    out = mask.copy()
    for _ in range(r):
        g = out.copy()
        g[1:] |= out[:-1]
        g[:-1] |= out[1:]
        g[:, 1:] |= out[:, :-1]
        g[:, :-1] |= out[:, 1:]
        out = g
    return out


def in_front(a, b, disp):
    """+1 if outline a is in front of b, -1 if b is in front of a, 0 when they don't touch or it's unclear."""
    if disp is None:
        return 0
    ga, gb = _grow(a, TOUCH), _grow(b, TOUCH)
    edge_a, edge_b = a & gb & ~b, b & ga & ~a
    if int(edge_a.sum()) < MIN_EDGE or int(edge_b.sum()) < MIN_EDGE:
        return 0
    da, db = float(np.median(disp[edge_a])), float(np.median(disp[edge_b]))
    if da > db * NEARER:
        return 1
    if db > da * NEARER:
        return -1
    return 0


def feet_cover(mask, fronts):
    """Share of the outline's bottom columns that rest on one of the outlines in front of it."""
    if not fronts:
        return 0.0
    H = mask.shape[0]
    cols = np.nonzero(mask.any(0))[0]
    if not len(cols):
        return 0.0
    bottom = H - 1 - np.argmax(mask[::-1, cols], axis=0)
    front = np.zeros_like(mask)
    for f in fronts:
        front |= f
    covered = 0
    for c, r in zip(cols, bottom):
        if front[min(H - 1, r + 1):min(H, r + 1 + TOUCH), c].any():
            covered += 1
    return covered / float(len(cols))


def _box_touch(a, b, pad=TOUCH):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def layer_name(distance_eyes):
    for limit, name in LAYERS:
        if distance_eyes < limit:
            return name
    return "far"


def contact_eyes(contact_v, H, horizon, focal):
    below = contact_v * H - horizon
    return None if below <= 0.012 * H else focal / below


def order(items, disp, H, horizon, focal):
    """Who hides whom.  items: [{"inst": dict, "mask": bool HxW}] (the masks at the picture's work size).

    Sets on each inst: "in_front_of"/"behind" (ids), "feet_hidden", "behind_v" (the nearest occluder's
    foot, a fraction of the height: the hidden thing stands further away than that), "layer".
    Returns counts {"near", "middle", "far", "hidden"}.
    """
    boxes = []
    for item in items:
        ys, xs = np.nonzero(item["mask"])
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None)
    fronts = [[] for _ in items]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if boxes[i] is None or boxes[j] is None or not _box_touch(boxes[i], boxes[j]):
                continue
            verdict = in_front(items[i]["mask"], items[j]["mask"], disp)
            if verdict > 0:
                fronts[j].append(i)
            elif verdict < 0:
                fronts[i].append(j)
    counts = {"near": 0, "middle": 0, "far": 0, "hidden": 0}
    for j, item in enumerate(items):
        inst = item["inst"]
        inst["behind"] = [items[i]["inst"].get("id") for i in fronts[j]]
        for i in fronts[j]:
            items[i]["inst"].setdefault("in_front_of", []).append(inst.get("id"))
        # only something standing nearer (its foot lower in the picture) can hide this one's foot
        own = float(inst.get("contact_v", 1.0))
        nearer = [i for i in fronts[j] if float(items[i]["inst"].get("contact_v", 0.0)) > own]
        share = feet_cover(item["mask"], [items[i]["mask"] for i in nearer])
        hidden = share >= HIDDEN_FEET
        inst["feet_hidden"] = bool(hidden)
        distance = contact_eyes(float(inst.get("contact_v", 1.0)), H, horizon, focal)
        if hidden:
            counts["hidden"] += 1
            # it stands behind the nearest of the things resting over its foot
            inst["behind_v"] = round(max(float(items[i]["inst"].get("contact_v", 0.0)) for i in nearer), 4)
            near = contact_eyes(inst["behind_v"], H, horizon, focal) or 1.0
            far = distance if distance else near * 4.0
            depth = inst.get("depth_eye")
            distance = min(far, max(near * 1.1, depth)) if depth else math.sqrt(near * max(near, far))
        if distance is None:
            distance = inst.get("depth_eye") or 30.0
        inst["layer"] = layer_name(distance)
        counts[inst["layer"]] += 1
    return counts


def line(counts):
    if not counts or not sum(counts[k] for k in ("near", "middle", "far")):
        return "Depth: no objects to sort"
    text = "Depth: %d near, %d in the middle, %d far" % (counts["near"], counts["middle"], counts["far"])
    if counts.get("hidden"):
        text += "; %d stand%s partly hidden behind others (placed behind them)" % (
            counts["hidden"], "s" if counts["hidden"] == 1 else "")
    return text


# --------------------------------------------------------------------------- #
# the background plate
# --------------------------------------------------------------------------- #
def _soft(a):
    """A [1 2 1] blur down and across: coarse levels come up smooth, not in blocks."""
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
    a = (p[:-2] + 2.0 * p[1:-1] + p[2:]) / 4.0
    return (a[:, :-2] + 2.0 * a[:, 1:-1] + a[:, 2:]) / 4.0


def fill(image, hole):
    """Paint over ``hole`` from its surroundings (push-pull: averages from ever coarser levels)."""
    img = np.asarray(image, np.float32)
    known = (~np.asarray(hole, bool)).astype(np.float32)
    levels = []
    c, w = img * known[..., None], known
    while True:
        levels.append((c, w))
        h, wd = w.shape
        if h <= 1 and wd <= 1:
            break
        ph, pw = h + (h % 2), wd + (wd % 2)
        c2 = np.pad(c, ((0, ph - h), (0, pw - wd), (0, 0)))
        w2 = np.pad(w, ((0, ph - h), (0, pw - wd)))
        c = c2.reshape(ph // 2, 2, pw // 2, 2, -1).sum((1, 3))
        w = w2.reshape(ph // 2, 2, pw // 2, 2).sum((1, 3))
    c, w = levels[-1]
    colour = c / np.maximum(w, 1e-6)[..., None]
    for c, w in reversed(levels[:-1]):
        h, wd = w.shape
        up = _soft(np.repeat(np.repeat(colour, 2, 0), 2, 1)[:h, :wd])
        trust = np.clip(w, 0.0, 1.0)[..., None]
        colour = (c / np.maximum(w, 1e-6)[..., None]) * trust + up * (1.0 - trust)
    out = img.copy()
    hole = np.asarray(hole, bool)
    out[hole] = colour[hole]
    return out


def plate(picture_path, standing, path):
    """The picture with ``standing`` (bool mask at any size) painted out, saved to ``path``."""
    import pygame
    surf = pygame.image.load(str(picture_path))
    W, H = surf.get_size()
    rgb = np.frombuffer(pygame.image.tobytes(surf, "RGB"), np.uint8).reshape(H, W, 3)
    small = np.asarray(standing, bool)
    ys = (np.arange(H) * small.shape[0] / float(H)).astype(int)
    xs = (np.arange(W) * small.shape[1] / float(W)).astype(int)
    hole = small[ys][:, xs]
    hole = _grow(hole, max(1, int(PLATE_GROW * max(W, H))))
    out = fill(rgb, hole)
    out = np.clip(out, 0, 255).astype(np.uint8)
    pygame.image.save(pygame.image.frombuffer(out.tobytes(), (W, H), "RGB"), str(path))
    return {"path": str(path), "share": round(float(hole.mean()), 3)}
