"""Buildings read from the picture's own geometry.  (Reference World Builder)

A painted building gives away two things:

* **which way each wall faces.**  Its horizontal edges (rows of windows, roof lines, kerbs) all aim at one
  vanishing point on the horizon.  A wall facing you has its vanishing point far off to the side (its
  edges stay level); a wall turned away has it near the picture; the walls of a street both aim at the
  middle, and the two walls of a corner building aim outwards, one to each side.
* **where it stands.**  The top edge of the floor outline under a wall is where the wall meets the
  ground.  A line on the ground is a straight line in the picture, and that line passes through the
  wall's vanishing point.

So each wall is a line in the picture, ``row(x) = horizon + g * (1 - k * (x - centre))``, with ``k`` from
the vanishing point (``k = 1 / (vanishing x - centre)``; 0 = facing you) and ``g`` from the ground under
it.  Everything is in picture pixels: Unreal turns the end points into the world through its own camera,
so the buildings line up with the picture at any field of view.  Two walls of one corner building also
give the camera's field of view (they are at right angles: focal^2 = -1 / (kA * kB)).

Plain numpy, no models.
"""

from __future__ import annotations

import math

import numpy as np

MIN_COHERENCE = 0.6       # a pixel on a clean straight edge, not a corner or texture
MIN_EDGE = 0.03           # gradient (brightness 0..1 per pixel)
TENSOR_RADIUS = 4         # pixels: the neighbourhood an edge's direction is read over
MAX_TILT = 45.0           # degrees from level: steeper edges are a building's uprights, not its rows
HORIZON_BAND = 0.06       # of the height: edges this close to the horizon say nothing about direction
TURNED = 0.25             # |slope| (depth per sideways step at 60 deg) above this = turned away from you
MIN_FACE = 1.0 / 12.0     # of the width: narrower runs of one direction join their neighbours
MIN_BASE_COLUMNS = 6      # a wall needs this many columns where its foot is seen
MIN_BASE_SHARE = 0.25     # ... and this share of its columns
MIN_CLEAN_SHARE = 0.08    # columns with nothing standing in front: this few are enough to place a wall
WALL_FOOT_QUANTILE = 0.3   # feet of things in front (bags, people) are always nearer than the wall's: lean to the far ones
HIDDEN_FOOT_QUANTILE = 0.2
MAX_BLOCKED = 0.6         # a "wall" whose foot is mostly other things' feet is those things
JOIN_REACH = 0.1          # of the width: two walls meeting this close to their boundary share a corner
STEP_JOIN = 0.2           # walls whose feet are this close in distance at the boundary are one frontage
MIN_ROWS_BELOW = 0.015    # of the height: a foot this close to the horizon is the end of the street, not a wall
UPRIGHT_DEPTH = 0.5       # depth changing this much less above a foot than below it = a wall stands there
FLOOR_GAP = 0.025         # of the height: gaps in the floor outline this small (markings, a kerb) are still floor
FOV_RANGE = (38.0, 100.0)
MIN_WALL_EYES = 5.5       # walls nearer than this many eye heights (about 9 m) are railings and kerbs
REFERENCE_FOV = 60.0      # slopes are reported at this field of view, to read as "depth per step"


def focal_for(W, fov=REFERENCE_FOV):
    return (W / 2.0) / math.tan(math.radians(fov) / 2.0)


def fov_for(W, focal):
    return math.degrees(2.0 * math.atan((W / 2.0) / focal))


# --------------------------------------------------------------------------- #
# which way the walls face
# --------------------------------------------------------------------------- #
def _blur(a, r):
    import reference_strokes
    return reference_strokes.blur(a, r)


def edge_directions(rgb, horizon, sky=None):
    """Per pixel: k (1 / (vanishing x - centre), 0 = facing you) and its weight (0 = no usable edge)."""
    x = np.asarray(rgb, np.float32)
    H, W = x.shape[:2]
    lum = x[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32) / 255.0
    # smoothed first: on a crisp one-pixel edge the two differences spread differently and tilt it 20%
    gy, gx = np.gradient(_blur(lum, 1))
    # the direction from a neighbourhood (a jagged, aliased line is still one line) ...
    jxx, jyy, jxy = _blur(gx * gx, TENSOR_RADIUS), _blur(gy * gy, TENSOR_RADIUS), _blur(gx * gy, TENSOR_RADIUS)
    trace = jxx + jyy
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / np.maximum(trace, 1e-9)
    angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)           # across the edge
    tx, ty = -np.sin(angle), np.cos(angle)                   # along it
    rows, cols = np.mgrid[0:H, 0:W].astype(np.float32)
    # ... but only pixels on the edge itself: one beside it would aim the line through itself, and
    # a few rows off a shallow line moves its vanishing point a long way
    strength = np.hypot(gx, gy)
    usable = (coherence > MIN_COHERENCE) & (strength > MIN_EDGE) & \
        (np.abs(ty) < math.sin(math.radians(MAX_TILT))) & (np.abs(horizon - rows) > HORIZON_BAND * H)
    if sky is not None:
        usable &= ~sky
    # the line through (x, y) along (tx, ty) meets the horizon at centre + 1/k
    across = (cols - W / 2.0) * ty + (horizon - rows) * tx
    k = ty / np.where(np.abs(across) < 1e-6, 1e-6, across)
    limit = 8.0 / focal_for(W)                               # slopes past 8 run straight away from you
    k = np.clip(k, -limit, limit)
    weight = np.where(usable, strength * np.abs(horizon - rows) / H, 0.0).astype(np.float32)
    return k.astype(np.float32), weight


def _weighted_median(values, weights):
    if len(values) == 0 or float(weights.sum()) <= 0.0:
        return None
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][min(len(order) - 1, int(np.searchsorted(cum, cum[-1] / 2.0)))])


def region_k(k, weight, mask=None, cols=None):
    """The vanishing direction of one region (a mask, or a range of columns)."""
    w = weight if mask is None else np.where(mask, weight, 0.0)
    if cols is not None:
        w = w[:, cols[0]:cols[1]]
        k = k[:, cols[0]:cols[1]]
    sel = w > 0
    if int(sel.sum()) < 30:
        return None
    return _weighted_median(k[sel], w[sel])


def column_k(k, weight, step=4, reach=None):
    """k per picture column, from the edges in a band of columns around it (None where unknown)."""
    H, W = k.shape
    reach = reach or max(8, W // 32)
    out = []
    for c in range(0, W, step):
        out.append(region_k(k, weight, cols=(max(0, c - reach), min(W, c + reach + 1))))
    values = np.array([np.nan if v is None else v for v in out], np.float64)
    full = np.interp(np.arange(W), np.arange(0, W, step), np.where(np.isnan(values), 0.0, values))
    known = np.interp(np.arange(W), np.arange(0, W, step), (~np.isnan(values)).astype(np.float64)) > 0.5
    return np.where(known, full, np.nan)


def slope(k_value, W, fov=REFERENCE_FOV):
    """k as 'depth gained per sideways step' at the reference field of view (for people and rules)."""
    return None if k_value is None else float(k_value) * focal_for(W, fov)


def street_or_corner(k, weight, W):
    """How the picture's walls run: a street (both sides aim at the middle), a corner (they aim outwards).

    Returns {"left", "right" (slopes of the outer thirds), "corner" (column of a convex corner, or None)}.
    """
    third = int(W * 0.35)
    left = slope(region_k(k, weight, cols=(0, third)), W)
    right = slope(region_k(k, weight, cols=(W - third, W)), W)
    corner = None
    best = 0.0
    # a corner is often cut off or rounded (a doorway facing you): look past a little of it each side
    near, far = int(W * 0.04), int(W * 0.3)
    for c in range(int(W * 0.25), int(W * 0.75), max(2, W // 64)):
        a = slope(region_k(k, weight, cols=(max(0, c - far), c - near)), W)
        b = slope(region_k(k, weight, cols=(c + near, min(W, c + far))), W)
        if a is None or b is None:
            continue
        # a convex corner: the wall on its left turns away to the left, the one on its right to the right
        strength = min(-a, b)
        if strength > 2 * TURNED and strength > best:
            best, corner = strength, c
    return {"left": None if left is None else round(left, 2), "right": None if right is None else round(right, 2),
            "corner": corner, "corner_strength": round(best, 2)}


def is_street(shape):
    """Could this be a street?  Only the walls themselves can say no: a corner facing you, or a side
    turned the wrong way (the left side running off to the left, the right side to the right).
    Level, unconverging edges (a drawing of flat facades) say nothing either way.
    """
    if shape.get("corner") is not None:
        return False
    left, right = shape.get("left"), shape.get("right")
    if left is not None and left < -TURNED * 0.6:
        return False
    if right is not None and right > TURNED * 0.6:
        return False
    return True


# --------------------------------------------------------------------------- #
# where the walls stand
# --------------------------------------------------------------------------- #
def ground_line(floor, horizon, blockers=None, sky=None, disp=None):
    """Per column, the row where the floor meets what stands on it (nan where it can't be seen).

    A column doesn't count where something in front (a person, a pole) stands on that edge, or where the
    floor runs on to the horizon (an opening: the far end of a street).  With a depth map, what stands on
    the edge must stand up: going up a wall the depth stays put, going up a sidewalk beyond a kerb it
    keeps falling away like the floor below.
    """
    H, W = floor.shape
    start = int(math.ceil(horizon + MIN_ROWS_BELOW * H))
    base = np.full(W, np.nan)
    if start >= H:
        return base
    look = max(3, int(0.08 * H))
    gap = max(2, int(FLOOR_GAP * H))
    for c in range(W):
        col = floor[start:, c]
        rows = np.nonzero(col)[0]
        if not len(rows):
            continue
        # up from the lowest floor pixel, through small gaps (road markings), to where the floor ends:
        # stray specks of "floor" up in a lit shop window aren't the wall's foot
        r = int(rows[-1])
        for prev in rows[::-1][1:]:
            if r - int(prev) > gap:
                break
            r = int(prev)
        r += start
        if r <= start:
            continue                                  # the floor runs on to the horizon: an opening
        above = slice(max(0, r - look), r)
        if blockers is not None and blockers[max(0, r - 3):r, c].any():
            continue                                  # something in front stands here
        if sky is not None and float(sky[above, c].mean()) > 0.4:
            continue
        if disp is not None and r - look >= 0 and r + look < H:
            up = (float(disp[r - 3, c]) - float(disp[r - look, c])) / (look - 3)
            down = (float(disp[r + look, c]) - float(disp[r + 3, c])) / (look - 3)
            if down > 1e-6 and up > UPRIGHT_DEPTH * down:
                continue                              # more ground, not a wall
        base[c] = r
    return base


def _runs(labels, min_len):
    """Runs of equal labels, short ones merged into the longer neighbour."""
    runs = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append([start, i, labels[start]])
            start = i
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for j, (a, b, _lab) in enumerate(runs):
            if b - a >= min_len:
                continue
            if j == 0:
                other = 1
            elif j == len(runs) - 1:
                other = j - 1
            else:
                other = j - 1 if runs[j - 1][1] - runs[j - 1][0] >= runs[j + 1][1] - runs[j + 1][0] else j + 1
            lo, hi = min(runs[j][0], runs[other][0]), max(runs[j][1], runs[other][1])
            runs[min(j, other)] = [lo, hi, runs[other][2]]
            del runs[max(j, other)]
            changed = True
            break
        # neighbours with the same label become one run
        merged = [runs[0]]
        for run in runs[1:]:
            if run[2] == merged[-1][2]:
                merged[-1][1] = run[1]
            else:
                merged.append(run)
        runs = merged
    return [(a, b, lab) for a, b, lab in runs]


def base_row(face, x):
    """The wall's foot at picture column x (pixels), or None where the line is behind the camera."""
    across = 1.0 - face["k"] * (x - face["cx"])
    if across <= 0.05:
        return None
    return face["horizon"] + face["g"] * across


def _fit_g(k_value, cols, rows, cx, horizon, quantile=0.5):
    across = 1.0 - k_value * (cols - cx)
    good = across > 0.05
    if int(good.sum()) < 3:
        return None, None
    g = (rows[good] - horizon) / across[good]
    g0 = float(np.quantile(g, quantile))
    # straightness from the far side only: what's nearer than the wall is something standing in front
    far = g[g <= np.quantile(g, min(1.0, quantile + 0.3))]
    spread = float(np.median(np.abs(far - g0))) / max(1e-6, g0)
    return g0, spread


def _ls_k(cols, rows, cx, horizon):
    """k and g fitted from the foot line alone (when the edges disagree): row - h = g - g k (x - cx)."""
    if len(cols) < 4:
        return None
    A = np.stack([np.ones_like(cols), -(cols - cx)], 1)
    sol, *_ = np.linalg.lstsq(A, rows - horizon, rcond=None)
    g, gk = sol
    if g <= 1.0:
        return None
    return float(gk / g), float(g)


def walls(rgb, horizon, floor, blockers=None, sky=None, directions=None, disp=None):
    """Every wall whose foot can be seen: [{k, g, x0, x1, ...}] left to right (pixels)."""
    H, W = floor.shape
    k, weight = directions if directions is not None else edge_directions(rgb, horizon, sky)
    clean = ground_line(floor, horizon, blockers, sky, disp)
    every = clean if blockers is None else ground_line(floor, horizon, None, sky, disp)
    ck = column_k(k, weight)
    f60 = focal_for(W)
    labels = []
    for c in range(W):
        s = ck[c] * f60 if not np.isnan(ck[c]) else 0.0
        labels.append(-1 if s < -TURNED else 1 if s > TURNED else 0)
    out = []
    cx = W / 2.0
    for a, b, _lab in _runs(labels, max(4, int(W * MIN_FACE))):
        cols = np.arange(a, b)
        need = max(MIN_BASE_COLUMNS, MIN_BASE_SHARE * (b - a))
        base, quantile = clean, WALL_FOOT_QUANTILE
        seen = ~np.isnan(base[a:b])
        if int(seen.sum()) < max(MIN_BASE_COLUMNS, MIN_CLEAN_SHARE * (b - a)):
            # people, bags and poles stand along most of it: their feet are nearer than the wall's
            base, quantile = every, HIDDEN_FOOT_QUANTILE
            seen = ~np.isnan(base[a:b])
            if int(seen.sum()) < need:
                continue
            blocked = float(np.isnan(clean[a:b])[seen].mean())
            if blocked > MAX_BLOCKED:
                continue                                 # a pole and a kiosk, not a wall behind them
        xs, ys = cols[seen].astype(np.float64) + 0.5, base[a:b][seen]
        # the direction comes from the edges (hundreds of window rows agree); the foot only says where.
        # A line fitted through the feet alone is pulled about by bags and people standing in front.
        kv = region_k(k, weight, cols=(a, b))
        if kv is None:
            fitted = _ls_k(xs, ys, cx, horizon)
            kv = fitted[0] if fitted is not None else 0.0
        g, spread = _fit_g(kv, xs, ys, cx, horizon, quantile)
        if g is None:
            continue
        if spread > 0.35:
            continue                                     # not one straight wall
        face = {"k": kv, "g": g, "cx": cx, "horizon": float(horizon), "x0": float(a), "x1": float(b),
                "spread": round(spread, 3), "seen": round(float(seen.mean()), 2)}
        # the wall's ends must be in front of the camera and below the horizon
        if base_row(face, face["x0"]) is None or base_row(face, face["x1"]) is None:
            continue
        out.append(face)
    return out


def _intersection(a, b):
    """Column where two walls' foot lines cross (pixels), or None."""
    den = a["g"] * a["k"] - b["g"] * b["k"]
    if abs(den) < 1e-9:
        return None
    return a["cx"] + (a["g"] - b["g"]) / den


def join(faces, W):
    """Walls that meet become one building: [[face, ...], ...].  Corners are snapped to where they meet."""
    buildings = []
    for face in faces:
        if buildings:
            prev = buildings[-1][-1]
            gap = face["x0"] - prev["x1"]
            if gap <= JOIN_REACH * W:
                boundary = (prev["x1"] + face["x0"]) / 2.0
                x = _intersection(prev, face)
                ra = base_row(prev, boundary)
                rb = base_row(face, boundary)
                if x is not None and abs(x - boundary) <= JOIN_REACH * W and base_row(prev, x) and base_row(face, x):
                    prev["x1"] = face["x0"] = float(x)
                    prev["corner_right"] = face["corner_left"] = True
                    buildings[-1].append(face)
                    continue
                if ra and rb and abs(ra - rb) / max(1.0, max(ra, rb) - prev["horizon"]) <= STEP_JOIN:
                    # one frontage with a small step: they meet halfway, so there's no notch between them
                    prev["x1"] = face["x0"] = float(boundary)
                    prev["row_right"] = face["row_left"] = (ra + rb) / 2.0
                    buildings[-1].append(face)
                    continue
        buildings.append([face])
    return buildings


def corner_fov(buildings, W):
    """The field of view that makes a corner building's walls meet at right angles, or None."""
    best = None
    for faces in buildings:
        # a wall turned left, then (past any cut-off corner facing you) one turned right
        pairs = []
        for i, a in enumerate(faces):
            for b in faces[i + 1:]:
                if abs(slope(b["k"], W)) > TURNED:
                    pairs.append((a, b))
                    break
        for a, b in pairs:
            sa, sb = slope(a["k"], W), slope(b["k"], W)
            if sa is None or sb is None or not (sa < -TURNED and sb > TURNED):
                continue
            focal = 1.0 / math.sqrt(-a["k"] * b["k"])
            fov = fov_for(W, focal)
            if FOV_RANGE[0] <= fov <= FOV_RANGE[1]:
                weight = (a["x1"] - a["x0"]) + (b["x1"] - b["x0"])
                if best is None or weight > best[1]:
                    best = (fov, weight)
    return None if best is None else round(best[0], 1)


def _face_record(face, roof, colour, H, W):
    """A wall in picture fractions: its foot's two ends, its roofline samples and its colour."""
    x0, x1 = face["x0"], face["x1"]
    samples = np.linspace(x0, x1 - 1, 9)
    roof_pts = [[round(float(x) / W, 4), round(float(roof[int(min(W - 1, max(0, x)))]) / H, 4)] for x in samples]
    lo, hi = int(max(0, x0)), int(min(W, max(x0 + 1, x1)))
    return {
        "a": [round(x0 / W, 4), round(face.get("row_left", base_row(face, x0)) / H, 4)],
        "b": [round(x1 / W, 4), round(face.get("row_right", base_row(face, x1)) / H, 4)],
        "k": round(face["k"] * W, 5),                       # per picture width, so it scales with the picture
        "slope": round(slope(face["k"], W), 2),
        "roof": roof_pts, "open_top": bool(np.median(roof[lo:hi]) <= 1),
        "color": [int(v) for v in colour[lo:hi].mean(0)],
        "corner_left": bool(face.get("corner_left")), "corner_right": bool(face.get("corner_right")),
    }


def facade_colours(rgb, roof, base, horizon):
    """Mean colour of each column between its roof and its foot (or the horizon)."""
    x = np.asarray(rgb, np.float32)
    H, W = x.shape[:2]
    out = np.zeros((W, 3), np.float32)
    for c in range(W):
        top = int(min(roof[c], H - 2))
        bottom = int(base[c]) if not np.isnan(base[c]) else int(horizon)
        bottom = max(top + 1, min(H, bottom))
        out[c] = x[top:bottom, c, :3].mean(0)
    return out


def frontage(rgb, horizon, floor, blockers=None, disp=None):
    """The buildings whose feet the picture shows, as walls in picture fractions.

    Returns {"buildings": [{"faces": [...], "corner": bool}], "fov": suggested field of view or None,
             "shape": street_or_corner(...), "line": ...}.
    """
    import reference_facades
    rgb = np.asarray(rgb)
    H, W = rgb.shape[:2]
    sky = reference_facades.sky_mask(rgb)
    directions = edge_directions(rgb, horizon, sky)
    shape = street_or_corner(*directions, W)
    if floor is None or not np.asarray(floor).any():
        return {"buildings": [], "fov": None, "shape": shape, "line": "Buildings: no floor to stand them on"}
    floor = np.asarray(floor, bool)
    level = reference_facades.upright_share(rgb, horizon)
    if level < reference_facades.MIN_UPRIGHT:
        return {"buildings": [], "fov": None, "shape": shape, "level": round(level, 3),
                "line": "Buildings: the picture isn't level, so their feet can't be read"}
    focal = focal_for(W)
    faces = []
    for face in walls(rgb, horizon, floor, blockers, sky, directions, disp):
        nearest = max(base_row(face, face["x0"]), base_row(face, face["x1"]))
        if focal / max(1e-6, nearest - horizon) >= MIN_WALL_EYES:
            faces.append(face)
    groups = join(faces, W)
    roof = reference_facades.roofline(sky)
    base = ground_line(floor, horizon, None, sky, disp)
    colour = facade_colours(rgb, roof, base, horizon)
    buildings = []
    for group in groups:
        records = [_face_record(f, roof, colour, H, W) for f in group]
        buildings.append({"faces": records, "corner": any(r["corner_right"] for r in records)})
    result = {"buildings": buildings, "fov": corner_fov(groups, W), "shape": shape}
    result["line"] = line(result)
    return result


def describe_face(face):
    s = face["slope"]
    if abs(s) <= TURNED:
        return "facing you"
    return "turned away to the %s" % ("left" if s < 0 else "right")


def line(result):
    buildings = result.get("buildings") or []
    if not buildings:
        return "Buildings: none standing where their foot can be seen"
    walls_n = sum(len(b["faces"]) for b in buildings)
    corners = sum(1 for b in buildings if b["corner"])
    text = "Buildings: %d read from the picture (%d walls%s)" % (
        len(buildings), walls_n, ", %d corner building%s" % (corners, "s" if corners > 1 else "") if corners else "")
    if result.get("fov"):
        text += "; the corner gives a %d deg field of view" % round(result["fov"])
    return text


def object_face(k_map, weight, mask, W):
    """The vanishing direction of one detected building (k per picture width), or None."""
    value = region_k(k_map, weight, mask=mask)
    return None if value is None else round(value * W, 5)
