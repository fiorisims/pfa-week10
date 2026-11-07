"""Streets of buildings: where one building ends and the next begins.  (Reference World Builder)

A picture taken looking down a street (one-point perspective) has its buildings lined up along both
sides.  Upright edges stay upright in such a picture, and each image column on a side is one distance
along the street.  So the side is cut into buildings by looking, column by column, for:

* a jump in the roofline (where the facade meets the sky),
* a change of facade colour,
* a long upright edge (a building's corner),
* the edge of a box the detector called a building or storefront.

``street_layout`` returns, per side, a list of buildings with their columns, roofline samples and
colours; rwb_street turns them into separate blocks.  Plain numpy, no models.
"""

from __future__ import annotations

import numpy as np

CENTRE_GAP = 0.07          # columns this close to the middle are the far end of the street
MIN_SIDE_FACADE = 0.6      # share of a side strip above the horizon that must be building, not sky
MIN_UPRIGHT = 0.22         # share of edge strength that must be upright or flat for a level, one-point street
EDGE_MIN = 0.06            # gradient (0..1 brightness per pixel) that counts as an edge
NOMINAL_STEP_CM = 563.0    # street half width x focal / width at the defaults (650 cm, 60 deg): cm per unit of 1/|u-.5|
MIN_BUILDING_CM = 450.0    # buildings shorter than this along the street are merged
MAX_PER_SIDE = 9
SKY_TOLERANCE = 0.12       # colour distance (0..1) that still counts as sky


def _luma(rgb):
    return np.asarray(rgb, np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) / 255.0


def sky_mask(rgb):
    """Sky: the colour of the top of the picture's middle, reached from the top without crossing an edge."""
    x = np.asarray(rgb, np.float32) / 255.0
    H, W = x.shape[:2]
    top = x[: max(2, H // 30), int(W * 0.35):int(W * 0.65)].reshape(-1, 3)
    ref = np.median(top, axis=0)
    close = np.linalg.norm(x - ref, axis=-1) < SKY_TOLERANCE
    lum = _luma(rgb)
    gy = np.abs(np.diff(lum, axis=0, prepend=lum[:1]))
    gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    smooth = (gx + gy) < 0.08
    sky = np.zeros((H, W), bool)
    open_ = close & smooth
    sky[0] = open_[0]
    for r in range(1, H):
        # grows down from the top, and sideways along the row (under overhangs)
        row = open_[r] & (sky[r - 1] | np.roll(sky[r - 1], 1) | np.roll(sky[r - 1], -1))
        sky[r] = row
    return sky


def roofline(sky, min_run=3):
    """Per column, the first row that is building rather than sky (0 = the building leaves the frame)."""
    H, W = sky.shape
    roof = np.zeros(W, np.int64)
    for c in range(W):
        col = ~sky[:, c]
        run = 0
        for r in range(H):
            run = run + 1 if col[r] else 0
            if run >= min_run:
                roof[c] = r - min_run + 1
                break
        else:
            roof[c] = H
    return roof


def upright_share(rgb, horizon):
    """How level the picture is: the share of edge strength above the horizon that is upright or flat.

    A level camera keeps a building's corners upright and its roofs' far ends flat; a tilted,
    looking-up picture spreads its edges over every angle.
    """
    lum = _luma(rgb)[: max(4, int(horizon))]
    gy, gx = np.gradient(lum)
    mag = np.hypot(gx, gy)
    strong = mag > EDGE_MIN
    if strong.sum() < 50:
        return 0.0
    angle = np.degrees(np.arctan2(gy[strong], gx[strong])) % 180.0
    angle = np.minimum(angle, 180.0 - angle)          # 0 = upright edge, 90 = flat edge
    weight = mag[strong]
    level = (angle < 6.0) | (angle > 75.0)
    return float(weight[level].sum() / weight.sum())


def _norm(a):
    a = np.asarray(a, np.float64)
    top = np.percentile(a, 97) if a.size else 0.0
    return np.clip(a / top, 0.0, 1.5) if top > 1e-9 else np.zeros_like(a)


def _smooth(a, k=2):
    if k <= 0:
        return a
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    return np.convolve(np.pad(a, k, mode="edge"), kernel, mode="valid")


def column_scores(rgb, horizon, roof, boxes=()):
    """How likely each column is the boundary between two buildings (0..~3)."""
    x = np.asarray(rgb, np.float32) / 255.0
    H, W = x.shape[:2]
    lum = _luma(rgb)
    h = max(4, int(horizon))
    colour = np.zeros((W, 3), np.float32)
    upright = np.zeros(W, np.float32)
    gx = np.abs(np.diff(lum, axis=1, append=lum[:, -1:]))
    for c in range(W):
        top = min(int(roof[c]), h - 2)
        colour[c] = x[top:h, c].mean(0)
        upright[c] = (gx[top:h, c] > EDGE_MIN * 1.5).mean()
    k = max(3, W // 40)
    change = np.zeros(W)
    jump = np.zeros(W)
    roof_s = roof.astype(np.float64)
    for c in range(W):
        a, b = max(0, c - k), min(W, c + k + 1)
        if c - a < 2 or b - c < 3:
            continue
        # medians: a thin strip of lit windows or a pole isn't a change of building
        change[c] = float(np.linalg.norm(np.median(colour[a:c], 0) - np.median(colour[c:b], 0)))
        jump[c] = abs(np.median(roof_s[a:c]) - np.median(roof_s[c:b])) / float(H)
    box_edge = np.zeros(W)
    for box in boxes:
        for edge in (box[0], box[2]):
            c = int(round(edge))
            if 0 <= c < W:
                box_edge[max(0, c - k):c + k + 1] = 1.0
    # an upright edge alone is a window or a pole: it only helps where the colour or the roof changes too
    support = np.clip(_norm(change) + _norm(jump), 0.0, 1.0)
    score = _norm(_smooth(change, 1)) + 1.5 * _norm(jump) + 0.6 * _norm(_smooth(upright, 1)) * support + 0.5 * box_edge
    return score, colour


def distance_unit(u):
    """Distance along the street in units of NOMINAL_STEP_CM (1/|u - 0.5|)."""
    return 1.0 / max(1e-3, abs(float(u) - 0.5))


def split_side(score, colour, roof, side, W, H, min_cm=MIN_BUILDING_CM, max_count=MAX_PER_SIDE):
    """Buildings along one side, nearest first.  Columns run from the picture's edge toward the middle."""
    if side == "left":
        cols = list(range(0, int(W * (0.5 - CENTRE_GAP))))
    else:
        cols = list(range(W - 1, int(W * (0.5 + CENTRE_GAP)), -1))
    if len(cols) < 6:
        return []
    min_units = min_cm / NOMINAL_STEP_CM
    order = sorted(cols[2:-2], key=lambda c: -score[c])
    cuts = []
    for c in order:
        if score[c] < 0.55 or len(cuts) >= max_count - 1:
            break
        d = distance_unit((c + 0.5) / W)
        ends = [distance_unit((cols[0] + 0.5) / W), distance_unit((cols[-1] + 0.5) / W)]
        if all(abs(d - distance_unit((o + 0.5) / W)) >= min_units for o in cuts) and \
                all(abs(d - e) >= min_units * 0.7 for e in ends):
            cuts.append(c)
    # positions along the side, nearest (picture edge) first
    rank = {c: i for i, c in enumerate(cols)}
    cuts.sort(key=lambda c: rank[c])
    bounds = [cols[0]] + cuts + [cols[-1]]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        lo, hi = (a, b) if a < b else (b, a)
        span = list(range(lo, hi + 1))
        samples = np.linspace(lo, hi, 9).round().astype(int)
        out.append({
            "u_near": round((a + 0.5) / W, 4), "u_far": round((b + 0.5) / W, 4),
            "roof": [[round((int(c) + 0.5) / W, 4), round(float(roof[int(c)]) / H, 4)] for c in samples],
            "open_top": bool(np.median(roof[span]) <= 1),
            "color": [int(v) for v in (colour[span].mean(0) * 255.0)],
        })
    return out


def street_layout(rgb, horizon, boxes=()):
    """Is this a picture down a street?  If so, the buildings along each side.

    boxes: detector boxes (x0, y0, x1, y1) of buildings and storefronts, in the picture's pixels.
    """
    rgb = np.asarray(rgb)
    H, W = rgb.shape[:2]
    sky = sky_mask(rgb)
    roof = roofline(sky)
    h = max(4, int(horizon))
    upright = upright_share(rgb, horizon)
    strips = {"left": sky[:h, : max(1, int(W * 0.2))], "right": sky[:h, int(W * 0.8):]}
    facade = {k: 1.0 - float(v.mean()) for k, v in strips.items()}
    street = upright >= MIN_UPRIGHT and min(facade.values()) >= MIN_SIDE_FACADE
    result = {"street": bool(street), "upright": round(upright, 3),
              "facade": {k: round(v, 3) for k, v in facade.items()}, "sides": {}}
    if not street:
        result["reason"] = ("the picture isn't level (upright edges %d%%)" % round(100 * upright)
                            if upright < MIN_UPRIGHT else "no buildings down both sides")
        return result
    score, colour = column_scores(rgb, horizon, roof, boxes)
    for side in ("left", "right"):
        result["sides"][side] = split_side(score, colour, roof, side, W, H)
    return result


def line(result):
    if not result.get("street"):
        return "Layout: open city (%s)" % result.get("reason", "")
    counts = {k: len(v) for k, v in result["sides"].items()}
    return "Layout: street, %d buildings on the left and %d on the right" % (counts.get("left", 0), counts.get("right", 0))
