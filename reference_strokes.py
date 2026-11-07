"""Read a painting's brushstrokes and lay them over 3D objects (Reference World Builder).

A painting separates into broad colour (a heavy blur) and everything the brush added on top
(the painting minus that blur).  This module:

* **measures** the strokes: their direction, width, length and how directional they are;
* **extracts** them as a seamless map by image quilting: real patches of your paint, stitched
  along their least-visible seams, so the marks are yours rather than a generated noise;
* **bakes** a texture for a mesh: surfaces facing the painter show the painting itself; surfaces
  it never showed (backs, sides) get that part of the object's broad colour with your brushstrokes
  over it, at the same scale they have in the painting.  That replaces the old mirrored backs and
  stretched sides.

Pure numpy, so it is testable without Blender or Unreal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

STROKES_VERSION = "rwb-strokes-1"
MAP_SIZE = 512             # the seamless stroke map (pixels)
WORK_LONG = 1024           # strokes are measured at this resolution
MIN_PATCH = 48             # a patch smaller than a mark chops strokes into confetti
MAX_PATCH = 140
DETAIL_RADIUS = 10         # blur for separating marks from colour: wider keeps flat colour blocks
CANDIDATES = 48            # patches tried per block when quilting
ROTATE_TILES = 96          # tile size for measuring local stroke direction


# --------------------------------------------------------------------------- #
# basics
# --------------------------------------------------------------------------- #
def blur(a, radius):
    """Fast box blur (HxW or HxWxC); two passes look nearly Gaussian."""
    radius = int(radius)
    if radius < 1:
        return a.astype(np.float32)
    out = a.astype(np.float64)
    for _ in range(2):
        for axis in (0, 1):
            pad = [(0, 0)] * out.ndim
            pad[axis] = (radius + 1, radius)
            total = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
            hi = np.take(total, range(2 * radius + 1, total.shape[axis]), axis=axis)
            lo = np.take(total, range(0, total.shape[axis] - 2 * radius - 1), axis=axis)
            out = (hi - lo) / (2.0 * radius + 1.0)
    return out.astype(np.float32)


def luminance(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def shrink(rgb, long_side):
    """Nearest-free downscale by averaging whole-number blocks (keeps strokes crisp)."""
    H, W = rgb.shape[:2]
    step = int(max(H, W) // long_side)
    if step <= 1:
        return rgb, 1
    h, w = H // step, W // step
    cut = rgb[:h * step, :w * step].astype(np.float32)
    shape = (h, step, w, step) + cut.shape[2:]
    return cut.reshape(shape).mean((1, 3)), step


def sample(image, u, v, wrap=False):
    """Bilinear lookup at pixel coordinates (arrays of any shape)."""
    H, W = image.shape[:2]
    if wrap:
        u, v = np.mod(u, W), np.mod(v, H)
    else:
        u, v = np.clip(u, 0, W - 1), np.clip(v, 0, H - 1)
    x0, y0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fx, fy = (u - x0)[..., None], (v - y0)[..., None]
    if wrap:
        x0, y0 = x0 % W, y0 % H
        x1, y1 = (x0 + 1) % W, (y0 + 1) % H
    else:
        x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    return (image[y0, x0] * (1 - fx) * (1 - fy) + image[y0, x1] * fx * (1 - fy)
            + image[y1, x0] * (1 - fx) * fy + image[y1, x1] * fx * fy)


# --------------------------------------------------------------------------- #
# reading the strokes
# --------------------------------------------------------------------------- #
def _reach(acf, angle, limit, floor=0.25):
    """How far along a direction the autocorrelation stays above ``floor``, in pixels."""
    H, W = acf.shape
    cy, cx = H // 2, W // 2
    dx, dy = math.cos(angle), math.sin(angle)
    for r in range(1, limit):
        y, x = int(round(cy + dy * r)), int(round(cx + dx * r))
        if not (0 <= y < H and 0 <= x < W) or acf[y, x] < floor:
            return float(r)
    return float(limit)


def read_strokes(rgb, valid=None):
    """Measure brushwork.  ``rgb`` HxWx3 (0-255).  Returns a dict; angle 0 = horizontal, 90 = vertical."""
    small, step = shrink(rgb, WORK_LONG)
    lum = luminance(small.astype(np.float32))
    detail = lum - blur(lum, 8)
    if valid is not None and step > 1:
        valid = valid[::step, ::step][:small.shape[0], :small.shape[1]]
    gy, gx = np.gradient(blur(detail, 1))
    if valid is not None:
        gx, gy = gx * valid, gy * valid
    # local direction per tile, so a painting with many stroke directions isn't averaged to mush
    T = ROTATE_TILES
    h, w = (gx.shape[0] // T) * T, (gx.shape[1] // T) * T
    if h == 0 or w == 0:
        h, w, T = gx.shape[0], gx.shape[1], min(gx.shape)
    shape = (h // T, T, w // T, T)
    jxx = (gx[:h, :w] ** 2).reshape(shape).mean((1, 3))
    jyy = (gy[:h, :w] ** 2).reshape(shape).mean((1, 3))
    jxy = (gx[:h, :w] * gy[:h, :w]).reshape(shape).mean((1, 3))
    power = jxx + jyy
    active = power > max(1e-6, float(np.percentile(power, 40)))
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / np.maximum(power, 1e-6)
    # strokes run across the gradient; the sums keep strong, directional tiles in charge
    weight = power * coherence * active
    a = 0.5 * math.atan2(float((2.0 * jxy * weight).sum()), float(((jxx - jyy) * weight).sum())) + math.pi / 2.0
    # how far a mark stays similar to itself, measured in the tiles with the clearest direction and
    # along that tile's own direction: averaging over a whole painting would blur every length away
    lengths, widths = [], []
    if weight.sum() > 0:
        rows, cols = np.nonzero(weight >= np.percentile(weight[weight > 0], 60))
        for r, c in zip(rows.tolist(), cols.tolist()):
            tile = detail[r * T:(r + 1) * T, c * T:(c + 1) * T]
            tile = tile - tile.mean()
            spectrum = np.fft.rfft2(tile)
            acf = np.fft.fftshift(np.fft.irfft2(spectrum * np.conj(spectrum), s=tile.shape))
            peak = float(acf.max())
            if peak <= 1e-6:
                continue
            local = 0.5 * math.atan2(2.0 * float(jxy[r, c]), float(jxx[r, c] - jyy[r, c])) + math.pi / 2.0
            lengths.append(_reach(acf / peak, local, T // 2))
            widths.append(_reach(acf / peak, local + math.pi / 2.0, T // 2))
    length = float(np.median(lengths)) if lengths else 8.0
    width = float(np.median(widths)) if widths else 4.0
    if length < width:
        length, width = width, length
    scale = float(step)
    return {
        "angle": round((-math.degrees(a)) % 180.0, 1),
        "coherence": round(float((coherence * active).sum() / max(1, active.sum())), 3),
        "width": round(width * scale, 1), "length": round(length * scale, 1),
        "energy": round(float(np.abs(detail).mean()), 2),
    }


def describe(stats):
    """One line for the panel."""
    a = stats["angle"]
    way = ("horizontal" if a < 22.5 or a >= 157.5 else
           "rising to the right" if a < 67.5 else
           "vertical" if a < 112.5 else "rising to the left")
    feel = "very directional" if stats["coherence"] > 0.55 else "directional" if stats["coherence"] > 0.35 else "loose"
    return "Brushstrokes: %s, %s, about %d x %d px" % (feel, way, stats["length"], stats["width"])


# --------------------------------------------------------------------------- #
# quilting
# --------------------------------------------------------------------------- #
def _cut_mask(error):
    """Minimum-error boundary through ``error`` (h x overlap): True = keep the new patch."""
    h, w = error.shape
    cost = error.copy()
    for y in range(1, h):
        left = np.concatenate(([np.inf], cost[y - 1, :-1]))
        right = np.concatenate((cost[y - 1, 1:], [np.inf]))
        cost[y] += np.minimum(np.minimum(left, cost[y - 1]), right)
    keep = np.zeros((h, w), bool)
    x = int(np.argmin(cost[-1]))
    for y in range(h - 1, -1, -1):
        keep[y, x:] = True
        if y:
            lo, hi = max(0, x - 1), min(w, x + 2)
            x = lo + int(np.argmin(cost[y - 1, lo:hi]))
    return keep


def _valid_corners(valid, patch):
    """Top-left corners whose whole patch is usable (at least 95%)."""
    H, W = valid.shape
    if H < patch or W < patch:
        return np.zeros((0, 2), np.int64)
    total = np.pad(np.cumsum(np.cumsum(valid.astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    filled = total[patch:, patch:] - total[:-patch, patch:] - total[patch:, :-patch] + total[:-patch, :-patch]
    return np.argwhere(filled >= 0.95 * patch * patch)


def quilt(source, size, patch, valid=None, seed=1):
    """Efros-Freeman image quilting: a ``size`` x ``size`` seamless tile from real patches.

    source: HxWxC float.  valid: bool HxW of usable pixels.  Returns None if nothing fits.
    """
    H, W = source.shape[:2]
    patch = int(min(patch, min(H, W) // 2 * 2))
    patch -= patch % 2
    if patch < 8:
        return None
    valid = np.ones((H, W), bool) if valid is None else valid
    corners = _valid_corners(valid, patch)
    if not len(corners):
        return None
    rng = np.random.default_rng(seed)
    overlap = max(2, patch // 4)
    stride = patch - overlap
    span = size + overlap                               # the extra band becomes the wrap-around
    blocks = int(math.ceil((span - overlap) / float(stride)))
    out = np.zeros((blocks * stride + overlap, blocks * stride + overlap) + source.shape[2:], np.float32)
    for by in range(blocks):
        for bx in range(blocks):
            y, x = by * stride, bx * stride
            picks = corners[rng.integers(0, len(corners), CANDIDATES)]
            best, best_cost = None, np.inf
            for cy, cx in picks:
                cand = source[cy:cy + patch, cx:cx + patch]
                cost = 0.0
                if bx:
                    cost += float(((cand[:, :overlap] - out[y:y + patch, x:x + overlap]) ** 2).sum())
                if by:
                    cost += float(((cand[:overlap] - out[y:y + overlap, x:x + patch]) ** 2).sum())
                if cost < best_cost:
                    best, best_cost = cand, cost
                if not bx and not by:
                    break
            keep = np.ones((patch, patch), bool)
            if bx:
                err = ((best[:, :overlap] - out[y:y + patch, x:x + overlap]) ** 2).reshape(patch, overlap, -1).sum(-1)
                keep[:, :overlap] &= _cut_mask(err)
            if by:
                err = ((best[:overlap] - out[y:y + overlap, x:x + patch]) ** 2).reshape(overlap, patch, -1).sum(-1)
                keep[:overlap] &= _cut_mask(err.T).T
            region = out[y:y + patch, x:x + patch]
            region[keep] = best[keep]
    # wrap: the last band fades into the first, so the tile repeats without a visible line
    tile = out[:span, :span].copy()
    ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    shape = (overlap,) + (1,) * (tile.ndim - 1)
    tile[:overlap] = tile[:overlap] * ramp.reshape(shape) + tile[size:span] * (1 - ramp.reshape(shape))
    shape = (1, overlap) + (1,) * (tile.ndim - 2)
    tile[:, :overlap] = tile[:, :overlap] * ramp.reshape(shape) + tile[:, size:span] * (1 - ramp.reshape(shape))
    return tile[:size, :size]


def patch_size(stats):
    """A patch must hold a whole mark or quilting chops strokes into dashes."""
    wanted = max(stats.get("width", 8) * 6.0, stats.get("length", 16) * 2.0)
    size = int(max(MIN_PATCH, min(MAX_PATCH, wanted)))
    return size - size % 2


def extract_detail(rgb, valid=None, radius=None):
    """(detail HxWx3 signed, broad colour HxWx3, usable mask) of a painting or cut-out."""
    rgb = rgb.astype(np.float32)
    radius = int(radius or DETAIL_RADIUS)
    low = blur(rgb, radius)
    detail = rgb - low
    spread = max(1.0, float(detail.std()))
    detail = np.clip(detail, -2.0 * spread, 2.0 * spread)
    energy = blur(np.abs(detail).mean(-1), radius + 2)
    # skip dead-flat paint and the hard edges between shapes: those aren't brushwork
    usable = (energy >= np.percentile(energy, 25)) & (energy <= np.percentile(energy, 70))
    if valid is not None:
        usable &= valid
    return detail, low, usable


def stroke_map(rgb, stats=None, valid=None, size=MAP_SIZE, seed=1):
    """A seamless map of just the brushwork (signed RGB), or None if there is too little paint."""
    stats = stats or read_strokes(rgb, valid)
    small, step = shrink(rgb, 2048)
    if valid is not None and step > 1:
        valid = valid[::step, ::step][:small.shape[0], :small.shape[1]]
    detail, _low, usable = extract_detail(small, valid)
    if int(usable.sum()) < MIN_PATCH * MIN_PATCH:
        return None
    return quilt(detail, size, patch_size(stats) // max(1, step), usable, seed)


def prepare(texture, cache_dir, stamp):
    """Cached: the picture's stroke stats + seamless map.  Returns {stats, map, line}."""
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "strokes.json"
    key = "%s|%s" % (STROKES_VERSION, stamp)
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("key") == key and (not meta.get("map") or Path(meta["map"]).is_file()):
                return meta
        except Exception:
            pass
    stats = read_strokes(texture)
    tile = stroke_map(texture, stats)
    meta = {"key": key, "stats": stats, "line": describe(stats), "map": None}
    if tile is not None:
        path = cache_dir / "stroke_map.npy"
        np.save(path, tile.astype(np.float16))
        meta["map"] = str(path)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta


# --------------------------------------------------------------------------- #
# baking onto a mesh
# --------------------------------------------------------------------------- #
def rasterize(tri_uv, tri_values, size):
    """Interpolate per-corner values over a UV atlas.

    tri_uv: T x 3 x 2 (Blender UVs, v up).  tri_values: list of T x 3 x C arrays.
    Returns (covered bool HxW, list of HxWxC maps).
    """
    maps = [np.zeros((size, size, v.shape[-1]), np.float32) for v in tri_values]
    covered = np.zeros((size, size), bool)
    px = tri_uv[..., 0] * size - 0.5
    py = (1.0 - tri_uv[..., 1]) * size - 0.5        # image rows run down, UVs run up
    for t in range(len(tri_uv)):
        x0, x1, x2 = px[t]
        y0, y1, y2 = py[t]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9:
            continue
        lox, hix = int(max(0, math.floor(min(x0, x1, x2)))), int(min(size - 1, math.ceil(max(x0, x1, x2))))
        loy, hiy = int(max(0, math.floor(min(y0, y1, y2)))), int(min(size - 1, math.ceil(max(y0, y1, y2))))
        if hix < lox or hiy < loy:
            continue
        ys, xs = np.mgrid[loy:hiy + 1, lox:hix + 1]
        w1 = ((xs - x0) * (y2 - y0) - (x2 - x0) * (ys - y0)) / area
        w2 = ((x1 - x0) * (ys - y0) - (xs - x0) * (y1 - y0)) / area
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -0.01) & (w1 >= -0.01) & (w2 >= -0.01)
        if not inside.any():
            continue
        iy, ix = ys[inside], xs[inside]
        weights = np.stack([w0[inside], w1[inside], w2[inside]], -1)
        covered[iy, ix] = True
        for m, values in zip(maps, tri_values):
            m[iy, ix] = weights @ values[t]
    return covered, maps


def triplanar(detail, position, normal, px_per_unit):
    """Look a seamless map up on all three axes, so no surface gets stretched marks."""
    weights = np.abs(normal) ** 4
    weights = weights / np.maximum(weights.sum(-1, keepdims=True), 1e-6)
    p = position * px_per_unit
    across_x = sample(detail, p[..., 1], -p[..., 2], wrap=True)        # surfaces facing +-X
    across_y = sample(detail, p[..., 0], -p[..., 2], wrap=True)        # surfaces facing +-Y (like the picture)
    across_z = sample(detail, p[..., 0], p[..., 1], wrap=True)         # surfaces facing up or down
    return (across_x * weights[..., 0:1] + across_y * weights[..., 1:2] + across_z * weights[..., 2:3])


def facing_painter(normal, method, wrap_art=False):
    """1 where the painter saw this surface straight on, 0 where it was never shown.

    wrap_art: for a round building the windows really do continue round the back.
    """
    amount = normal[..., 2] if method == "ground" else -normal[..., 1]
    if wrap_art:
        amount = np.abs(amount)
    t = np.clip((amount - 0.25) / 0.4, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bake(surface, art, detail, uv_of, method, px_per_unit, strength=1.0, size=1024, wrap_art=False):
    """The final texture of a mesh.

    surface: {positions V3, normals V3, tris T3, tri_uvs T32}.  art: HxWx3 (0-255).
    detail: seamless stroke map (or None).  uv_of(positions) -> Nx2 picture UVs, v down.
    Returns (rgb HxWx3 uint8, share of texels that show the painting directly).
    """
    tris = surface["tris"]
    covered, (position, normal) = rasterize(surface["tri_uvs"],
                                            [surface["positions"][tris], surface["normals"][tris]], size)
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.maximum(length, 1e-6)
    H, W = art.shape[:2]
    art = art.astype(np.float32)
    uv = np.clip(uv_of(position.reshape(-1, 3)).reshape(size, size, 2), 0.0, 1.0)
    u, v = uv[..., 0] * (W - 1), uv[..., 1] * (H - 1)
    painted = sample(art, u, v)
    # broad colour of the same part of the object: strong enough to lose every painted mark
    broad = sample(blur(art, max(3, int(max(H, W) * 0.035))), u, v)
    unseen = broad
    if detail is not None and strength > 0.0:
        unseen = broad + triplanar(detail, position, normal, px_per_unit) * float(strength)
    amount = facing_painter(normal, method, wrap_art)[..., None]
    out = painted * amount + unseen * (1.0 - amount)
    direct = float((amount[covered] > 0.5).mean()) if covered.any() else 0.0
    return bleed(np.clip(out, 0, 255), covered), direct


def bleed(rgb, known, passes=6):
    """Spread colours past the islands' edges, so filtering never shows the empty atlas."""
    out = rgb.astype(np.float32).copy()
    filled = known.copy()
    if not filled.any():
        return out.astype(np.uint8)
    for _ in range(passes):
        grown = np.zeros_like(out)
        count = np.zeros(filled.shape, np.float32)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            f = np.roll(filled, shift, axis)
            grown += np.roll(out * filled[..., None], shift, axis)
            count += f
        new = (~filled) & (count > 0)
        out[new] = grown[new] / count[new][:, None]
        filled |= new
    out[~filled] = out[known].mean(0)
    return np.clip(out, 0, 255).astype(np.uint8)
