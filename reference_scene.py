"""Single-image scene reconstruction for the Reference World Builder.

Turns ONE flat image into a believable 3D layout - no layers needed:

1. **Depth** - Depth Anything V2 (small, ONNX, runs on the CPU through
   onnxruntime) predicts how near or far every pixel is.  Downloaded once from
   Hugging Face, or put the file at ``models/depth_anything_v2_small.onnx``.
2. **Camera** - the sky is found (depth + colour), the horizon line is fitted
   from how depth changes down the picture, and a simple pinhole camera
   (``--fov``, eye height) is solved so ground pixels land at the right
   distance.  Lower in the picture = closer, with real perspective.
3. **Geometry** - every non-sky pixel becomes a 3D point.  The lowest points
   per ground cell make the terrain; tall clusters above it become trees /
   rocks with their real size, colours and cropped artwork.
4. **Projection** - every terrain vertex gets the UV of where the camera saw
   it, so the painting is projected onto the ground (hidden spots fall back
   to colour).  The part of the picture beyond the terrain becomes a backdrop
   that lines up with the far edge.

Without the depth model it still does steps 2 and 4 on a flat ground plane.

CLI (used by Unreal):
    python reference_scene.py build --cache DIR [--eye 160 --depth 8000 ...]
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
import pygame  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODEL_REPO = "onnx-community/depth-anything-v2-small"
MODEL_FILE = "onnx/model.onnx"
LOCAL_MODEL = ROOT / "models" / "depth_anything_v2_small.onnx"
SCENE_VERSION = "rwb-scene-1"
WORK_LONG = 768          # analysis resolution (long side)
TEXTURE_LONG = 2048      # projected painting resolution
NET_LONG = 518           # depth network input (long side, multiple of 14)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# --------------------------------------------------------------------------- #
# image helpers
# --------------------------------------------------------------------------- #
def _surface(path):
    p = Path(path)
    if p.suffix.lower() in (".psd", ".psb"):
        import turnaround_sheet as ts
        p = ts.flatten_psd(p)
    raw = pygame.image.load(str(p))
    surf = pygame.Surface(raw.get_size(), pygame.SRCALPHA, 32)
    surf.blit(raw, (0, 0))
    return surf


def _scaled(surf, long_side):
    w, h = surf.get_size()
    s = min(1.0, long_side / float(max(w, h)))
    if s < 1.0:
        surf = pygame.transform.smoothscale(surf, (max(2, int(round(w * s))), max(2, int(round(h * s)))))
    return surf


def _to_rgb(surf):
    w, h = surf.get_size()
    return np.frombuffer(pygame.image.tobytes(surf, "RGB"), np.uint8).reshape(h, w, 3).copy()


def _save_rgba(array, path):
    h, w = array.shape[:2]
    surf = pygame.image.frombuffer(np.ascontiguousarray(array, np.uint8).tobytes(), (w, h), "RGBA")
    pygame.image.save(surf, str(path))
    return str(path)


def _resize(a, w, h):
    """Bilinear resize of an HxW or HxWxC float array."""
    a = np.asarray(a, np.float32)
    H, W = a.shape[:2]
    ys = np.clip((np.arange(h) + 0.5) * H / float(h) - 0.5, 0, H - 1)
    xs = np.clip((np.arange(w) + 0.5) * W / float(w) - 0.5, 0, W - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    if a.ndim == 3:
        wy = wy[..., None]
        wx = wx[..., None]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bottom = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def _hsv(rgb):
    r, g, b = [c / 255.0 for c in rgb[:3]]
    high, low = max(r, g, b), min(r, g, b)
    d = high - low
    if d == 0:
        hue = 0.0
    elif high == r:
        hue = (60.0 * ((g - b) / d)) % 360.0
    elif high == g:
        hue = 60.0 * ((b - r) / d) + 120.0
    else:
        hue = 60.0 * ((r - g) / d) + 240.0
    return hue, (0.0 if high == 0 else d / high), high


# --------------------------------------------------------------------------- #
# depth
# --------------------------------------------------------------------------- #
def _model_path():
    if LOCAL_MODEL.is_file():
        return LOCAL_MODEL
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE))


def estimate_depth(rgb, cache_dir, stamp):
    """Relative disparity, 0 = far (sky) ... 1 = nearest.  (None, reason) on failure."""
    cache = cache_dir / "depth.npy"
    meta = cache_dir / "depth.json"
    if cache.is_file() and meta.is_file():
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            disp = np.load(cache)
            if info.get("stamp") == stamp and disp.shape == rgb.shape[:2]:
                return disp, info.get("model", "depth")
        except Exception:
            pass
    try:
        import onnxruntime as ort
    except ImportError:
        return None, "onnxruntime is not installed (pip install onnxruntime)"
    try:
        model = _model_path()
    except Exception as error:
        return None, "could not get the depth model (%s)" % str(error).splitlines()[0][:160]
    H, W = rgb.shape[:2]
    s = NET_LONG / float(max(H, W))
    nh = max(14, int(round(H * s / 14.0)) * 14)
    nw = max(14, int(round(W * s / 14.0)) * 14)
    x = _resize(rgb.astype(np.float32) / 255.0, nw, nh)
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    try:
        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        out = session.run(None, {session.get_inputs()[0].name: x})[0]
    except Exception as error:
        return None, "depth model failed (%s)" % str(error).splitlines()[0][:160]
    out = np.squeeze(np.asarray(out, np.float32))
    disp = _resize(out, W, H)
    lo, hi = np.percentile(disp, 1.0), float(disp.max())     # no top clipping: it would bend the ground fit
    disp = np.clip((disp - lo) / max(1e-6, hi - lo), 0.0, 1.0).astype(np.float32)
    np.save(cache, disp)
    model_name = "Depth Anything V2 small"
    meta.write_text(json.dumps({"stamp": stamp, "model": model_name}), encoding="utf-8")
    return disp, model_name


# --------------------------------------------------------------------------- #
# sky, horizon, camera
# --------------------------------------------------------------------------- #
def sky_mask(rgb, disp):
    """Pixels of sky connected to the top edge (column runs from the top)."""
    img = rgb.astype(np.float32)
    H = img.shape[0]
    if disp is not None:
        threshold = max(0.02, float(np.percentile(disp, 1.0)) + 0.012)
        cand = disp < threshold
    else:
        top = img[0]
        lum = (0.2126 * top[:, 0] + 0.7152 * top[:, 1] + 0.0722 * top[:, 2]).mean() / 255.0
        blueish = float((top[:, 2] >= top[:, 0]).mean())
        cand = np.zeros(img.shape[:2], bool)
        if lum > 0.3 or blueish > 0.5:
            cand[0] = True
            for y in range(1, H):
                step = np.abs(img[y] - img[y - 1]).sum(1) < 28.0
                near = np.abs(img[y] - top).sum(1) < 110.0
                cand[y] = cand[y - 1] & step & near
    return np.cumprod(cand, axis=0).astype(bool)


def solve_camera(rgb, disp):
    """Horizon row and the disparity->distance fit (A, B: disp = A*row + B on the ground)."""
    H, W = rgb.shape[:2]
    sky = sky_mask(rgb, disp)
    col_bottom = sky.sum(0)
    sky_cols = col_bottom > 0
    sky_bottom = float(np.median(col_bottom[sky_cols])) if sky_cols.mean() > 0.3 else None
    fit = None
    horizon = None
    if disp is not None:
        rows = np.arange(int(H * 0.6), H)
        c0, c1 = int(W * 0.15), int(W * 0.85)
        meds = []
        for r in rows:
            vals = disp[r, c0:c1][~sky[r, c0:c1]]
            # objects standing on the ground are always NEARER than the ground behind
            # them in the same row, so the ground is the low end of each row
            meds.append(float(np.percentile(vals, 15)) if vals.size > 10 else np.nan)
        meds = np.array(meds)
        ok = np.isfinite(meds)
        if ok.sum() > 10:
            A, B = np.polyfit(rows[ok], meds[ok], 1)
            if A > 1e-5:
                # infinitely far = the sky's disparity (about 0); without sky, the farthest ground
                if sky.mean() > 0.02:
                    far = float(np.median(disp[sky]))
                else:
                    ground_px = disp[~sky]
                    far = float(np.percentile(ground_px, 1.0)) if ground_px.size else 0.0
                horizon = (far - B) / A
                fit = (float(A), float(B))
    if horizon is None or not np.isfinite(horizon):
        horizon = (sky_bottom + 0.03 * H) if sky_bottom is not None else 0.33 * H
    if sky_bottom is not None:
        horizon = min(max(horizon, sky_bottom - 0.02 * H), sky_bottom + 0.30 * H)
    horizon = float(min(max(horizon, 0.02 * H), 0.85 * H))
    return sky, horizon, sky_bottom, fit


def _stamp(path):
    st = Path(path).stat()
    return "%s|%s|%d|%d" % (SCENE_VERSION, Path(path).resolve(), st.st_size, int(st.st_mtime))


def prepare(image_path, cache_dir):
    """Analysis step (cached): depth, sky, horizon.  Returns a summary for the panel."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp(image_path)
    meta_path = cache_dir / "scene.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("stamp") == stamp and (cache_dir / "rgb.npy").is_file() \
                    and Path(meta.get("texture", "")).is_file() and (meta.get("depth") or not meta.get("retry_depth")):
                return meta
        except Exception:
            pass
    surf = _surface(image_path)
    texture_surf = _scaled(surf, TEXTURE_LONG)
    texture = cache_dir / "scene_image.png"
    pygame.image.save(texture_surf, str(texture))
    rgb = _to_rgb(_scaled(surf, WORK_LONG))
    np.save(cache_dir / "rgb.npy", rgb)
    disp, depth_info = estimate_depth(rgb, cache_dir, stamp)
    sky, horizon, sky_bottom, fit = solve_camera(rgb, disp)
    H, W = rgb.shape[:2]
    meta = {
        "stamp": stamp, "cache": str(cache_dir), "texture": str(texture),
        "width": W, "height": H, "horizon": horizon, "sky_bottom": sky_bottom, "fit": fit,
        "depth": depth_info if disp is not None else None,
        "depth_error": None if disp is not None else depth_info,
        "retry_depth": disp is None,
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta


# --------------------------------------------------------------------------- #
# generate-time reconstruction
# --------------------------------------------------------------------------- #
def _fill(values, known):
    """Fill unknown cells from their known neighbours (diffusion)."""
    out = np.where(known, values, 0.0).astype(np.float64)
    filled = known.copy()
    n = max(out.shape)
    for _ in range(n * 2):
        if filled.all():
            break
        total = np.zeros_like(out)
        count = np.zeros_like(out)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = np.roll(np.roll(np.where(filled, out, 0.0), dy, 0), dx, 1)
            mask = np.roll(np.roll(filled, dy, 0), dx, 1).astype(float)
            if dy == 1:
                shifted[0] = 0; mask[0] = 0
            if dy == -1:
                shifted[-1] = 0; mask[-1] = 0
            if dx == 1:
                shifted[:, 0] = 0; mask[:, 0] = 0
            if dx == -1:
                shifted[:, -1] = 0; mask[:, -1] = 0
            total += shifted
            count += mask
        new = (~filled) & (count > 0)
        out[new] = total[new] / count[new]
        filled |= new
    return out


def _blur(values, passes=1):
    out = values.astype(np.float64)
    for _ in range(passes):
        pad = np.pad(out, 1, mode="edge")
        out = sum(pad[1 + dy:1 + dy + out.shape[0], 1 + dx:1 + dx + out.shape[1]]
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    return out


def _components(mask):
    n_rows, n_cols = mask.shape
    seen = np.zeros_like(mask)
    comps = []
    for r in range(n_rows):
        for c in range(n_cols):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < n_rows and 0 <= nx < n_cols and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    return comps


def _category(colors, height_cm):
    hue, sat, val = _hsv(_hex_rgb(colors["dominant"]))
    if 55.0 <= hue <= 175.0 and sat > 0.15:
        return "tree" if height_cm > 420 else "bush"
    if val < 0.2:
        return "pine" if height_cm > 420 else "bush"
    if sat < 0.16:
        return "rock" if height_cm < 650 else "cliff"
    return "tree" if height_cm > 500 else "bush"


def _hex_rgb(value):
    value = str(value).lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def build(cache_dir, eye=160.0, depth_cm=8000.0, size=1.0, relief=1.0, fov=60.0, grid=72, density=0.55, seed=1):
    import reference_layers as rl
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "scene.json").read_text(encoding="utf-8"))
    rgb = np.load(cache_dir / "rgb.npy")
    disp = None
    if meta.get("depth") and (cache_dir / "depth.npy").is_file():
        disp = np.load(cache_dir / "depth.npy")
        if disp.shape != rgb.shape[:2]:
            disp = None
    H, W = rgb.shape[:2]
    f = (W / 2.0) / math.tan(math.radians(max(20.0, min(120.0, fov))) / 2.0)
    horizon = float(meta["horizon"])
    sky = sky_mask(rgb, disp)
    fit = meta.get("fit") if disp is not None else None
    hc = float(eye) * float(size)                     # camera height in cm (1 unit below)
    dmax = max(4.0, float(depth_cm) / hc)              # terrain depth in camera heights
    if horizon + f / dmax > H * 0.97:
        dmax = f / max(1.0, H * 0.97 - horizon)
    far_row = horizon + f / dmax
    half_w = (W / 2.0) * dmax / f * 1.05
    rng = np.random.default_rng(int(seed))

    # ---- per-pixel 3D points (camera at height 1, looking along +d)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    if fit:
        A, B = fit
        a = A * f
        b = A * horizon + B
        invd = (disp - b) / a
    else:
        invd = (ys - horizon) / f
    valid = (~sky) & (invd > 1.0 / (dmax * 6.0))
    if disp is not None:
        gy, gx = np.gradient(disp)
        valid &= np.hypot(gx, gy) < 0.02           # drop smeared silhouette edges
    d = 1.0 / np.maximum(invd, 1e-4)
    X = (xs - W / 2.0) * d / f
    Z = 1.0 - (ys - horizon) * d / f
    if not fit:
        Z = np.zeros_like(Z)
    inside = valid & (d <= dmax) & (np.abs(X) <= half_w)

    n = int(grid)
    col = np.clip(((X + half_w) / (2 * half_w) * n).astype(int), 0, n - 1)
    row = np.clip(((dmax - d) / dmax * n).astype(int), 0, n - 1)    # row 0 = far
    idx = (row * n + col)[inside]
    zin = Z[inside]
    ground = np.full(n * n, np.inf)
    top = np.full(n * n, -np.inf)
    np.minimum.at(ground, idx, zin)
    np.maximum.at(top, idx, zin)
    count = np.bincount(idx, minlength=n * n)
    seen = count >= 3
    ground = np.where(seen, np.clip(ground, -3.0, 25.0), 0.0).reshape(n, n)
    top = np.where(seen, top, -np.inf).reshape(n, n)
    seen = seen.reshape(n, n)
    if seen.any():
        # a cell that only saw an object (no ground) sticks up: compare with its neighbourhood
        known = np.where(seen, ground, np.nan)
        pad = np.pad(known, 2, mode="constant", constant_values=np.nan)
        stack = np.stack([pad[2 + dy:2 + dy + n, 2 + dx:2 + dx + n] for dy in range(-2, 3) for dx in range(-2, 3)])
        import warnings
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            local = np.nanmedian(stack, axis=0)
        spike = seen & np.isfinite(local) & (ground - local > 0.8)
        seen_ground = seen & ~spike
        ground = _blur(_fill(ground, seen_ground), 2)
    else:
        ground = np.zeros((n, n))

    # ground colour per cell (points close to the ground surface)
    ground_px = inside.copy()
    ground_px[inside] = zin < ground.reshape(-1)[idx] + 0.25
    gidx = (row * n + col)[ground_px]
    csum = np.zeros((n * n, 3))
    for ch in range(3):
        np.add.at(csum[:, ch], gidx, rgb[..., ch][ground_px].astype(np.float64))
    ccount = np.bincount(gidx, minlength=n * n)
    cknown = (ccount > 0).reshape(n, n)
    colors = np.zeros((n, n, 3))
    for ch in range(3):
        avg = np.where(ccount > 0, csum[:, ch] / np.maximum(1, ccount), 0.0).reshape(n, n)
        colors[..., ch] = _fill(avg, cknown) if cknown.any() else np.full((n, n), 110.0)

    # ---- vertices: cell centres
    vx = -half_w + (np.arange(n) + 0.5) * (2 * half_w / n)            # camera heights
    vd = dmax - (np.arange(n) + 0.5) * (dmax / n)                       # row 0 = far
    grid_x, grid_d = np.meshgrid(vx, vd)
    height_units = ground * relief
    u_px = W / 2.0 + grid_x * f / grid_d
    v_px = horizon + (1.0 - ground) * f / grid_d
    vis = (u_px >= 0) & (u_px <= W - 1) & (v_px >= 0) & (v_px <= H - 1) & (grid_d > 0.3)
    pu = np.clip(u_px, 0, W - 1).astype(int)
    pv = np.clip(v_px, 0, H - 1).astype(int)
    vis &= ~sky[pv, pu]
    if fit:
        d_pix = 1.0 / np.maximum(invd[pv, pu], 1e-4)
        vis &= grid_d <= d_pix * 1.15 + 0.3                              # hidden behind something
    vis_soft = _blur(vis.astype(float), 1)

    # ---- camera placement in Unreal (cm).  Terrain is centred on the origin.
    cam_y = depth_cm_eff = dmax * hc
    cam_y = depth_cm_eff / 2.0
    ground_cam = float(_blur(ground, 0)[n - 1, n // 2]) * relief
    scene = {
        "info": "", "far_row": far_row, "horizon": horizon, "focal": f,
        "camera": {"x": 0.0, "y": cam_y, "z": (1.0 + ground_cam) * hc},
        "terrain": {
            "n": n,
            "xs": [float(v * hc) for v in vx],
            "ys": [float(cam_y - v * hc) for v in vd],
            "heights": (height_units * hc).round(1).tolist(),
            "uv": np.stack([u_px / W, v_px / H], -1).round(5).tolist(),
            "vis": vis_soft.round(3).tolist(),
            "colors": colors.round().astype(int).tolist(),
            "texture": meta["texture"],
        },
        "world_x": float(2 * half_w * hc), "world_y": float(depth_cm_eff),
        "layers": [],
    }

    # ---- objects
    stem = "o%d" % int(seed % 1000)
    for old in cache_dir.glob("obj_*.png"):
        try:
            old.unlink()
        except OSError:
            pass
    instances = {}
    if fit:
        rise = np.where(seen, top - ground, 0.0)
        tall = seen & (rise > 1.1 / max(0.4, size))      # about 1.8 m and up
        comps = sorted(_components(tall), key=len, reverse=True)[:60]
        pix_idx = (row * n + col)
        for k, cells in enumerate(comps):
            cell_ids = np.array([r * n + c for r, c in cells])
            member = inside & np.isin(pix_idx, cell_ids) & (Z > ground.reshape(-1)[pix_idx] + 0.25)
            count_px = int(member.sum())
            if count_px < 30:
                continue
            py, px = np.nonzero(member)
            obj_x = X[member]
            obj_d = d[member]
            base = float(min(ground[r, c] for r, c in cells))
            height_cm = (float(np.percentile(Z[member], 98)) - base) * hc
            width_cm = (float(np.percentile(obj_x, 95)) - float(np.percentile(obj_x, 5))) * hc
            if height_cm < 80:
                continue
            x0, x1, y0, y1 = int(px.min()), int(px.max()) + 1, int(py.min()), int(py.max()) + 1
            sample = np.linspace(0, count_px - 1, min(count_px, 3000)).astype(int)
            pts = [(int(px[i]), int(py[i]), tuple(int(v) for v in rgb[py[i], px[i]])) for i in sample]
            colors_obj = rl._colour_profile(pts, (x0, y0, x1, y1))
            category = _category(colors_obj, height_cm)
            # split wide clusters into a row
            pieces = 1
            if width_cm > 1.8 * height_cm and category in ("tree", "pine", "bush", "rock"):
                pieces = max(2, min(8, int(round(width_cm / (height_cm * 0.9)))))
            edges = np.percentile(obj_x, np.linspace(0, 100, pieces + 1))
            for p in range(pieces):
                sel = (obj_x >= edges[p]) & (obj_x <= edges[p + 1]) if pieces > 1 else np.ones(count_px, bool)
                if sel.sum() < 15:
                    continue
                spy, spx = py[sel], px[sel]
                bx0, bx1, by0, by1 = int(spx.min()), int(spx.max()) + 1, int(spy.min()), int(spy.max()) + 1
                crop = np.zeros((by1 - by0, bx1 - bx0, 4), np.uint8)
                crop[..., :3] = rgb[by0:by1, bx0:bx1]
                crop[spy - by0, spx - bx0, 3] = 255
                tex = _save_rgba(crop, cache_dir / ("obj_%s_%02d_%d.png" % (stem, k, p)))
                sx = obj_x[sel]
                piece_w = (float(sx.max() - sx.min()) if pieces > 1 else width_cm / hc) * hc
                instances.setdefault(category, []).append({
                    "bbox": [bx0, by0, bx1, by1],
                    "colors": colors_obj if pieces == 1 else rl._colour_profile(
                        [(int(spx[i]), int(spy[i]), tuple(int(v) for v in rgb[spy[i], spx[i]]))
                         for i in np.linspace(0, len(spx) - 1, min(len(spx), 1500)).astype(int)],
                        (bx0, by0, bx1, by1)),
                    "texture": tex,
                    "world": {"x": float(np.median(sx)) * hc,
                              "y": cam_y - float(np.percentile(obj_d[sel], 30)) * hc,
                              "w": max(40.0, piece_w), "h": height_cm},
                })
    else:
        # no depth: plants and rocks from the colours of the visible ground, sized by type
        for r in range(n):
            for c in range(n):
                if not vis[r, c] or rng.random() > density * 0.05:
                    continue
                cell = colors[r, c]
                hue, sat, val = _hsv(cell)
                if 70 <= hue <= 170 and sat > 0.2:
                    category, h = ("tree", 900.0) if rng.random() < 0.45 else ("bush", 150.0)
                elif sat < 0.15 and val < 0.55:
                    category, h = "rock", 160.0
                elif sat > 0.4 and (hue < 45 or hue > 300):
                    category, h = "flower", 40.0
                else:
                    continue
                h *= float(rng.uniform(0.7, 1.3)) * size
                hexc = "#%02X%02X%02X" % tuple(int(v) for v in cell)
                instances.setdefault(category, []).append({
                    "bbox": [0, 0, 1, 1], "texture": None,
                    "colors": {"avg": hexc, "dominant": hexc, "accent": hexc, "dark": hexc, "top": hexc, "bottom": hexc},
                    "world": {"x": float(vx[c] + rng.uniform(-0.4, 0.4) * (2 * half_w / n)) * hc,
                              "y": cam_y - float(vd[r]) * hc, "w": h * 0.8, "h": h},
                })
    for i, (category, items) in enumerate(sorted(instances.items())):
        scene["layers"].append({
            "id": "s%02d" % i, "name": "Detected %s" % category, "category": category, "kind": "object",
            "instances": items, "colors": items[0]["colors"],
        })

    # ---- backdrop: everything beyond the terrain, lined up with its far edge
    tex_surf = pygame.image.load(meta["texture"])
    TW, TH = tex_surf.get_size()
    big = np.frombuffer(pygame.image.tobytes(tex_surf, "RGB"), np.uint8).reshape(TH, TW, 3).copy()
    cut = int(min(TH, max(2, round(far_row / H * TH))))
    back = np.zeros((cut, TW, 4), np.uint8)
    back[..., :3] = big[:cut]
    back[..., 3] = 255
    if fit:
        near_mask = valid & (d < dmax)                               # belongs to the 3D terrain range
        near_big = _resize(near_mask.astype(np.float32), TW, TH)[:cut] > 0.5
        for y in range(cut):                                          # paint over with neighbouring colours
            row_mask = near_big[y]
            if row_mask.any() and not row_mask.all():
                good = np.nonzero(~row_mask)[0]
                nearest = good[np.clip(np.searchsorted(good, np.arange(TW)), 0, len(good) - 1)]
                back[y, row_mask, :3] = back[y, nearest[row_mask], :3]
    back_path = _save_rgba(back, cache_dir / ("backdrop_%d.png" % int(far_row)))
    distance = depth_cm_eff * 1.4
    s_cm = distance / f                                               # cm per (work) pixel at that distance
    cam_z = scene["camera"]["z"]
    top_z = cam_z + horizon * s_cm
    bottom_z = cam_z - (far_row - horizon) * s_cm
    scene["backdrop"] = {"texture": back_path, "x": 0.0, "y": cam_y - distance,
                         "width": W * s_cm, "height": top_z - bottom_z, "z": (top_z + bottom_z) / 2.0}

    found = sum(len(v) for v in instances.values())
    scene["info"] = "%s | horizon at %d%% of the image | %d objects | terrain %.0f m deep" % (
        ("depth: " + meta["depth"]) if fit else "perspective only (no depth: %s)" % (meta.get("depth_error") or "fit failed"),
        round(100 * horizon / H), found, depth_cm_eff / 100.0)
    return scene


def main(argv=None):
    parser = argparse.ArgumentParser(description="Single-image scene reconstruction for Unreal")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("image")
    p.add_argument("--cache", required=True)
    b = sub.add_parser("build")
    b.add_argument("--cache", required=True)
    b.add_argument("--eye", type=float, default=160.0)
    b.add_argument("--depth", type=float, default=8000.0)
    b.add_argument("--size", type=float, default=1.0)
    b.add_argument("--relief", type=float, default=1.0)
    b.add_argument("--fov", type=float, default=60.0)
    b.add_argument("--grid", type=int, default=72)
    b.add_argument("--density", type=float, default=0.55)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "prepare":
            result = prepare(args.image, args.cache)
        else:
            result = build(args.cache, args.eye, args.depth, args.size, args.relief, args.fov, args.grid,
                           args.density, args.seed)
    except Exception as error:
        print("SCENE_ERROR: %s" % error, file=sys.stderr)
        return 1
    text = json.dumps(result)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(json.dumps({"out": args.out}))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
