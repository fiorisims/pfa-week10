"""Color Script Palette Organizer - pull dominant colors out of images and build
a customizable swatch sheet for Photoshop.

    python palette_sheet.py front.png side.png --swatches 6 --send
    python palette_sheet.py *.png --shared --orientation grid --cols 4
    python palette_sheet.py shot.psd --shape circle --aco palette.aco

Inputs are images (PNG / JPG / BMP / TGA / GIF / WEBP) and Photoshop files
(``.psd`` / ``.psb``, flattened through Photoshop first). There is no limit on
how many images you throw at it.

What it does:
  * pulls the *dominant colors* out of each image (a small, deterministic
    k-means on a subsampled grid - no numpy needed);
  * lays the palettes out as a customizable swatch sheet (horizontal rows,
    vertical columns or a grid; square / rounded / circular swatches; hex
    labels; a gradient strip under every palette);
  * renders the sheet to a PNG *and* rebuilds the very same layout natively in
    Photoshop - one real, editable layer per palette with exact fills, live
    text labels and the gradients as image layers - so nothing has to be
    re-done with the eyedropper.

The layout is planned once (``plan_layout``) into a plain spec dict and both
renderers - the pygame PNG renderer here and ``photoshop_palette.jsx`` - draw
from that same spec, so the preview and the Photoshop document always match.

The Photoshop hand-off uses ``photoshop_bridge.py`` (COM on Windows,
AppleScript on macOS) and launches Photoshop if it is closed. A ``.aco`` color
swatch file can also be written for the real Swatches panel (Load Swatches),
because Photoshop's panel is not scriptable in 2026 (see ``write_aco``).
"""

import argparse
import atexit
import json
import math
import struct
import sys
from pathlib import Path

import pygame

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE / "palette_sheet.png"
ACO_DEFAULT = HERE / "color_script.aco"
STRIP_DIR = HERE / "_palette_strips"

DOC_NAME = "Color Script"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif", ".webp"}
PSD_EXTS = {".psd", ".psb"}
ALL_EXTS = IMAGE_EXTS | PSD_EXTS

# layout constants (all in sheet pixels)
MARGIN = 48
TITLE_SIZE = 30
CAPTION_SIZE = 15
LABEL_SIZE = 13
CAPTION_BAND = 30
LABEL_BAND = 24
STRIP_BAND = 30
STRIP_GAP = 8
IMAGE_GAP = 14        # space between an image thumbnail / row and the swatches
BLOCK_GAP = 44        # vertical space between palettes (horizontal/grid)
COLUMN_GAP = 48       # horizontal space between palettes (vertical)
MAX_OUTPUT_PIXELS = 64_000_000

SHAPES = ("square", "rounded", "circle")
ORIENTATIONS = ("horizontal", "vertical", "grid")

DOC_BG_DEFAULT = (24, 30, 40)

DEFAULT_CAPTION_COLOR = (232, 238, 246)
DEFAULT_LABEL_COLOR = (232, 238, 246)
INK_LIGHT = (238, 242, 248)
INK_DARK = (34, 38, 46)

# cross-platform sans list (pygame picks the first available)
FONT_CANDIDATES = ("Segoe UI,segoeui,Helvetica Neue,helvetica,Helvetica,"
                   "Arial,arial,DejaVu Sans,dejavusans,Liberation Sans,"
                   "liberation sans,Noto Sans,notosans,FreeSans,freesans")

# selectable text faces (pygame picks the first available in each list). The
# key is what the GUI stores in a text item's ``font``/``font_key``.
FONT_FACES = {
    "Sans": FONT_CANDIDATES,
    "Serif": ("Georgia,Times New Roman,Times,DejaVu Serif,liberation serif,"
              "Noto Serif,FreeSerif,serif"),
    "Mono": ("Consolas,Courier New,Courier,DejaVu Sans Mono,liberation mono,"
             "Noto Sans Mono,FreeMono,monospace"),
}
FONT_ORDER = ["Sans", "Serif", "Mono"]
DEFAULT_FONT = FONT_ORDER[0]

# a text item's height may be tuned between these fractions of its planned size
TEXT_SIZE_MIN = 0.5
TEXT_SIZE_MAX = 3.0

_fonts = {}


def resolve_face(face):
    """Map a font face key (``"Serif"``) or candidate string to a SysFont list."""
    if not face:
        return FONT_CANDIDATES
    if isinstance(face, (tuple, list)):
        return tuple(face)
    return FONT_FACES.get(face, face)


def font(size, bold=False, face=None):
    key = (int(size), bool(bold), resolve_face(face))
    f = _fonts.get(key)
    if f is None:
        if not pygame.font.get_init():
            pygame.font.init()
        f = pygame.font.SysFont(key[2], int(size), bold=bold)
        _fonts[key] = f
    return f


def text_font(el, size=None):
    """The font a text element (title / caption / label / free box) draws with.

    Every text element may carry a ``font`` (a :data:`FONT_FACES` key or a
    SysFont candidate string) and ``bold``; ``size`` overrides ``el["size"]``.
    """
    px = int(size if size is not None else el.get("size", LABEL_SIZE))
    return font(max(6, px), bool(el.get("bold")), el.get("font"))


# --------------------------------------------------------------------------- #
# image loading + dominant color extraction (pure python, no numpy)
# --------------------------------------------------------------------------- #
def load_image(path):
    """Load an image (PSD/PSB flattened through Photoshop first)."""
    p = Path(path)
    if p.suffix.lower() in PSD_EXTS:
        import turnaround_sheet as ts
        p = ts.flatten_psd(p)
    return pygame.image.load(str(p))


def _sample_pixels(img, samples=4000):
    """A deterministic grid subsample of RGBA pixels (alpha < 128 skipped)."""
    w, h = img.get_size()
    if w <= 0 or h <= 0:
        return []
    data = pygame.image.tobytes(img, "RGBA")
    step = max(1, int(math.sqrt((w * h) / float(max(1, samples)))))
    out = []
    for y in range(0, h, step):
        row = y * w * 4
        for x in range(0, w, step):
            i = row + x * 4
            if data[i + 3] < 128:
                continue
            out.append((data[i], data[i + 1], data[i + 2]))
    return out


def _dist2(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _kmeans(pixels, count, iters=15, min_dist=18.0):
    """Deterministic k-means: farthest-point seeding, Lloyd iterations.

    Returns the cluster centers sorted by population, with near-duplicates
    (closer than ``min_dist``) merged into the larger cluster.
    """
    n = len(pixels)
    if not n:
        return []
    k = max(1, min(int(count), n))
    mean = [sum(p[i] for p in pixels) / float(n) for i in range(3)]
    centers = [max(pixels, key=lambda p: _dist2(p, mean))]
    while len(centers) < k:
        best, best_d = None, -1.0
        for p in pixels:
            d = min(_dist2(p, c) for c in centers)
            if d > best_d:
                best, best_d = p, d
        if best is None or best_d <= 0:
            break
        centers.append(best)

    assign = [0] * n
    for _ in range(max(1, iters)):
        changed = False
        for idx, p in enumerate(pixels):
            bi, bd = 0, _dist2(p, centers[0])
            for ci in range(1, len(centers)):
                d = _dist2(p, centers[ci])
                if d < bd:
                    bi, bd = ci, d
            if assign[idx] != bi:
                assign[idx] = bi
                changed = True
        sums = [[0.0, 0.0, 0.0] for _ in centers]
        sizes = [0] * len(centers)
        for idx, p in enumerate(pixels):
            ci = assign[idx]
            sums[ci][0] += p[0]
            sums[ci][1] += p[1]
            sums[ci][2] += p[2]
            sizes[ci] += 1
        for ci, s in enumerate(sizes):
            if s:
                centers[ci] = (sums[ci][0] / s, sums[ci][1] / s, sums[ci][2] / s)
        if not changed:
            break

    sizes = [0] * len(centers)
    for idx in assign:
        sizes[idx] += 1
    ranked = sorted(((centers[ci], sizes[ci]) for ci in range(len(centers))
                     if sizes[ci] > 0),
                    key=lambda cs: (-cs[1], cs[0]))
    kept = []
    for center, size in ranked:
        if all(_dist2(center, c) > min_dist ** 2 for c, _s in kept):
            kept.append((center, size))
    return [(int(round(c[0])), int(round(c[1])), int(round(c[2])))
            for c, _s in kept[:k]]


def dominant_colors(img, count=6, samples=4000):
    """The ``count`` dominant colors of a surface, most common first."""
    return _kmeans(_sample_pixels(img, samples), count)


def dominant_colors_of(path, count=6, samples=4000):
    return dominant_colors(load_image(path), count, samples)


def per_image_palettes(paths, count=6, samples=4000):
    """``[(label, [color, ...]), ...]`` - one palette per input image."""
    out = []
    for p in paths:
        colors = dominant_colors_of(p, count, samples)
        if colors:
            out.append((Path(p).stem, colors))
    return out


def merged_palette(paths, count=6, samples=4000):
    """One palette across every image (samples pooled, then clustered)."""
    pool = []
    for p in paths:
        pool.extend(_sample_pixels(load_image(p), samples))
    return _kmeans(pool, count)


# --------------------------------------------------------------------------- #
# colors + shapes
# --------------------------------------------------------------------------- #
def hex_color(c):
    return "#%02X%02X%02X" % (int(c[0]), int(c[1]), int(c[2]))


def parse_color(text):
    """``#RRGGBB`` / ``RRGGBB`` / ``r,g,b`` -> (r, g, b)."""
    t = str(text).strip().lstrip("#")
    if "," in t:
        parts = [int(float(v)) for v in t.split(",")]
        if len(parts) != 3:
            raise ValueError("expected r,g,b")
        return tuple(max(0, min(255, v)) for v in parts)
    if len(t) != 6:
        raise ValueError("expected #RRGGBB")
    return tuple(int(t[i:i + 2], 16) for i in (0, 2, 4))


def luminance(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast_ink(c, light=INK_LIGHT, dark=INK_DARK):
    """Readable text color on top of ``c`` (Microsoft's 5:1-ish rule)."""
    return light if luminance(c) < 118 else dark


def rect_pts(x, y, w, h):
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def circle_pts(x, y, w, h, segs=48):
    cx, cy = x + w / 2.0, y + h / 2.0
    rx, ry = w / 2.0, h / 2.0
    pts = []
    for i in range(segs):
        a = i / float(segs) * 2.0 * math.pi
        pts.append((cx + math.cos(a) * rx, cy + math.sin(a) * ry))
    return pts


def rounded_pts(x, y, w, h, segs_per_corner=7):
    r = min(w, h) * 0.22
    corners = ((x + w - r, y + r, -90.0, 0.0),
               (x + w - r, y + h - r, 0.0, 90.0),
               (x + r, y + h - r, 90.0, 180.0),
               (x + r, y + r, 180.0, 270.0))
    pts = []
    for cx, cy, a0, a1 in corners:
        for s in range(segs_per_corner):
            a = math.radians(a0 + (a1 - a0) * s / (segs_per_corner - 1.0))
            pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
    return pts


def shape_pts(shape, x, y, w, h):
    if shape == "circle":
        return circle_pts(x, y, w, h)
    if shape == "rounded":
        return rounded_pts(x, y, w, h)
    return rect_pts(x, y, w, h)


def gradient_strip(colors, w, h):
    """A horizontal linear gradient through ``colors`` (evenly spaced stops)."""
    w, h = max(1, int(w)), max(1, int(h))
    surf = pygame.Surface((w, h))
    n = len(colors)
    if n == 0:
        surf.fill((0, 0, 0))
        return surf
    if n == 1:
        surf.fill(tuple(int(v) for v in colors[0]))
        return surf
    for x in range(w):
        t = x / float(max(1, w - 1)) * (n - 1)
        i = min(n - 2, int(t))
        k = t - i
        a, b = colors[i], colors[i + 1]
        col = (int(round(a[0] + (b[0] - a[0]) * k)),
               int(round(a[1] + (b[1] - a[1]) * k)),
               int(round(a[2] + (b[2] - a[2]) * k)))
        pygame.draw.line(surf, col, (x, 0), (x, h - 1))
    return surf


# --------------------------------------------------------------------------- #
# layout planner (the single source of truth for PNG + Photoshop)
# --------------------------------------------------------------------------- #
def _block_grid(n, orientation, cols):
    """(columns, rows) of swatches for one palette."""
    n = max(1, n)
    if orientation == "vertical":
        return 1, n
    if orientation == "grid":
        c = max(1, min(int(cols), n))
        return c, int(math.ceil(n / float(c)))
    return n, 1


def _thumbnail_layout(surfaces, max_w, max_h):
    """Sizes for a centred row of thumbnails inside ``max_w`` px.

    ``surfaces`` is ``[(surface, name), ...]``; the row always fits ``max_w``
    (each image is scaled down until the whole row fits) and no image is taller
    than ``max_h``. Returns ``[{"surf", "name", "w", "h"}, ...]``.
    """
    n = len(surfaces)
    if not n:
        return []
    gap = IMAGE_GAP
    avail = max(16, max_w - gap * (n - 1))
    per = avail / float(n)
    out = []
    for surf, name in surfaces:
        iw, ih = surf.get_size()
        if iw <= 0 or ih <= 0:
            continue
        scale = min(per / iw, max_h / float(ih))
        out.append({"surf": surf, "name": name,
                    "w": max(1, int(round(iw * scale))),
                    "h": max(1, int(round(ih * scale)))})
    return out


def plan_layout(palettes, shape="square", orientation="horizontal", cols=4,
                swatch=96, gap=16, labels=True, gradients=True, captions=True,
                bg=DOC_BG_DEFAULT, title="", dpi=150, doc_name=DOC_NAME,
                images=None, image_size=None, font=None, texts=None):
    """Plan the sheet. Returns a spec dict shared by both renderers.

    ``palettes`` is ``[(label, [(r, g, b), ...]), ...]``. ``images`` (optional)
    is a list aligned with ``palettes`` whose entries are the source surfaces
    (either a bare surface or a ``(surface, name)`` pair, or a list of those):
    each palette then gets its image(s) laid out *above* its swatches.
    ``image_size`` caps a thumbnail's height (default: the swatch size).

    ``font`` picks the face for the title / captions / hex labels (a
    :data:`FONT_FACES` key). ``texts`` is an optional list of free, editable
    text boxes -- each ``{"id", "text", "cx", "y", "size", "color", "bold",
    "font"}`` in sheet pixels (the GUI builds these); they are stored in
    ``spec["texts"]`` and rendered by both renderers just like the labels.

    Coordinates are sheet pixels; ``strip["png"]``/``image["png"]`` are filled
    in later by ``save_strip_pngs``. Every movable element carries an ``id`` and
    ``dx``/``dy`` offsets (see ``spec_items``/``set_offset``) so the GUI can drag
    them around without re-planning the layout.
    """
    shape = shape if shape in SHAPES else "square"
    orientation = orientation if orientation in ORIENTATIONS else "horizontal"
    font_key = font if font in FONT_FACES else DEFAULT_FONT
    swatch = max(16, int(swatch))
    gap = max(0, int(gap))
    bg = tuple(int(v) for v in bg)
    img_h = max(16, int(image_size)) if image_size else swatch

    label_band = LABEL_BAND if labels else 0
    strip_band = (STRIP_BAND + STRIP_GAP) if gradients else 0
    caption_band = CAPTION_BAND if captions else 0

    entries = []
    for i, (label, colors) in enumerate(palettes):
        colors = [tuple(int(v) for v in c) for c in colors]
        if not colors:
            continue
        surfaces = []
        if images and i < len(images) and images[i]:
            for item in images[i]:
                if item is None:
                    continue
                surf, name = item if isinstance(item, (tuple, list)) else (item,
                                                                           label)
                if surf is not None:
                    surfaces.append((surf, name))
        entries.append((label, colors, surfaces))
    if not entries:
        raise ValueError("no colors to lay out (are the images readable?)")

    blocks = []
    for label, colors, surfaces in entries:
        c, r = _block_grid(len(colors), orientation, cols)
        bw = c * swatch + (c - 1) * gap
        bh = r * swatch + (r - 1) * gap
        thumbs = _thumbnail_layout(surfaces, bw, img_h)
        img_band = (max(t["h"] for t in thumbs) + IMAGE_GAP) if thumbs else 0
        blocks.append({"label": label, "colors": colors, "cols": c, "rows": r,
                       "w": bw, "h": bh, "thumbs": thumbs, "img_band": img_band})
    # Palettes in a row share one baseline (tallest image band wins) so stacked
    # / vertical sheets keep their swatch rows aligned; a horizontal sheet gives
    # each block its own band so a short image does not leave a huge gap.
    row_img_band = max(b["img_band"] for b in blocks)
    for b in blocks:
        b["band"] = (row_img_band if orientation in ("vertical", "grid")
                     else b["img_band"])
    max_img_band = max(b["band"] for b in blocks)

    body_gap = COLUMN_GAP if orientation == "vertical" else BLOCK_GAP
    body_w = (sum(b["w"] for b in blocks) + body_gap * (len(blocks) - 1)
              if orientation == "vertical" else max(b["w"] for b in blocks))
    for b in blocks:
        b["block_h"] = (caption_band + b["band"] + b["h"]
                        + label_band + strip_band)
    if orientation == "vertical":
        block_h = max(b["block_h"] for b in blocks)
        body_h = block_h
    else:
        block_h = max(b["block_h"] for b in blocks)
        body_h = (sum(b["block_h"] for b in blocks)
                  + body_gap * (len(blocks) - 1))

    title_band = (TITLE_SIZE + 28) if title else 8
    W = int(MARGIN * 2 + body_w)
    H = int(MARGIN * 2 + title_band + body_h)
    if W * H > MAX_OUTPUT_PIXELS:
        raise ValueError("layout too large (%dx%d px)" % (W, H))

    spec = {"w": W, "h": H, "dpi": int(dpi), "doc_name": doc_name,
            "bg": list(bg), "shape": shape, "orientation": orientation,
            "margin": MARGIN, "title_band": title_band, "font": font_key,
            "palettes": [], "title": None, "texts": []}

    if title:
        spec["title"] = {"id": "title", "text": str(title), "cx": W / 2.0,
                         "y": MARGIN + TITLE_SIZE, "size": TITLE_SIZE,
                         "color": list(contrast_ink(bg)), "bold": True,
                         "font": font_key, "dx": 0, "dy": 0}

    top = MARGIN + title_band
    x_cursor = MARGIN
    y_cursor = top
    for pi, b in enumerate(blocks):
        if orientation == "vertical":
            bx = x_cursor
            by = y_cursor
            x_cursor += b["w"] + COLUMN_GAP
        else:
            bx = MARGIN
            by = y_cursor
            y_cursor += b["block_h"] + BLOCK_GAP

        pal = {"label": b["label"],
               "layer_name": "Palette %d - %s" % (pi + 1, b["label"]),
               "swatches": [], "caption": None, "strip": None, "images": []}
        pal["caption"] = {
            "id": "cap:%d" % pi,
            "text": str(b["label"]).upper(), "cx": bx + b["w"] / 2.0,
            "y": by + CAPTION_SIZE + 4, "size": CAPTION_SIZE,
            "color": list(DEFAULT_CAPTION_COLOR), "bold": True,
            "font": font_key, "dx": 0, "dy": 0,
        } if captions else None

        if b["thumbs"]:
            row_w = (sum(t["w"] for t in b["thumbs"])
                     + IMAGE_GAP * (len(b["thumbs"]) - 1))
            tx = bx + (b["w"] - row_w) / 2.0
            for ti, t in enumerate(b["thumbs"]):
                pal["images"].append({
                    "id": "img:%d:%d" % (pi, ti),
                    "x": tx, "y": by + caption_band, "w": t["w"], "h": t["h"],
                    "name": "Image %d - %s" % (pi + 1, t["name"]),
                    "png": None, "surf": t["surf"], "dx": 0, "dy": 0})
                tx += t["w"] + IMAGE_GAP

        grid_top = by + caption_band + b["band"]
        for i, color in enumerate(b["colors"]):
            col = i % b["cols"]
            row = i // b["cols"]
            sx = bx + col * (swatch + gap)
            sy = grid_top + row * (swatch + gap)
            entry = {"id": "sw:%d:%d" % (pi, i), "x": sx, "y": sy,
                     "w": swatch, "h": swatch, "shape": shape,
                     "color": list(color), "hex": hex_color(color),
                     "dx": 0, "dy": 0}
            if labels:
                entry["label_cx"] = sx + swatch / 2.0
                entry["label_y"] = sy + swatch + LABEL_SIZE + 4
                entry["label_size"] = LABEL_SIZE
                entry["label_color"] = list(contrast_ink(bg))
                entry["label_font"] = font_key
            pal["swatches"].append(entry)

        if gradients:
            sy = grid_top + b["h"] + label_band + STRIP_GAP
            pal["strip"] = {"id": "grad:%d" % pi, "x": bx, "y": sy, "w": b["w"],
                            "h": STRIP_BAND,
                            "colors": [list(c) for c in b["colors"]],
                            "name": "Gradient %d - %s" % (pi + 1, b["label"]),
                            "png": None, "dx": 0, "dy": 0}
        spec["palettes"].append(pal)
    spec["texts"] = normalize_texts(texts, bg, font_key)
    return spec


def normalize_texts(texts, bg=DOC_BG_DEFAULT, font=None):
    """Clean a list of free text boxes into spec-ready dicts.

    Each entry keeps whatever the caller supplied and fills in sensible defaults;
    ``id`` is forced to ``txt:<index>`` so positions survive a rebuild.
    """
    out = []
    for i, t in enumerate(texts or []):
        t = dict(t or {})
        el = {"id": "txt:%d" % i, "text": str(t.get("text", "")),
              "cx": float(t.get("cx", 0.5)), "y": float(t.get("y", 0.5)),
              "size": int(t.get("size", TITLE_SIZE)),
              "color": list(t.get("color", contrast_ink(bg))),
              "bold": bool(t.get("bold")),
              "font": t.get("font", font or DEFAULT_FONT),
              "dx": int(t.get("dx", 0)), "dy": int(t.get("dy", 0))}
        out.append(el)
    return out


def save_strip_pngs(spec):
    """Rasterize the gradient strips *and* image thumbnails next to the sheet.

    Also drops the transient ``surf`` surfaces from the spec so it stays
    JSON-serializable (both renderers reload the written PNGs instead).
    """
    STRIP_DIR.mkdir(parents=True, exist_ok=True)
    for i, pal in enumerate(spec["palettes"]):
        strip = pal.get("strip")
        if strip:
            colors = [tuple(c) for c in strip["colors"]]
            surf = gradient_strip(colors, strip["w"], strip["h"])
            out = STRIP_DIR / ("strip_%02d_%dx%d.png"
                               % (i, strip["w"], strip["h"]))
            pygame.image.save(surf, str(out))
            strip["png"] = str(out)
        for j, img in enumerate(pal.get("images", [])):
            surf = img.pop("surf", None)
            if surf is None:
                continue
            size = (max(1, int(img["w"])), max(1, int(img["h"])))
            if surf.get_size() != size:
                surf = pygame.transform.smoothscale(surf, size)
            out = STRIP_DIR / ("image_%02d_%d_%dx%d.png"
                               % (i, j, size[0], size[1]))
            pygame.image.save(surf, str(out))
            img["png"] = str(out)
    return spec


# --------------------------------------------------------------------------- #
# movable items (the GUI drags these around; both renderers honour dx/dy)
# --------------------------------------------------------------------------- #
def _off(el):
    return int(el.get("dx", 0)), int(el.get("dy", 0))


def spec_items(spec):
    """Every movable element as ``{"id", "kind", "el", "x", "y", "w", "h"}``.

    Coordinates already include the element's ``dx``/``dy`` offset, so the GUI
    can hit-test, hover and drag them without touching the layout maths.
    """
    items = []

    def add(el, kind, x, y, w, h):
        dx, dy = _off(el)
        items.append({"id": el["id"], "kind": kind, "el": el,
                      "dx": dx, "dy": dy,
                      "x": x + dx, "y": y + dy, "w": w, "h": h})

    title = spec.get("title")
    if title:
        f = text_font(title)
        w, h = f.size(title["text"])
        add(title, "title", title["cx"] - w / 2.0, title["y"] - h / 2.0, w, h)

    for pi, pal in enumerate(spec["palettes"]):
        cap = pal.get("caption")
        if cap:
            f = text_font(cap)
            w, h = f.size(cap["text"])
            add(cap, "caption", cap["cx"] - w / 2.0, cap["y"] - h / 2.0, w, h)
        for im in pal.get("images", []):
            add(im, "image", im["x"], im["y"], im["w"], im["h"])
        for sw in pal["swatches"]:
            add(sw, "swatch", sw["x"], sw["y"], sw["w"], sw["h"])
            if sw.get("label_cx") is not None:
                f = text_font(sw, sw.get("label_size", LABEL_SIZE))
                w, h = f.size(sw["hex"])
                add(sw, "label", sw["label_cx"] - w / 2.0,
                    sw["label_y"] - h / 2.0, w, h)
        strip = pal.get("strip")
        if strip:
            add(strip, "strip", strip["x"], strip["y"], strip["w"], strip["h"])

    for el in spec.get("texts", []):
        f = text_font(el)
        w, h = f.size(el.get("text") or " ")
        add(el, "text", el["cx"] - w / 2.0, el["y"] - h / 2.0, w, h)
    return items


def _find_item(spec, item_id):
    if item_id == "title":
        return spec.get("title")
    for el in spec.get("texts", []):
        if el.get("id") == item_id:
            return el
    for pal in spec["palettes"]:
        for el in pal.get("images", []):
            if el.get("id") == item_id:
                return el
        for sw in pal["swatches"]:
            if sw.get("id") == item_id:
                return sw
        for el in (pal.get("caption"), pal.get("strip")):
            if el is not None and el.get("id") == item_id:
                return el
    return None


def set_offset(spec, item_id, dx, dy):
    """Move one item by ``(dx, dy)`` sheet pixels from its planned position."""
    el = _find_item(spec, item_id)
    if el is None:
        return False
    el["dx"], el["dy"] = int(round(dx)), int(round(dy))
    return True


def get_offset(spec, item_id):
    el = _find_item(spec, item_id)
    return (0, 0) if el is None else _off(el)


def apply_offsets(spec, offsets):
    """Re-apply a ``{id: (dx, dy)}`` map (used after a rebuild)."""
    for item_id, delta in (offsets or {}).items():
        set_offset(spec, item_id, delta[0], delta[1])
    return spec


def clear_offsets(spec):
    """Snap every dragged item back to its planned position."""
    if spec.get("title"):
        spec["title"]["dx"] = spec["title"]["dy"] = 0
    for el in spec.get("texts", []):
        el["dx"] = el["dy"] = 0
    for pal in spec["palettes"]:
        for el in pal.get("images", []):
            el["dx"] = el["dy"] = 0
        for sw in pal["swatches"]:
            sw["dx"] = sw["dy"] = 0
        for el in (pal.get("caption"), pal.get("strip")):
            if el is not None:
                el["dx"] = el["dy"] = 0
    return spec


# --------------------------------------------------------------------------- #
# editing text items (the GUI's click-select-type + font controls)
# --------------------------------------------------------------------------- #
# A "text item" is any element carrying on-sheet words: the title, a palette
# caption, a free text box (all with a ``text`` field) or a swatch's hex label
# (which lives on the swatch as ``hex``). These helpers give the GUI one small
# surface to edit them through, so it never has to know which is which.
def text_items(spec):
    """Every editable text element as ``{"id", "kind", "el", "field", "text"}``.

    ``field`` names the dict key the words live in (``"text"`` for title /
    caption / free box, ``"hex"`` for a hex label). ``kind`` is a short label
    the GUI can show ("title", "caption", "label", "text").
    """
    items = []
    title = spec.get("title")
    if title:
        items.append({"id": "title", "kind": "title", "el": title,
                      "field": "text", "text": title["text"]})
    for pal in spec["palettes"]:
        cap = pal.get("caption")
        if cap:
            items.append({"id": cap["id"], "kind": "caption", "el": cap,
                          "field": "text", "text": cap["text"]})
        for sw in pal["swatches"]:
            if sw.get("label_cx") is not None:
                items.append({"id": sw["id"], "kind": "label", "el": sw,
                              "field": "hex", "text": sw["hex"]})
    for el in spec.get("texts", []):
        items.append({"id": el["id"], "kind": "text", "el": el,
                      "field": "text", "text": el["text"]})
    return items


def find_text_item(spec, item_id):
    """The :func:`text_items` entry for ``item_id`` (or ``None``)."""
    for it in text_items(spec):
        if it["id"] == item_id:
            return it
    return None


def set_text(spec, item_id, text):
    """Set the words of a text item in place. Returns the item or ``None``."""
    it = find_text_item(spec, item_id)
    if it is None:
        return None
    it["el"][it["field"]] = str(text)
    return it


# how much a "size" control moves an item's font (a fraction of the planned px)
def text_size(spec, item_id):
    it = find_text_item(spec, item_id)
    if it is None:
        return None
    el = it["el"]
    if it["kind"] == "label":
        return float(el.get("label_size", LABEL_SIZE))
    return float(el.get("size", LABEL_SIZE))


def set_text_size(spec, item_id, size):
    """Set a text item's font size (px), clamped to a sane band."""
    it = find_text_item(spec, item_id)
    if it is None:
        return False
    size = max(6, int(round(size)))
    el = it["el"]
    if it["kind"] == "label":
        el["label_size"] = size
    else:
        el["size"] = size
    return True


def set_text_font(spec, item_id, face):
    """Set a text item's font face (a :data:`FONT_FACES` key or candidate list)."""
    it = find_text_item(spec, item_id)
    if it is None:
        return False
    if it["kind"] == "label":
        it["el"]["label_font"] = face
    else:
        it["el"]["font"] = face
    return True


def set_text_bold(spec, item_id, bold=None):
    """Toggle (or set) a text item's bold flag."""
    it = find_text_item(spec, item_id)
    if it is None:
        return False
    el = it["el"]
    el["bold"] = (not el.get("bold")) if bold is None else bool(bold)
    return True


def add_text(spec, text="Text", cx=None, y=None, size=None, color=None,
             bold=False, font=None):
    """Append a free, movable text box to ``spec["texts"]`` and return it."""
    texts = spec.setdefault("texts", [])
    el = {"id": "txt:%d" % len(texts), "text": str(text),
          "cx": float(spec["w"] / 2.0 if cx is None else cx),
          "y": float(spec["h"] * 0.3 if y is None else y),
          "size": int(size if size is not None else TITLE_SIZE),
          "color": list(color if color is not None else contrast_ink(spec["bg"])),
          "bold": bool(bold),
          "font": font or spec.get("font", DEFAULT_FONT),
          "dx": 0, "dy": 0}
    texts.append(el)
    return el


def remove_text(spec, item_id):
    """Remove a free text box (title/captions/labels cannot be deleted)."""
    texts = spec.get("texts", [])
    for i, el in enumerate(texts):
        if el.get("id") == item_id:
            texts.pop(i)
            return True
    return False



def _blit_png(surf, entry):
    """Blit a saved PNG (strip/thumbnail) at its offset position."""
    png = entry.get("png")
    if not png or not Path(png).exists():
        return
    dx, dy = _off(entry)
    surf.blit(pygame.image.load(png), (int(entry["x"] + dx),
                                       int(entry["y"] + dy)))


def render_spec(spec):
    surf = pygame.Surface((spec["w"], spec["h"]))
    surf.fill(tuple(spec["bg"]))

    title = spec.get("title")
    if title:
        dx, dy = _off(title)
        img = text_font(title).render(title["text"], True,
                                      tuple(title["color"]))
        surf.blit(img, img.get_rect(center=(int(title["cx"] + dx),
                                            int(title["y"] + dy))))

    # Draw in explicit z-ordered passes so a text label can never be painted
    # over: images first, then swatches, then every caption/label on top, then
    # the gradient strips. (Within a pass the draw order does not matter.)
    for pal in spec["palettes"]:
        for im in pal.get("images", []):
            _blit_png(surf, im)

    for pal in spec["palettes"]:
        for sw in pal["swatches"]:
            color = tuple(sw["color"])
            dx, dy = _off(sw)
            x, y = int(sw["x"] + dx), int(sw["y"] + dy)
            w, h = int(sw["w"]), int(sw["h"])
            if sw["shape"] == "circle":
                pygame.draw.circle(surf, color, (x + w // 2, y + h // 2),
                                   min(w, h) // 2)
            elif sw["shape"] == "rounded":
                pygame.draw.rect(surf, color, (x, y, w, h),
                                 border_radius=int(min(w, h) * 0.22))
            else:
                pygame.draw.rect(surf, color, (x, y, w, h))

    # captions and hex labels are the topmost text: nothing can cover them
    for pal in spec["palettes"]:
        cap = pal.get("caption")
        if cap:
            dx, dy = _off(cap)
            img = text_font(cap).render(cap["text"], True,
                                        tuple(cap["color"]))
            surf.blit(img, img.get_rect(center=(int(cap["cx"] + dx),
                                                int(cap["y"] + dy))))
        for sw in pal["swatches"]:
            if sw.get("label_cx") is None:
                continue
            dx, dy = _off(sw)
            el = {"size": sw.get("label_size", LABEL_SIZE),
                  "font": sw.get("label_font"), "bold": False}
            img = text_font(el).render(
                sw["hex"], True, tuple(sw.get("label_color",
                                              DEFAULT_LABEL_COLOR)))
            surf.blit(img, img.get_rect(
                center=(int(sw["label_cx"] + dx),
                        int(sw["label_y"] + dy))))

    # free text boxes sit above everything (they are the user's own annotations)
    for el in spec.get("texts", []):
        if not str(el.get("text", "")).strip():
            continue
        dx, dy = _off(el)
        img = text_font(el).render(str(el["text"]), True, tuple(el["color"]))
        surf.blit(img, img.get_rect(center=(int(el["cx"] + dx),
                                            int(el["y"] + dy))))

    for pal in spec["palettes"]:
        if pal.get("strip"):
            _blit_png(surf, pal["strip"])
    return surf


# --------------------------------------------------------------------------- #
# the whole pipeline + Photoshop hand-off
# --------------------------------------------------------------------------- #
def _load_images(paths):
    """``[(stem, surface_or_None), ...]`` - each input loaded once."""
    out = []
    for p in paths:
        try:
            out.append((Path(p).stem, load_image(p)))
        except Exception:                                      # noqa: BLE001
            out.append((Path(p).stem, None))
    return out


def build_sheet(paths, swatches=6, shared=False, shape="square",
                orientation="horizontal", cols=4, swatch=96, gap=16,
                labels=True, gradients=True, captions=True,
                bg=DOC_BG_DEFAULT, title="", dpi=150, doc_name=DOC_NAME,
                show_images=True, image_size=None, offsets=None, font=None,
                texts=None):
    """Extract colors, plan the layout and render the PNG.

    With ``show_images`` (the default) every image is also shown as a thumbnail
    above its palette's swatches. ``offsets`` is an optional ``{id: (dx, dy)}``
    map (see ``set_offset``) applied on top of the planned positions - that is
    how the GUI keeps hand-moved items across a rebuild. ``font`` picks the
    title/caption/label face and ``texts`` adds free, editable text boxes (see
    :func:`plan_layout`).

    Returns ``(surface, spec, palettes)`` with the strip/thumbnail PNGs saved
    and the spec ready to hand to ``send_to_photoshop``.
    """
    loaded = _load_images(paths)
    if shared:
        pool = []
        for _stem, img in loaded:
            if img is not None:
                pool.extend(_sample_pixels(img))
        colors = _kmeans(pool, swatches)
        label = title or "All images"
        palettes = [(label, colors)] if colors else []
        images = [[(img, stem) for stem, img in loaded if img is not None]]
    else:
        palettes, images = [], []
        for stem, img in loaded:
            if img is None:
                continue
            colors = dominant_colors(img, swatches)
            if colors:
                palettes.append((stem, colors))
                images.append([(img, stem)])
    if not palettes:
        raise ValueError("no colors found in %d image(s)" % len(paths))
    spec = plan_layout(palettes, shape=shape, orientation=orientation,
                       cols=cols, swatch=swatch, gap=gap, labels=labels,
                       gradients=gradients, captions=captions, bg=bg,
                       title=title, dpi=dpi, doc_name=doc_name,
                       images=images if show_images else None,
                       image_size=image_size, font=font, texts=texts)
    save_strip_pngs(spec)
    if offsets:
        apply_offsets(spec, offsets)
    return render_spec(spec), spec, palettes


def _json_drop(obj):
    """Fallback for ``json.dumps`` - surfaces must never reach the JSX."""
    return None


def build_palette_jsx(spec):
    """Fill the JSX template with this spec (JSON is valid JS source)."""
    template = (HERE / "photoshop_palette.jsx").read_text(encoding="utf-8-sig")
    payload = json.dumps(spec, default=_json_drop).replace("\\", "/")
    return template % payload


def aco_entries(palettes):
    """Palette colours -> ``[(r, g, b, "Name"), ...]`` for :func:`write_aco`.

    Accepts either the ``[(label, [(r, g, b), ...]), ...]`` list that
    :func:`build_sheet` returns, or the spec's ``spec["palettes"]`` list of
    dicts (each carrying a ``label`` and ``swatches[*]["color"]``).
    """
    entries = []
    for pal in palettes:
        if isinstance(pal, dict):
            label = str(pal.get("label") or "Palette")[:20]
            colors = [s.get("color") for s in pal.get("swatches", [])]
        else:
            label, colors = pal
            label = str(label)[:20]
        for i, c in enumerate(colors, 1):
            if not c:
                continue
            entries.append((int(c[0]), int(c[1]), int(c[2]),
                            "%s %d" % (label, i)))
    return entries


def import_swatches(spec, path=None):
    """Write the palette as a Photoshop ``.aco`` swatch file.

    This is how the colours get into Photoshop's Swatches panel: the panel
    itself is **not scriptable** in Photoshop 2026 (``addSwatch`` and the
    swatch ``make``/``set`` Action Manager events are all unavailable), so the
    file is written next to the sheet and loaded with a single click --
    Swatches panel menu > Load Swatches (you can also just drag the ``.aco``
    onto the panel).

    ``spec`` may be the layout dict from :func:`build_sheet` or a plain
    ``[(label, [colors]), ...]`` palette list. Returns the path written.
    """
    palettes = spec.get("palettes", []) if isinstance(spec, dict) else spec
    entries = aco_entries(palettes)
    if not entries:
        raise ValueError("no palette colours to export")
    return write_aco(entries, path if path is not None else ACO_DEFAULT)


def send_to_photoshop(spec, doc_name=None, aco_path=None):
    """Rebuild the layout as a native Photoshop document via the bridge.

    It also writes the palette as a Photoshop ``.aco`` swatch file
    automatically (see :func:`import_swatches`) and appends its path to the
    returned message, so the colours can be loaded into the Swatches panel
    with one click.

    Returns ``(ok, message)``; never raises for scripting problems (they come
    back as ``ok=False`` with the bridge's error text).
    """
    note = ""
    try:
        aco = import_swatches(spec, aco_path)
        note = "  |  swatches: %s  (Swatches panel > Load Swatches)" % aco
    except Exception as exc:                                   # noqa: BLE001
        note = "  |  swatch file skipped: %s" % exc
    if doc_name:
        spec["doc_name"] = doc_name
    payload = build_palette_jsx(spec)
    try:
        import photoshop_bridge
        text = photoshop_bridge.run(payload).strip()
    except Exception as exc:                                   # noqa: BLE001
        return False, "Photoshop error: %s%s" % (exc, note)
    line = next((ln for ln in text.splitlines()
                 if ln.startswith("__result__")), text)
    return "BRIDGE ERROR" not in text, line + note


def write_aco(entries, path):
    """Write a Photoshop Color Swatch file (``.aco``, v1 + named v2 block).

    ``entries`` is ``[(r, g, b, "Name"), ...]``. Photoshop's Swatches panel is
    not scriptable here (the ``addSwatch``/``make swatch`` Action Manager events
    are unavailable in Photoshop 2026), so this is the way to get the colors
    into the panel: Swatches panel menu > Load Swatches > pick the ``.aco``.
    """
    out = bytearray()
    n = len(entries)
    out += struct.pack(">HH", 1, n)
    for r, g, b, _name in entries:
        out += struct.pack(">HHHHH", 2, r * 257, g * 257, b * 257, 0)
    out += struct.pack(">HH", 2, n)
    for r, g, b, name in entries:
        raw = str(name).encode("utf-8")[:255]
        pstr = bytes([len(raw)]) + raw
        if len(pstr) % 2:
            pstr += b"\x00"
        out += struct.pack(">HHHH", 2, r * 257, g * 257, b * 257)
        out += struct.pack(">I", len(pstr)) + pstr
    p = Path(path)
    p.write_bytes(bytes(out))
    return p


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Pull dominant colors out of images and build a swatch "
                    "sheet for Photoshop.")
    ap.add_argument("inputs", nargs="+", help="images (png/jpg/bmp/tga/gif/webp/psd)")
    ap.add_argument("--swatches", type=int, default=6,
                    help="colors per image (or in total with --shared)")
    ap.add_argument("--shared", action="store_true",
                    help="one merged palette for every image instead of one "
                         "palette per image")
    ap.add_argument("--shape", choices=SHAPES, default="square")
    ap.add_argument("--orientation", choices=ORIENTATIONS, default="horizontal")
    ap.add_argument("--cols", type=int, default=4,
                    help="swatches per row in grid orientation (default 4)")
    ap.add_argument("--swatch-size", type=int, default=96,
                    help="swatch size in pixels (default 96)")
    ap.add_argument("--gap", type=int, default=16, help="space between swatches")
    ap.add_argument("--no-labels", action="store_true",
                    help="hide the hex labels under the swatches")
    ap.add_argument("--no-gradients", action="store_true",
                    help="hide the gradient strip under each palette")
    ap.add_argument("--no-captions", action="store_true",
                    help="hide the image name above each palette")
    ap.add_argument("--no-images", action="store_true",
                    help="hide the source images above the swatches")
    ap.add_argument("--image-size", type=int, default=None,
                    help="max image thumbnail height in px (default: --swatch-size)")
    ap.add_argument("--bg", default="#182028",
                    help="sheet background, #RRGGBB or r,g,b (default #182028)")
    ap.add_argument("--title", default="", help="header text on the sheet")
    ap.add_argument("--dpi", type=int, default=150,
                    help="resolution recorded on the Photoshop document")
    ap.add_argument("--out", default=str(OUT_DEFAULT), help="output PNG path")
    ap.add_argument("--aco", help="also write a .aco swatch file here")
    ap.add_argument("--send", action="store_true",
                    help="also rebuild the sheet natively in Photoshop")
    ap.add_argument("--doc-name", default=DOC_NAME)
    a = ap.parse_args(argv)

    for raw in a.inputs:
        if Path(raw).suffix.lower() not in ALL_EXTS:
            ap.error("unsupported input: %s" % raw)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sheet, spec, palettes = build_sheet(
        a.inputs, a.swatches, a.shared, a.shape, a.orientation, a.cols,
        a.swatch_size, a.gap, not a.no_labels, not a.no_gradients,
        not a.no_captions, parse_color(a.bg) if a.bg else DOC_BG_DEFAULT,
        a.title, a.dpi, a.doc_name, not a.no_images, a.image_size)

    out = Path(a.out)
    pygame.image.save(sheet, str(out))
    print("wrote %s  (%dx%d px, %d palette(s), %d swatches)"
          % (out, spec["w"], spec["h"], len(palettes),
             sum(len(c) for _l, c in palettes)))

    if a.aco:
        write_aco(aco_entries(palettes), a.aco)
        print("wrote %s  (load it in the Swatches panel > Load Swatches)"
              % a.aco)

    if a.send:
        ok, msg = send_to_photoshop(spec, a.doc_name)
        print("[photoshop] %s" % msg)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
