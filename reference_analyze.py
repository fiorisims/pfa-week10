"""Turn a reference image (PNG/JPG/PSD/PSB) into environment data for Unreal.

Runs in the normal system Python (pygame is installed there).  The Unreal
Reference World Builder shells out to this script and reads the JSON from
stdout, so this module never imports ``unreal``.

Output: palette, a colour/luminance grid, a mood description and water level.
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

import pygame  # noqa: E402

import palette_sheet as ps  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif", ".webp", ".psd", ".psb"}


def _pixel_bytes(surface):
    return pygame.image.tobytes(surface, "RGBA")


def _sample_cell(data, width, x0, y0, x1, y1):
    r = g = b = 0
    n = 0
    for y in range(y0, y1):
        row = y * width * 4
        for x in range(x0, x1):
            i = row + x * 4
            if data[i + 3] < 128:
                continue
            r += data[i]
            g += data[i + 1]
            b += data[i + 2]
            n += 1
    if not n:
        return None
    return (r // n, g // n, b // n)


def _rgb_to_hsv(color):
    r, g, b = [v / 255.0 for v in color]
    high = max(r, g, b)
    low = min(r, g, b)
    delta = high - low
    if delta == 0:
        hue = 0.0
    elif high == r:
        hue = (60.0 * ((g - b) / delta)) % 360.0
    elif high == g:
        hue = 60.0 * ((b - r) / delta) + 120.0
    else:
        hue = 60.0 * ((r - g) / delta) + 240.0
    saturation = 0.0 if high == 0 else delta / high
    return hue, saturation, high


def _mood(palette, average_luminance, average_saturation, warm_ratio, green_ratio):
    if average_luminance < 0.32:
        name = "Twilight" if average_saturation > 0.28 else "Moonlit"
    elif average_luminance > 0.72 and average_saturation < 0.22:
        name = "Overcast"
    elif green_ratio > 0.30:
        name = "Verdant"
    elif warm_ratio > 0.38:
        name = "Golden Hour" if average_luminance > 0.52 else "Ember"
    elif average_saturation > 0.42:
        name = "Vivid"
    else:
        name = "Soft Daylight"
    accent = max(palette, key=lambda entry: entry["population"] * (0.6 + entry["saturation"]))
    return {
        "name": name,
        "average_luminance": round(average_luminance, 3),
        "average_saturation": round(average_saturation, 3),
        "warm_ratio": round(warm_ratio, 3),
        "green_ratio": round(green_ratio, 3),
        "accent": accent["hex"],
    }


def analyze(path, grid=24, colors=6):
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(str(image_path))
    if image_path.suffix.lower() not in IMAGE_EXTS:
        raise ValueError("unsupported image type: " + image_path.suffix)

    surface = ps.load_image(str(image_path))
    width, height = surface.get_size()
    if width < 2 or height < 2:
        raise ValueError("image is too small")
    data = _pixel_bytes(surface)

    palette_rgb = ps.dominant_colors(surface, count=colors)
    palette = []
    for color in palette_rgb:
        # Population is approximated from the sample grid below; equal weight is
        # fine here because k-means already sorted the centres by cluster size.
        hue, saturation, value = _rgb_to_hsv(color)
        palette.append(
            {
                "hex": ps.hex_color(color),
                "rgb": [int(color[0]), int(color[1]), int(color[2])],
                "hue": round(hue, 1),
                "saturation": round(saturation, 3),
                "value": round(value, 3),
                "population": round(1.0 / max(1, len(palette_rgb)), 3),
            }
        )

    grid = max(4, min(48, int(grid)))
    cell_w = width / float(grid)
    cell_h = height / float(grid)
    lum = []
    sat = []
    hue_grid = []
    cells = []
    for gy in range(grid):
        y0 = int(round(gy * cell_h))
        y1 = max(y0 + 1, int(round((gy + 1) * cell_h)))
        lum_row, sat_row, hue_row, cell_row = [], [], [], []
        for gx in range(grid):
            x0 = int(round(gx * cell_w))
            x1 = max(x0 + 1, int(round((gx + 1) * cell_w)))
            color = _sample_cell(data, width, x0, y0, min(x1, width), min(y1, height))
            if color is None:
                color = (0, 0, 0)
            hue, saturation, _value = _rgb_to_hsv(color)
            lum_row.append(round(ps.luminance(color) / 255.0, 4))
            sat_row.append(round(saturation, 4))
            hue_row.append(round(hue, 1))
            cell_row.append(list(color))
        lum.append(lum_row)
        sat.append(sat_row)
        hue_grid.append(hue_row)
        cells.append(cell_row)

    flat_lum = [v for row in lum for v in row]
    flat_sat = [v for row in sat for v in row]
    warm = cool = green = 0
    for row in hue_grid:
        for hue in row:
            if 35.0 <= hue <= 165.0:
                green += 1
            if hue <= 45.0 or hue >= 315.0:
                warm += 1
            if 165.0 < hue < 315.0:
                cool += 1
    total = max(1, len(flat_lum))
    average_luminance = sum(flat_lum) / total
    average_saturation = sum(flat_sat) / total
    mood = _mood(palette, average_luminance, average_saturation, warm / total, green / total)

    sorted_lum = sorted(flat_lum)
    water_percentile = sorted_lum[int(len(sorted_lum) * 0.22)]
    # Sky/backdrop band: the top 25% of a concept image is usually sky, not
    # walkable terrain.  Keep the data available so the Unreal side can choose.
    terrain_top = int(round(height * 0.24))

    return {
        "image": str(image_path.resolve()),
        "name": image_path.stem,
        "width": width,
        "height": height,
        "palette": palette,
        "mood": mood,
        "water_level": round(water_percentile, 4),
        "terrain_start_row": terrain_top,
        "grid": {
            "size": grid,
            "luminance": lum,
            "saturation": sat,
            "hue": hue_grid,
            "colors": cells,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze a reference image for Unreal")
    parser.add_argument("image")
    parser.add_argument("-o", "--out", default=None)
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--colors", type=int, default=6)
    parser.add_argument("--layers-dir", default=None,
                        help="for PSD/PSB: export and analyse each layer into this folder")
    parser.add_argument("--setting", default=None, help="kind of place (reference_setting), default: read it")
    parser.add_argument("--keywords", default=None,
                        help="layer-name vocabulary JSON (created with defaults if missing)")
    args = parser.parse_args(argv)
    try:
        if args.layers_dir and Path(args.image).suffix.lower() in (".psd", ".psb"):
            import reference_layers
            keywords = args.keywords or str(Path(__file__).with_name("rwb_layer_keywords.json"))
            result = reference_layers.analyze_psd_layers(
                args.image, args.layers_dir, keywords, args.grid, analyze, colors=args.colors)
        else:
            result = analyze(args.image, grid=args.grid, colors=args.colors)
        # single-image scene (perspective + depth): always prepared, used when the
        # flat-image build is chosen.  Never fatal.
        import rwb_progress
        rwb_progress.report(0.05, "Measuring depth and the horizon")
        try:
            import reference_scene
            base_dir = Path(args.layers_dir) if args.layers_dir else Path(__file__).with_name("_rwb_cache") / Path(args.image).stem
            source = Path(result.get("layer_dir", "")) / "composite.png" if result.get("layers") is not None else Path(args.image)
            result["scene"] = reference_scene.prepare(str(source), base_dir / "scene")
        except Exception as scene_error:
            result["scene"] = {"error": str(scene_error)}
        # what objects are in the picture, and how many of each (never fatal)
        try:
            import reference_inventory
            keywords = args.keywords or str(Path(__file__).with_name("rwb_layer_keywords.json"))
            result["inventory"] = reference_inventory.build(result, str(source), base_dir / "inventory", keywords,
                                                              args.setting)
        except Exception as inventory_error:
            import traceback
            result["inventory"] = {"error": str(inventory_error), "trace": traceback.format_exc()[-600:]}
    except Exception as error:
        print("ANALYZE_ERROR: " + str(error), file=sys.stderr)
        return 1
    text = json.dumps(result)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
