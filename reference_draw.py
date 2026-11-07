"""Draw a shape on your painting and send it to Unreal as a new 3D object.

Opened by the Reference World Builder panel (Draw a new shape...) and stays open while you keep
working in Unreal.  You draw directly on the picture, so where you draw is where the object goes
in the world, using the same camera as everything else.

Texture comes from the painting either way:
* **Under the shape** - the paint you drew over;
* **Picked area** - drag a box over any part of the painting (the T tool) and the shape is
  filled with new paint stitched from that area's real brushstrokes.

Each Send writes a job into ``<out>/outbox``; Unreal watches that folder.

    python reference_draw.py --scene SCENE_CACHE_DIR --out DRAWN_DIR
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

import reference_objects as ro
import reference_strokes as rs
import turnaround_gui as tg
from turnaround_gui import Button, Slider

WINDOW = (1240, 840)
CANVAS_AREA = pygame.Rect(16, 62, 900, 762)
PANEL = pygame.Rect(930, 62, 296, 762)
TILE_SIZE = 512
METHODS = (("inflate", "Puffy"), ("lathe", "Round"), ("slab", "Flat front"), ("ground", "Lying down"))
TOOLS = (("lasso", "Outline"), ("brush", "Brush"), ("erase", "Eraser"), ("pick", "Pick texture"))
UNDO_LIMIT = 30
MIN_PIXELS = 40


def load_scene(scene_dir):
    """(texture HxWx3, work size (H, W)) from a reference_scene cache."""
    scene_dir = Path(scene_dir)
    meta = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    surf = pygame.image.load(str(scene_dir / "scene_image.png"))
    rgb = pygame.Surface(surf.get_size(), 0, 24)
    rgb.blit(surf, (0, 0))
    w, h = rgb.get_size()
    texture = np.frombuffer(pygame.image.tobytes(rgb, "RGB"), np.uint8).reshape(h, w, 3).copy()
    return texture, (int(meta["height"]), int(meta["width"]))


def next_number(out_dir):
    numbers = [int(p.stem.split("_")[-1]) for p in Path(out_dir).glob("drawn_*.png") if p.stem.split("_")[-1].isdigit()]
    return max(numbers, default=0) + 1


def compose(mask, texture, work_size, tile, out_dir, method, number=None, replaces=None):
    """Turn a drawn mask into a job for Unreal.  ``mask`` is bool, any size.

    Returns the job dict (written to the outbox), or None if the shape is too small.
    """
    out_dir = Path(out_dir)
    (out_dir / "outbox").mkdir(parents=True, exist_ok=True)
    H, W = work_size
    work = ro._resize(mask.astype(np.float32), W, H) > 0.5
    if int(work.sum()) < MIN_PIXELS:
        return None
    place = ro.describe_mask(work)
    TH, TW = texture.shape[:2]
    full = ro._resize(mask.astype(np.float32), TW, TH)
    sx, sy = TW / float(W), TH / float(H)
    x0, y0, x1, y1 = place["bbox"]
    bx0, by0 = int(x0 * sx), int(y0 * sy)
    bx1, by1 = max(bx0 + 2, min(TW, int(math.ceil(x1 * sx)))), max(by0 + 2, min(TH, int(math.ceil(y1 * sy))))
    crop = np.zeros((by1 - by0, bx1 - bx0, 4), np.uint8)
    if tile is not None:
        ys, xs = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
        crop[..., :3] = np.clip(rs.sample(tile, xs, ys, wrap=True), 0, 255).astype(np.uint8)
    else:
        crop[..., :3] = texture[by0:by1, bx0:bx1]
    crop[..., 3] = np.clip(full[by0:by1, bx0:bx1] * 255.0, 0, 255).astype(np.uint8)
    number = number or next_number(out_dir)
    name = "drawn_%03d" % number
    path = out_dir / (name + ".png")
    pygame.image.save(pygame.image.frombuffer(crop.tobytes(), (bx1 - bx0, by1 - by0), "RGBA"), str(path))
    job = {
        "id": name, "name": name, "label": "Drawn %d" % number, "category": "drawn", "method": method,
        "cutout": str(path.resolve()), "bbox_px": place["bbox"], "bbox": place["bbox"],
        "contact_u": place["contact_u"], "contact_v": place["contact_v"], "top_v": place["top_v"],
        "width_frac": place["width_frac"], "area": place["area"],
        "centre_u": (x0 + x1) / 2.0 / W, "depth_eye": None,
        "aspect": (y1 - y0) / float(max(1, x1 - x0)),
    }
    if replaces:
        job["replaces"] = replaces
    temp = out_dir / "outbox" / (name + ".tmp")
    temp.write_text(json.dumps(job), encoding="utf-8")
    temp.replace(out_dir / "outbox" / (name + ".json"))       # Unreal never sees half a file
    return job


class Sketch:
    """The drawing itself: an alpha mask at canvas size, with undo."""

    def __init__(self, size):
        self.size = size
        self.surface = pygame.Surface(size, pygame.SRCALPHA, 32)
        self.undo_stack = []

    def checkpoint(self):
        self.undo_stack.append(self.surface.copy())
        del self.undo_stack[:-UNDO_LIMIT]

    def undo(self):
        if self.undo_stack:
            self.surface = self.undo_stack.pop()

    def clear(self):
        self.checkpoint()
        self.surface.fill((0, 0, 0, 0))

    def fill_outline(self, points):
        if len(points) >= 3:
            pygame.draw.polygon(self.surface, (255, 255, 255, 255), points)

    def stroke(self, a, b, radius, erase=False):
        color = (0, 0, 0, 0) if erase else (255, 255, 255, 255)
        steps = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / max(1.0, radius / 2.0)))
        for i in range(steps + 1):
            t = i / float(steps)
            pygame.draw.circle(self.surface, color, (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t)), radius)

    def mask(self):
        """Bool array, rows then columns."""
        return pygame.surfarray.array_alpha(self.surface).T > 127

    def empty(self):
        return not self.mask().any()


class App:
    def __init__(self, scene_dir, out_dir, title="Shape Sketch", replace=None, start_cutout=None, start_bbox=None):
        self.out_dir = Path(out_dir)
        self.screen = pygame.display.set_mode(WINDOW)
        pygame.display.set_caption(title)
        self.title = title
        self.texture, self.work_size = load_scene(scene_dir)
        TH, TW = self.texture.shape[:2]
        self.scale = min(CANVAS_AREA.w / float(TW), CANVAS_AREA.h / float(TH))
        size = (int(TW * self.scale), int(TH * self.scale))
        self.canvas = pygame.Rect(CANVAS_AREA.x + (CANVAS_AREA.w - size[0]) // 2,
                                  CANVAS_AREA.y + (CANVAS_AREA.h - size[1]) // 2, *size)
        raw = pygame.image.frombuffer(self.texture.tobytes(), (TW, TH), "RGB")
        self.base = pygame.transform.smoothscale(raw, size).convert()
        self.dim = self.base.copy()
        self.dim.fill((150, 150, 165), special_flags=pygame.BLEND_RGB_MULT)
        self.sketch = Sketch(size)
        self.tool = "lasso"
        self.method = "inflate"
        self.radius = 18
        self.drag = None
        self.last = None
        self.pick_rect = None
        self.tile = None
        self.tile_view = None
        self.stats = None
        self.sent = 0
        self.preview = None
        self.status = "Draw around a shape on your painting, then press Send to Unreal."
        self.running = True
        self.widgets = []
        self.replace = replace
        if start_cutout and start_bbox:
            self.load_outline(start_cutout, start_bbox)
        self._layout()
        self._refresh_preview()

    def load_outline(self, cutout, bbox):
        """Start from a model's current outline, so fixing it is easier than redrawing it."""
        art = pygame.image.load(str(cutout)).convert_alpha()
        H, W = self.work_size
        sx, sy = self.canvas.w / float(W), self.canvas.h / float(H)
        x0, y0, x1, y1 = bbox
        size = (max(1, int((x1 - x0) * sx)), max(1, int((y1 - y0) * sy)))
        art = pygame.transform.smoothscale(art, size)
        alpha = pygame.surfarray.array_alpha(art)
        surface = pygame.Surface(size, pygame.SRCALPHA, 32)
        surface.fill((255, 255, 255, 255))
        pygame.surfarray.pixels_alpha(surface)[:] = np.where(alpha > 127, 255, 0)
        self.sketch.surface.blit(surface, (int(x0 * sx), int(y0 * sy)))
        self.status = "This is the model's outline now. Fix it with the Brush and Eraser, then Send."

    # ---- controls ------------------------------------------------------------
    def _layout(self):
        small = tg.font(13, True)
        x, w = PANEL.x + 14, PANEL.w - 28
        half = (w - 8) // 2
        y = PANEL.y + 34
        for i, (key, label) in enumerate(TOOLS):
            self.widgets.append(Button((x + (half + 8) * (i % 2), y + 42 * (i // 2), half, 36), label,
                                       lambda k=key: self.set_tool(k), tfont=small,
                                       selected=lambda k=key: self.tool == k, radius=11))
        y += 94
        self.size_y = y
        self.widgets.append(Slider((x, y + 22, w, 22), lambda: self.radius, self.set_radius, 3, 80))
        y += 62
        self.shape_y = y
        y += 26
        for i, (key, label) in enumerate(METHODS):
            self.widgets.append(Button((x + (half + 8) * (i % 2), y + 42 * (i // 2), half, 36), label,
                                       lambda k=key: self.set_method(k), tfont=small,
                                       selected=lambda k=key: self.method == k, radius=11))
        y += 94
        self.texture_y = y
        y += 26
        self.widgets.append(Button((x, y, half, 36), "Under shape", self.use_under, tfont=small,
                                   selected=lambda: self.tile is None, radius=11))
        self.widgets.append(Button((x + half + 8, y, half, 36), "Picked area", self.use_picked, tfont=small,
                                   selected=lambda: self.tile is not None, radius=11))
        self.swatch = pygame.Rect(x, y + 44, w, 104)
        y += 160
        self.widgets.append(Button((x, y, half, 36), "Undo", self.undo, tfont=small, radius=11))
        self.widgets.append(Button((x + half + 8, y, half, 36), "Clear", self.clear, tfont=small, radius=11))
        y += 46
        self.widgets.append(Button((x, y, w, 50), "Send to Unreal", self.send, base=(60, 170, 90)))

    def set_tool(self, key):
        self.tool = key
        if key == "pick":
            self.status = "Drag a box over the paint you want the shape made from."

    def set_radius(self, value):
        self.radius = int(value)

    def set_method(self, key):
        self.method = key

    def use_under(self):
        self.tile = None
        self.tile_view = None
        self._refresh_preview()
        self.status = "The shape uses the paint it was drawn over."

    def use_picked(self):
        if self.tile is None and self.pick_rect is None:
            self.set_tool("pick")
            return
        if self.tile is None:
            self.make_tile(self.pick_rect)

    def undo(self):
        self.sketch.undo()
        self._refresh_preview()

    def clear(self):
        self.sketch.clear()
        self._refresh_preview()

    # ---- texture -------------------------------------------------------------
    def to_texture(self, rect):
        """Canvas rect -> texture pixel box."""
        s = 1.0 / self.scale
        return (int(rect.x * s), int(rect.y * s), int(rect.right * s), int(rect.bottom * s))

    def make_tile(self, rect):
        """Stitch a seamless paint swatch from the picked area's real strokes."""
        x0, y0, x1, y1 = self.to_texture(rect)
        region = self.texture[y0:y1, x0:x1]
        if region.shape[0] < 24 or region.shape[1] < 24:
            self.status = "Pick a bigger area: there aren't enough brushstrokes in that."
            return False
        self.pick_rect = rect
        self.stats = rs.read_strokes(region)
        patch = min(rs.patch_size(self.stats), min(region.shape[:2]) // 2 * 2)
        tile = rs.quilt(region.astype(np.float32), TILE_SIZE, max(16, patch), seed=len(region))
        if tile is None:
            self.status = "Could not read brushstrokes from that area. Try a bigger one."
            return False
        self.tile = tile
        view = np.clip(tile, 0, 255).astype(np.uint8)
        self.tile_view = pygame.image.frombuffer(view.tobytes(), (TILE_SIZE, TILE_SIZE), "RGB").convert()
        self.tool = "lasso"
        self.status = rs.describe(self.stats) + ". Now draw a shape."
        self._refresh_preview()
        return True

    def _refresh_preview(self):
        """The drawn shape filled with its texture, over the dimmed painting."""
        mask_surf = self.sketch.surface
        if self.tile_view is not None:
            s = max(1, int(TILE_SIZE * self.scale))
            tile = pygame.transform.smoothscale(self.tile_view, (s, s))
            fill = pygame.Surface(self.canvas.size, pygame.SRCALPHA, 32)
            for y in range(0, self.canvas.h, s):
                for x in range(0, self.canvas.w, s):
                    fill.blit(tile, (x, y))
        else:
            fill = self.base.convert_alpha()
        fill.blit(mask_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.preview = fill

    # ---- sending -------------------------------------------------------------
    def send(self):
        job = compose(self.sketch.mask(), self.texture, self.work_size, self.tile, self.out_dir, self.method,
                      replaces=self.replace)
        if job is not None:
            self.replace = None                  # later sends from this window are new shapes
        if job is None:
            self.status = "Draw a shape first." if self.sketch.empty() else "That shape is too small to build."
            return None
        self.sent += 1
        self.status = "Sent %s to Unreal (%d so far). It appears in the world in a few seconds." % (job["label"], self.sent)
        self.sketch.clear()
        self.sketch.undo_stack.clear()
        self._refresh_preview()
        return job

    # ---- input ---------------------------------------------------------------
    def _local(self, pos):
        return (pos[0] - self.canvas.x, pos[1] - self.canvas.y)

    def handle(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type == pygame.KEYDOWN:
            keys = {pygame.K_l: "lasso", pygame.K_b: "brush", pygame.K_e: "erase", pygame.K_t: "pick"}
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key in keys:
                self.set_tool(keys[event.key])
            elif event.key == pygame.K_z:
                self.undo()
            elif event.key == pygame.K_RETURN:
                self.send()
            return
        for widget in self.widgets:
            widget.handle(event)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.canvas.collidepoint(event.pos):
            p = self._local(event.pos)
            self.drag = [p]
            self.last = p
            if self.tool != "pick":
                self.sketch.checkpoint()
            if self.tool in ("brush", "erase"):
                self.sketch.stroke(p, p, self.radius, self.tool == "erase")
                self._refresh_preview()
        elif event.type == pygame.MOUSEMOTION and self.drag is not None:
            p = self._local(event.pos)
            p = (max(0, min(self.canvas.w - 1, p[0])), max(0, min(self.canvas.h - 1, p[1])))
            if self.tool in ("brush", "erase"):
                self.sketch.stroke(self.last, p, self.radius, self.tool == "erase")
                self._refresh_preview()
            self.drag.append(p)
            self.last = p
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.drag is not None:
            points, self.drag = self.drag, None
            if self.tool == "lasso":
                self.sketch.fill_outline(points)
                self._refresh_preview()
            elif self.tool == "pick" and len(points) >= 2:
                xs, ys = [p[0] for p in points], [p[1] for p in points]
                rect = pygame.Rect(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                if rect.w > 4 and rect.h > 4:
                    self.make_tile(rect)

    # ---- drawing -------------------------------------------------------------
    def draw(self):
        screen = self.screen
        tg.draw_backdrop(screen)
        tg.title_bar(screen, self.title)
        screen.blit(self.dim, self.canvas)
        screen.blit(self.preview, self.canvas)
        outline = pygame.mask.from_surface(self.sketch.surface, 127).outline()
        if len(outline) > 2:
            pygame.draw.lines(screen, (255, 255, 255), True, [(x + self.canvas.x, y + self.canvas.y) for x, y in outline], 2)
        if self.pick_rect is not None and self.tile is not None:
            pygame.draw.rect(screen, (120, 230, 255), self.pick_rect.move(self.canvas.topleft), 2)
        if self.drag and len(self.drag) > 1:
            pts = [(x + self.canvas.x, y + self.canvas.y) for x, y in self.drag]
            if self.tool == "lasso":
                pygame.draw.lines(screen, (255, 240, 120), False, pts, 2)
            elif self.tool == "pick":
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                pygame.draw.rect(screen, (120, 230, 255), (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)), 2)
        mouse = pygame.mouse.get_pos()
        if self.canvas.collidepoint(mouse) and self.tool in ("brush", "erase"):
            pygame.draw.circle(screen, (255, 255, 255), mouse, self.radius, 1)
        screen.blit(tg.glass_panel(PANEL.size), PANEL)
        heading, small = tg.font(15, True), tg.font(13, True)
        x = PANEL.x + 14
        tg.text_shadow(screen, "Tool", heading, tg.WHITE, (x, PANEL.y + 10))
        tg.text_shadow(screen, "Brush size: %d" % self.radius, small, tg.WHITE, (x, self.size_y))
        tg.text_shadow(screen, "3D shape", heading, tg.WHITE, (x, self.shape_y))
        tg.text_shadow(screen, "Texture", heading, tg.WHITE, (x, self.texture_y))
        pygame.draw.rect(screen, (40, 60, 80), self.swatch, border_radius=8)
        if self.tile_view is not None:
            screen.blit(self.tile_view, self.swatch, (0, 0, self.swatch.w, self.swatch.h))
        else:
            tg.text_shadow(screen, "painting under the shape", small, tg.WHITE, self.swatch.center, center=True)
        pygame.draw.rect(screen, (255, 255, 255), self.swatch, 1, border_radius=8)
        for widget in self.widgets:
            widget.draw(screen)
        tg.text_shadow(screen, self.status, small, tg.WHITE, (CANVAS_AREA.x, WINDOW[1] - 16 - 8))
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                self.handle(event)
            self.draw()
            clock.tick(60)


def main():
    parser = argparse.ArgumentParser(description="Draw a shape on a painting for Unreal")
    parser.add_argument("--scene", required=True, help="reference_scene cache folder")
    parser.add_argument("--out", required=True, help="folder for drawn shapes (Unreal watches its outbox)")
    parser.add_argument("--title", default="Shape Sketch")
    parser.add_argument("--replace", default=None, help="key of the model this drawing replaces")
    parser.add_argument("--start-cutout", default=None, help="begin from this model's outline")
    parser.add_argument("--start-bbox", default=None, help="x0,y0,x1,y1 of that outline")
    args = parser.parse_args()
    # reference_objects asks for a headless SDL at import; this tool is the one that needs a window
    if os.environ.get("SDL_VIDEODRIVER") == "dummy" and not os.environ.get("RWB_DRAW_HEADLESS"):
        del os.environ["SDL_VIDEODRIVER"]
    pygame.init()
    bbox = [float(v) for v in args.start_bbox.split(",")] if args.start_bbox else None
    App(args.scene, args.out, args.title, args.replace, args.start_cutout, bbox).run()
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
