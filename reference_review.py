"""Check what was recognised in a painting before any models are built.

Opened by the Reference World Builder panel (Check labels...).  Every detected object is drawn as a
box over the painting.  Click one, then click what it really is, or remove it.  Done saves the
corrections to ``<inventory cache>/label_fixes.json``; the panel then re-reads the picture, reusing
the cached detection, so a wrong label costs one click and no model is built from it.

    python reference_review.py --scene SCENE_CACHE_DIR --inventory INVENTORY_CACHE_DIR
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

import reference_objects as ro
import turnaround_gui as tg
from turnaround_gui import Button

WINDOW = (1240, 840)
CANVAS_AREA = pygame.Rect(16, 62, 900, 762)
PANEL = pygame.Rect(930, 62, 296, 762)
CATEGORIES = ("tree", "pine", "bush", "rock", "log", "mushroom", "house", "building", "castle", "tower",
              "wall", "fence", "bridge", "gate", "pillar", "lantern", "furniture", "vehicle", "boat",
              "character", "creature", "prop_card", "sign", "lily")
NORMAL, CHANGED, REMOVED, SELECTED = (90, 210, 255), (255, 214, 90), (240, 80, 80), (255, 255, 255)


def load_items(inventory_dir):
    """Detected objects as drawn here: [{key, category, detected, bbox_frac, score}]."""
    data = json.loads((Path(inventory_dir) / "recognized.json").read_text(encoding="utf-8"))
    return [{"key": ro.fix_key(obj), "category": obj["category"], "detected": obj["label"],
             "bbox_frac": obj["bbox_frac"], "score": obj["score"]} for obj in data["objects"]]


class App:
    def __init__(self, scene_dir, inventory_dir):
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW)
        pygame.display.set_caption("Check labels")
        self.inventory_dir = inventory_dir
        self.items = load_items(inventory_dir)
        self.fixes = ro.load_label_fixes(inventory_dir)
        art = pygame.image.load(str(Path(scene_dir) / "scene_image.png")).convert()
        scale = min(CANVAS_AREA.w / art.get_width(), CANVAS_AREA.h / art.get_height())
        size = (int(art.get_width() * scale), int(art.get_height() * scale))
        self.picture = pygame.transform.smoothscale(art, size)
        self.canvas = self.picture.get_rect(center=CANVAS_AREA.center)
        self.selected = None
        self.running = True
        self.saved = False
        self.status = "Click a box to fix it. Close the window to leave everything as it was."
        self.widgets = []
        self._layout()

    # ---- controls ------------------------------------------------------------
    def _layout(self):
        small = tg.font(13, True)
        x, w = PANEL.x + 14, PANEL.w - 28
        half = (w - 8) // 2
        top = PANEL.y + 74
        for i, category in enumerate(CATEGORIES):
            self.widgets.append(Button((x + (half + 8) * (i % 2), top + 38 * (i // 2), half, 32),
                                       category.replace("_", " "), lambda c=category: self.set_category(c),
                                       tfont=small, selected=lambda c=category: self.shown_category() == c, radius=10))
        y = top + 38 * ((len(CATEGORIES) + 1) // 2) + 8
        self.widgets.append(Button((x, y, half, 36), "Remove", self.remove, tfont=small, radius=11, base=(190, 70, 70)))
        self.widgets.append(Button((x + half + 8, y, half, 36), "Restore", self.restore, tfont=small, radius=11))
        self.widgets.append(Button((x, PANEL.bottom - 64, w, 50), "Done", self.done, base=(60, 170, 90)))

    def state(self, item):
        return self.fixes.get(item["key"]) or {}

    def category_of(self, item):
        return self.state(item).get("category") or item["category"]

    def shown_category(self):
        return None if self.selected is None else self.category_of(self.selected)

    def pick(self, pos):
        """The object under a canvas point; the smallest wins, so boxes inside boxes can be reached."""
        u, v = (pos[0] - self.canvas.x) / self.canvas.w, (pos[1] - self.canvas.y) / self.canvas.h
        hits = [item for item in self.items if item["bbox_frac"][0] <= u <= item["bbox_frac"][2]
                and item["bbox_frac"][1] <= v <= item["bbox_frac"][3]]
        return min(hits, key=lambda i: (i["bbox_frac"][2] - i["bbox_frac"][0]) * (i["bbox_frac"][3] - i["bbox_frac"][1]),
                   default=None)

    def set_category(self, category):
        if self.selected is None:
            self.status = "Click a box on the painting first."
            return
        if category == self.selected["category"]:
            self.fixes.pop(self.selected["key"], None)
        else:
            self.fixes[self.selected["key"]] = {"category": category}
        self.status = "This one is now: %s." % category.replace("_", " ")

    def remove(self):
        if self.selected is None:
            self.status = "Click a box on the painting first."
            return
        self.fixes[self.selected["key"]] = {"remove": True}
        self.status = "Removed. No model will be built from it."

    def restore(self):
        if self.selected is not None:
            self.fixes.pop(self.selected["key"], None)
            self.status = "Back to what the detector said."

    def done(self):
        if self.fixes != ro.load_label_fixes(self.inventory_dir):
            ro.save_label_fixes(self.inventory_dir, self.fixes)
            self.saved = True
        self.running = False

    def handle(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                self.remove()
            return
        for widget in self.widgets:
            if widget.handle(event):
                return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.canvas.collidepoint(event.pos):
            self.selected = self.pick(event.pos)
            if self.selected is not None:
                self.status = "Detected as %s (%.0f%%). What is it really?" % (self.selected["detected"], self.selected["score"] * 100)

    # ---- drawing -------------------------------------------------------------
    def box_rect(self, item):
        u0, v0, u1, v1 = item["bbox_frac"]
        return pygame.Rect(self.canvas.x + int(u0 * self.canvas.w), self.canvas.y + int(v0 * self.canvas.h),
                           max(2, int((u1 - u0) * self.canvas.w)), max(2, int((v1 - v0) * self.canvas.h)))

    def draw_item(self, item, label_font):
        state = self.state(item)
        colour = REMOVED if state.get("remove") else CHANGED if state else NORMAL
        rect = self.box_rect(item)
        chosen = item is self.selected
        pygame.draw.rect(self.screen, SELECTED if chosen else colour, rect, 3 if chosen else 2)
        if state.get("remove"):
            pygame.draw.line(self.screen, REMOVED, rect.topleft, rect.bottomright, 2)
            pygame.draw.line(self.screen, REMOVED, rect.topright, rect.bottomleft, 2)
        name = label_font.render("removed" if state.get("remove") else self.category_of(item).replace("_", " "), True, (15, 25, 35))
        tag = pygame.Rect(rect.x, max(self.canvas.y, rect.y - name.get_height() - 2), name.get_width() + 6, name.get_height() + 2)
        pygame.draw.rect(self.screen, SELECTED if chosen else colour, tag)
        self.screen.blit(name, (tag.x + 3, tag.y + 1))

    def draw(self):
        tg.draw_backdrop(self.screen)
        tg.title_bar(self.screen, "Check labels")
        self.screen.blit(self.picture, self.canvas)
        label_font = tg.font(11, True)
        for item in self.items:
            if item is not self.selected:
                self.draw_item(item, label_font)
        if self.selected is not None:
            self.draw_item(self.selected, label_font)
        self.screen.blit(tg.glass_panel(PANEL.size), PANEL)
        tg.text_shadow(self.screen, "What is it?", tg.font(15, True), tg.WHITE, (PANEL.x + 14, PANEL.y + 12))
        tg.text_shadow(self.screen, "%d objects, %d changed" % (len(self.items), len(self.fixes)), tg.font(13, True), tg.WHITE,
                       (PANEL.x + 14, PANEL.y + 40))
        for widget in self.widgets:
            widget.draw(self.screen)
        tg.text_shadow(self.screen, self.status, tg.font(13, True), tg.WHITE, (CANVAS_AREA.x, WINDOW[1] - 14))
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                self.handle(event)
            self.draw()
            clock.tick(60)
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Check what was recognised in a painting")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--inventory", required=True)
    args = parser.parse_args()
    App(args.scene, args.inventory).run()


if __name__ == "__main__":
    main()
