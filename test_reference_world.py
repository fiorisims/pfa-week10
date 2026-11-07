"""Checks for the Reference World Builder's recognition, placement and modelling.

    python test_reference_world.py            offline checks (a few seconds, no models needed)
    python test_reference_world.py --live     also runs the real vision models and Blender on a
                                              picture (downloads ~350 MB of models the first time)
    python test_reference_world.py --live --image PATH    use your own picture

Unreal itself is not started; rwb_layout has no ``unreal`` import so its maths is tested here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import numpy as np  # noqa: E402
import pygame  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ReferenceWorldBuilder" / "Content" / "Python"))

import reference_layers as rl  # noqa: E402
import reference_mesh as rm  # noqa: E402
import reference_objects as ro  # noqa: E402
import rwb_layout  # noqa: E402

failures = []
count = 0


def check(name, condition, detail=""):
    global count
    count += 1
    if not condition:
        failures.append(name)
        print("FAIL  %s  %s" % (name, detail))


def near(a, b, tolerance):
    return abs(a - b) <= tolerance


# --------------------------------------------------------------------------- #
# placement
# --------------------------------------------------------------------------- #
def test_camera():
    cam = rwb_layout.PaintCamera(768, 548, 335, 60.0, 160.0, 8000.0)
    check("focal from fov", near(cam.focal, 665.1, 0.2), cam.focal)
    # picture -> ground -> picture
    for u, v in ((0.5, 0.9), (0.2, 0.75), (0.8, 0.7), (0.5, 0.66)):
        x, y = cam.to_world(u, v)
        back = cam.to_image(x, y)
        check("round trip %.2f,%.2f" % (u, v), back and near(back[0], u, 0.002) and near(back[1], v, 0.002), back)
    # lower in the picture = closer; left of the picture = -X; the world goes into -Y
    near_y = cam.to_world(0.5, 0.95)[1]
    far_y = cam.to_world(0.5, 0.65)[1]
    check("lower in picture is closer", near_y > far_y, (near_y, far_y))
    check("everything is in front of the player", near_y < cam.player_y)
    check("left stays left", cam.to_world(0.1, 0.8)[0] < 0 < cam.to_world(0.9, 0.8)[0])
    # ground above the horizon cannot be seen: it clamps to the far edge, never behind the player
    check("above horizon clamps far", near(cam.player_y - cam.to_world(0.5, 0.3)[1], cam.far, 1.0))
    check("outside the frame has no picture position", cam.to_image(5000.0, cam.player_y - 200.0) is None)
    check("behind the player has no picture position", cam.to_image(0.0, cam.player_y + 100.0) is None)


def _reproject(cam, placed):
    """Where a placed object's base and top land in the picture, from the start position."""
    base_row = cam.horizon + (cam.eye - placed["z"]) * cam.focal / placed["distance"]
    top_row = cam.horizon + (cam.eye - placed["z"] - placed["height"]) * cam.focal / placed["distance"]
    column = cam.W / 2.0 + placed["x"] * cam.focal / placed["distance"]
    return column, base_row, top_row


def test_placement():
    cam = rwb_layout.PaintCamera(768, 548, 335, 60.0, 160.0, 8000.0)
    # the promise of the whole design: placed objects cover the part of the frame they were painted in
    inst = {"contact_u": 0.70, "centre_u": 0.70, "contact_v": 0.80, "top_v": 0.10, "width_frac": 0.2}
    placed = cam.place(inst, "tower")
    column, base, top = _reproject(cam, placed)
    check("column as painted", near(column, 0.70 * 768, 2.0), column)
    check("base row as painted", near(base, 0.80 * 548, 2.0), base)
    check("top row as painted", near(top, 0.10 * 548, 2.0), top)
    check("width as painted", near(placed["width"] * cam.focal / placed["distance"], 0.2 * 768, 1.0))
    # same painted size further back = bigger in the world
    a = cam.place({"contact_u": 0.5, "contact_v": 0.95, "top_v": 0.85, "width_frac": 0.1}, "rock")
    b = cam.place({"contact_u": 0.5, "contact_v": 0.68, "top_v": 0.58, "width_frac": 0.1}, "rock")
    check("further = bigger for the same pixels", b["height"] > a["height"] * 1.5, (a["height"], b["height"]))
    check("faces the start position", near(placed["yaw"], math.degrees(math.atan2(placed["x"], placed["distance"])), 0.01))
    # feet hidden behind something: the depth map overrides the visible bottom
    hidden = cam.place({"contact_u": 0.5, "contact_v": 0.62, "top_v": 0.2, "width_frac": 0.1, "depth_eye": 8.0}, "tree")
    check("depth map wins when feet are hidden", near(hidden["distance"], 8.0 * 160.0, 1.0), hidden["distance"])
    # a butterfly in the air floats; a rock with the same outline stands on the far ground
    fly = {"contact_u": 0.2, "contact_v": 0.34, "top_v": 0.28, "width_frac": 0.05, "depth_eye": 11.0}
    flying = cam.place(fly, "floating")
    check("airborne stays in the air", flying["airborne"] and flying["z"] > cam.eye, flying)
    column, base, _top = _reproject(cam, flying)
    check("airborne base row as painted", near(base, 0.34 * 548, 2.0), base)
    check("ground things never float", cam.place(fly, "rock")["z"] == 0.0)
    # FG / MG / BG tags decide the distance, whatever the picture says
    tagged = cam.place({"contact_u": 0.5, "contact_v": 0.95, "top_v": 0.5, "width_frac": 0.1, "band_t": 0.9}, "tree")
    check("BG tag sends it far", tagged["distance"] > cam.far * 0.7, tagged["distance"])
    # nothing lands outside the map
    wide = cam.place({"contact_u": 0.999, "contact_v": 0.62, "top_v": 0.4, "width_frac": 0.1}, "tree")
    check("stays on the map", abs(wide["x"]) <= cam.limit and abs(wide["y"]) <= cam.size / 2.0, wide)


def test_ground_placement():
    cam = rwb_layout.PaintCamera(768, 548, 335, 60.0, 160.0, 8000.0)
    check("canoe lies flat", cam.lies_flat({"bbox_px": [253, 400, 485, 530]}))
    check("something crossing the horizon stands", not cam.lies_flat({"bbox_px": [404, 278, 485, 399]}))
    placed = cam.place_ground({"bbox_px": [253, 400, 485, 530]})
    near_edge = placed["distance"] - (cam.row_distance(400) - cam.row_distance(530)) * placed["scale_y"] / 2.0
    check("near edge stays where it was painted", near(near_edge, cam.row_distance(530), 1.0), near_edge)
    stretched = cam.place_ground({"bbox_px": [300, 345, 340, 500]})
    check("perspective can't make a lying thing absurdly long", stretched["scale_y"] < 1.0, stretched)
    check("extra copies stay out of the painted view",
          not rwb_layout.scatter_allowed(cam, 0.0, 0.0) and rwb_layout.scatter_allowed(cam, 3800.0, cam.player_y - 400.0))


# --------------------------------------------------------------------------- #
# recognition
# --------------------------------------------------------------------------- #
def test_filtering():
    table = {"a tree": "tree", "a rock": "rock", "a pond": "water"}
    phrases = list(table)
    W, H = 400, 300
    side = 400

    def raw(*rows):
        scores = np.zeros((len(rows), 3), np.float32)
        boxes = np.zeros((len(rows), 4), np.float32)
        for i, (q, score, box) in enumerate(rows):
            scores[i, q] = score
            x0, y0, x1, y1 = box
            boxes[i] = [(x0 + x1) / 2 / side, (y0 + y1) / 2 / side, (x1 - x0) / side, (y1 - y0) / side]
        return scores, boxes

    s, b = raw((0, 0.5, (10, 10, 60, 100)), (0, 0.3, (12, 12, 62, 102)))
    out = ro.filter_boxes(s, b, phrases, table, W, H, side)
    check("duplicate boxes merge, the best survives", len(out) == 1 and near(out[0]["score"], 0.5, 1e-6), out)
    s, b = raw((0, 0.5, (10, 10, 60, 100)), (1, 0.3, (10, 10, 60, 100)))
    check("one box gets one label", len(ro.filter_boxes(s, b, phrases, table, W, H, side)) == 1)
    s, b = raw((0, 0.5, (10, 10, 60, 100)), (1, 0.4, (200, 100, 260, 160)))
    check("separate things both survive", len(ro.filter_boxes(s, b, phrases, table, W, H, side)) == 2)
    s, b = raw((0, 0.09, (10, 10, 60, 100)))
    check("below the score floor is a guess", not ro.filter_boxes(s, b, phrases, table, W, H, side))
    s, b = raw((0, 0.9, (0, 0, 399, 299)))
    check("a box the size of the picture is no object", not ro.filter_boxes(s, b, phrases, table, W, H, side))
    s, b = raw((0, 0.3, (0, 0, 200, 200)), (0, 0.5, (10, 10, 50, 50)), (0, 0.5, (100, 100, 150, 150)))
    out = ro.filter_boxes(s, b, phrases, table, W, H, side)
    check("a box around two trees is a group, not a tree", len(out) == 2, out)
    # tiles map back into picture coordinates and cover the picture
    tiles = ro.tile_boxes(768, 548)
    check("tiles cover the picture", tiles[0][:2] == [0, 0] and tiles[-1][2] >= 767 and tiles[-1][3] >= 547, tiles)
    check("tiles overlap", tiles[0][2] > tiles[1][0], tiles)


def test_masks():
    mask = np.zeros((60, 60), bool)
    mask[10:50, 10:50] = True
    mask[20:24, 20:24] = False                      # a window
    filled = ro.fill_holes(mask)
    check("small holes close", filled[22, 22])
    ring = np.zeros((60, 60), bool)
    ring[5:55, 5:55] = True
    ring[12:48, 12:48] = False                      # something seen through it
    check("a big enclosed region stays open", not ro.fill_holes(ring)[30, 30])
    place = ro.describe_mask(mask)
    check("contact is the bottom of the shape", near(place["contact_v"], 50 / 60.0, 1e-6), place)
    check("box of the shape", place["bbox"] == [10, 10, 50, 50], place)
    big = {"mask": mask, "name": "big"}
    part = np.zeros((60, 60), bool)
    part[15:25, 30:40] = True
    apart = np.zeros((60, 60), bool)
    apart[52:58, 52:58] = True
    kept = ro.drop_parts([{"mask": part, "name": "part"}, big, {"mask": apart, "name": "apart"}])
    names = {d["name"] for d in kept}
    check("parts join the whole, separate things stay", names == {"big", "apart"}, names)
    # scenery
    ground = {"bbox_frac": [0.0, 0.8, 0.5, 1.0], "contact_v": 1.0, "area": 0.1}
    check("ground is scenery", ro.is_scenery(ground, "bush", 0.6))
    boat = {"bbox_frac": [0.3, 0.7, 0.65, 0.97], "contact_v": 0.97, "area": 0.05}
    check("a boat isn't", not ro.is_scenery(boat, "boat", 0.6))
    canopy = {"bbox_frac": [0.2, 0.0, 0.6, 0.45], "contact_v": 0.45, "area": 0.15}
    check("canopy is scenery", ro.is_scenery(canopy, "thorns", 0.6))
    butterfly = {"bbox_frac": [0.15, 0.25, 0.2, 0.33], "contact_v": 0.33, "area": 0.004}
    check("a small flying thing isn't", not ro.is_scenery(butterfly, "floating", 0.6))
    framing = {"bbox_frac": [0.0, 0.0, 0.25, 0.85], "contact_v": 0.85, "area": 0.12}
    check("a framing tree isn't", not ro.is_scenery(framing, "tree", 0.6))
    check("a framing wall of dark is", ro.is_scenery({**framing, "bbox_frac": [0.75, 0.3, 1.0, 1.0]}, "cliff", 0.6))


def test_labels():
    phrases = ["a lily pad", "a butterfly", "a pond"]
    table = {"a lily pad": "lily", "a butterfly": "floating", "a pond": "water"}
    owl = np.array([0.3, 0.1, 0.9])
    clip = np.array([0.4, 0.5, 0.1])
    phrase, _ = ro.fused_label(owl, clip, phrases, table, above_horizon=False)
    check("area words never label an object", phrase == "a lily pad", phrase)
    phrase, _ = ro.fused_label(owl, clip, phrases, table, above_horizon=True)
    check("a lily pad can't be above the horizon", phrase == "a butterfly", phrase)
    check("article stripped", ro.short_label("an archway") == "archway" and ro.short_label("reeds") == "reeds")


def test_psd_names():
    vocab = {"words": {"tree": ("tree", "object"), "shadow": ("ignore", "ignore")}, "phrases": [], "weak": {}, "depth": {}}
    generic = rl.classify("Group 1", vocab)
    check("a generic group isn't ignored (its layers would vanish)", generic["kind"] != "ignore", generic)
    check("a generic name is flagged to be looked at", generic.get("generic") is True, generic)
    check("'Layer 3' too", rl.classify("Layer 3", vocab).get("generic") is True)
    check("real names still work", rl.classify("Big tree", vocab)["category"] == "tree")
    check("paint-only names still skip", rl.classify("Tree shadow", vocab)["kind"] == "ignore")


# --------------------------------------------------------------------------- #
# modelling
# --------------------------------------------------------------------------- #
def blob(h=60, w=40):
    mask = np.zeros((h, w), bool)
    for r in range(h):
        half = int(w / 2 * math.sin(math.pi * (r + 1) / (h + 1)))
        mask[r, w // 2 - half:w // 2 + half] = True
    return mask


def closed(faces):
    """Every edge used exactly twice, in opposite directions = watertight + consistent winding."""
    edges = {}
    for face in faces:
        for a, b in zip(face, list(face[1:]) + [face[0]]):
            edges[(a, b)] = edges.get((a, b), 0) + 1
    return all(edges.get((b, a), 0) == n for (a, b), n in edges.items()), len(edges)


def outward_share(verts, faces):
    """Share of faces whose normal points away from the shape's centre."""
    centre = verts.mean(0)
    out = total = 0
    for face in faces:
        pts = verts[list(face)]
        normal = np.zeros(3)
        for i in range(len(pts)):
            normal += np.cross(pts[i], pts[(i + 1) % len(pts)])
        if np.linalg.norm(normal) < 1e-9:
            continue
        total += 1
        out += float(np.dot(normal, pts.mean(0) - centre)) > 0.0
    return out / float(max(1, total))


def test_mesh():
    mask = blob()
    for name, (verts, faces) in (("inflate", rm.inflate(mask, 0.85)), ("slab", rm.slab(mask, 0.2)), ("lathe", rm.lathe(mask))):
        is_closed, n_edges = closed(faces)
        check("%s is watertight" % name, is_closed, n_edges)
        check("%s faces outward" % name, outward_share(verts, faces) > 0.9, outward_share(verts, faces))
        width, height = np.ptp(verts[:, 0]), np.ptp(verts[:, 2])
        check("%s keeps the painted proportions" % name, near(height / width, 60.0 / 40.0, 0.35), height / width)
    # front is -Y here so it faces +Y in Unreal, toward the player, without mirroring the picture
    verts, faces = rm.inflate(mask, 0.85)
    tl, bl, br = [verts[i] for i in faces[0][:3]]
    normal = np.cross(bl - tl, br - tl)
    check("first face is the front and looks toward -Y", normal[1] < 0, normal)
    # thick parts bulge more than thin parts
    fat = np.zeros((40, 60), bool)
    fat[5:35, 5:30] = True
    fat[18:22, 30:58] = True
    verts, _ = rm.inflate(fat, 1.0)
    front = verts[len(verts) // 2:]
    body = np.abs(front[(front[:, 0] > 8) & (front[:, 0] < 28)][:, 1]).max()
    arm = np.abs(front[front[:, 0] > 45][:, 1]).max()
    check("branches stay thin next to a fat trunk", arm < body * 0.45, (arm, body))
    # a 1-cell-wide rod survives
    rod = np.zeros((50, 9), bool)
    rod[:, 4] = True
    verts, faces = rm.slab(rod, 0.5)
    check("a thin rod isn't lost", len(faces) > 40 and np.ptp(verts[:, 2]) >= 48, len(faces))
    # thin parts survive the downsample
    alpha = np.zeros((800, 400), np.uint8)
    alpha[:, 200:202] = 255
    check("2 px line survives the silhouette grid", rm.silhouette_grid(alpha).any(1).sum() >= rm.GRID_LONG - 1)


def test_ground_shape():
    cam = {"W": 768.0, "H": 548.0, "horizon": 335.0, "focal": 665.1}
    bbox = [253, 400, 485, 530]
    mask = np.zeros((40, 60), bool)
    mask[3:37, 3:57] = True
    verts, faces = rm.ground_shape(mask, bbox, cam)
    is_closed, _ = closed(faces)
    check("ground shape is watertight", is_closed)
    check("ground shape sits on the ground", near(verts[:, 2].min(), 0.0, 1e-6))
    check("wider than tall", np.ptp(verts[:, 2]) < np.ptp(verts[:, 0]))
    # looked at from the camera, the ground shape covers its painted box
    uv = rm.art_uv(verts, "ground", {"focal": cam["focal"], "horizon": cam["horizon"], "W": cam["W"], "bbox": bbox})
    flat = verts[:, 2] <= 1e-9
    check("the picture lands back on the shape", uv[flat, 0].min() < 0.1 and uv[flat, 0].max() > 0.9
          and uv[flat, 1].min() < 0.1 and uv[flat, 1].max() > 0.9, (uv[flat].min(0), uv[flat].max(0)))
    near_row, far_row = verts[verts[:, 1].argmin()], verts[verts[:, 1].argmax()]
    check("the bottom of the picture is the near end", near_row[1] < far_row[1])
    check("method: crossing the horizon means standing",
          rm.method_for("boat", 0.5) == "ground" and rm.method_for("mushroom", 1.0) == "lathe"
          and rm.method_for("mushroom", 0.2) == "inflate" and rm.method_for("rock", 1.0) == "inflate"
          and rm.method_for("wall", 0.5) == "slab")


def test_end_to_end_no_blender():
    pygame.display.init()
    tmp = Path(tempfile.mkdtemp(prefix="rwb_mesh_"))
    try:
        rgba = np.zeros((120, 80, 4), np.uint8)
        rgba[20:110, 20:60] = (200, 40, 40, 255)
        cutout = tmp / "cut.png"
        pygame.image.save(pygame.image.frombuffer(rgba.tobytes(), (80, 120), "RGBA"), str(cutout))
        cam = {"W": 768.0, "H": 548.0, "horizon": 335.0, "focal": 665.1}
        objects = [
            {"id": "a", "cutout": str(cutout), "category": "mushroom", "label": "mushroom", "bbox": [100, 100, 180, 400]},
            {"id": "b", "cutout": str(cutout), "category": "boat", "label": "boat", "bbox": [100, 400, 300, 500]},
            {"id": "c", "cutout": str(cutout), "category": "boat", "label": "standing boat", "bbox": [100, 200, 300, 500]},
        ]
        out = tmp / "m"
        manifest = rm.build_all(objects, out, cam, use_blender=False)
        models = {m["id"]: m for m in manifest["models"]}
        check("every object became a model", set(models) == {"a", "b", "c"}, manifest.get("skipped"))
        check("round thing lathed", models["a"]["method"] == "lathe")
        check("lying thing projected", models["b"]["method"] == "ground")
        check("label can't lay down something crossing the horizon", models["c"]["method"] == "slab")
        check("mesh written", Path(models["a"]["mesh"]).is_file())
        art = pygame.image.load(models["a"]["art"])
        check("art has no background showing at the edge", art.get_at((1, 1))[0] > 150, art.get_at((1, 1)))
        again = rm.build_all(objects, out, cam, use_blender=False)
        check("second build is cached", again["key"] == manifest["key"])
        lines = Path(models["a"]["mesh"]).read_text().splitlines()
        check("OBJ has geometry", sum(1 for l in lines if l.startswith("f ")) > 100)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# brushstrokes
# --------------------------------------------------------------------------- #
def stripes(angle_degrees, size=512, period=14):
    """A picture of parallel paint marks at a known angle (0 = horizontal, y up)."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    a = math.radians(angle_degrees)
    across = -xs * math.sin(a) - ys * math.cos(a)            # distance across the marks, image y down
    wave = np.sin(across * 2.0 * math.pi / period)
    rgb = np.stack([120 + 60 * wave, 90 + 40 * wave, 70 + 30 * wave], -1)
    return rgb.astype(np.float32)


def angle_gap(a, b):
    gap = abs(a - b) % 180.0
    return min(gap, 180.0 - gap)


def _world_builder_functions(*names):
    """Run named functions from world_builder.py without Unreal."""
    import ast
    source = (HERE / "ReferenceWorldBuilder" / "Content" / "Python" / "world_builder.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    return source, nodes


def test_clicks_are_queued():
    import ast
    source, nodes = _world_builder_functions("_queue_action", "_run_queued_actions")
    ticks = []
    scope = {"_actions": {"queue": [], "running": False}, "_ensure_tick": lambda: ticks.append(1)}
    exec(compile(ast.Module(nodes, []), "queue", "exec"), scope)
    queue, run_queued, actions = scope["_queue_action"], scope["_run_queued_actions"], scope["_actions"]
    done = []
    check("a click only queues", queue("a", lambda: done.append("a")) and not done)
    check("the tick is switched on for it", ticks)
    check("a double-click doesn't run it twice", not queue("a", lambda: done.append("again")))
    run_queued()
    check("the tick runs it", done == ["a"] and not actions["queue"])

    def nested():
        done.append("outer")
        check("nothing can be queued during a job", not queue("b", lambda: done.append("b")))
        run_queued()                                    # the progress window ticks during a job
    queue("c", nested)
    queue("d", lambda: done.append("d"))
    run_queued()
    check("a job never starts inside another", done == ["a", "outer"], done)
    run_queued()
    check("the next job waits its turn", done == ["a", "outer", "d"], done)

    def broken():
        raise ValueError("boom")
    queue("e", broken)
    try:
        run_queued()
    except ValueError:
        pass
    check("a failed job doesn't block the panel", actions["running"] is False)
    tree = ast.parse(source)
    body = {n.name: ast.get_source_segment(source, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    check("a button click only queues", "action(target)" not in body["_make_click_callback"].split("def callback")[1])
    check("a row click only queues", "action(target)" not in body["_row_action"].split("def callback")[1])
    check("the watcher stays out of a running job", '_actions["running"]' in body["_watch_for_manually_opened_panel"])
    check("old handlers are never freed", "_retired_callbacks" in body["bind_widget"] and "_retired_callbacks" in body["_refresh_model_rows"])


def test_catalog():
    import reference_catalog as rc
    check("names become words", rc.phrase_for("statue_(sculpture)") == "statue sculpture" and rc.phrase_for("ArmChair_01") == "arm chair",
          (rc.phrase_for("statue_(sculpture)"), rc.phrase_for("ArmChair_01")))
    check("only free licences are allowed", set(rc.ALLOWED_LICENSES) == {"cc0", "by", "by-sa"})

    class FakeClip:
        """Pictures are 'embedded' by their mean colour, so similarity is predictable."""
        def image(self, rgb):
            v = rgb.reshape(-1, 3).mean(0) + 1.0
            return v / np.linalg.norm(v)

    tmp = Path(tempfile.mkdtemp(prefix="rwb_cat_"))
    saved = rc.ONLINE
    try:
        rc.ONLINE = tmp
        (tmp / "thumbs").mkdir()
        pygame.display.init()

        def thumb(name, colour, noise=True):
            rng = np.random.default_rng(len(name))
            arr = np.full((64, 64, 3), colour, np.float32)
            if noise:
                arr += rng.normal(0, 30, arr.shape)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            path = tmp / "thumbs" / (name + ".png")
            pygame.image.save(pygame.image.frombuffer(arr.tobytes(), (64, 64), "RGB"), str(path))
            return path

        red = np.array([1.0, 0.05, 0.05]); red /= np.linalg.norm(red)
        cands = [
            {"source": "objaverse", "id": "r", "name": "red", "thumb_url": "x", "text_score": 0.8},
            {"source": "objaverse", "id": "b", "name": "blue", "thumb_url": "x", "text_score": 0.95},
            {"source": "objaverse", "id": "blank", "name": "blank", "thumb_url": "x", "text_score": 0.99},
            {"source": "objaverse", "id": "gone", "name": "gone", "thumb_url": "x", "text_score": 0.99},
            {"source": "library", "id": "mine.glb", "name": "mine", "thumb_url": None, "text_score": 0.7},
        ]
        thumb("objaverse_r", (220, 40, 40)); thumb("objaverse_b", (40, 40, 220)); thumb("objaverse_blank", (128, 128, 128), False)
        real = rc._thumb

        def fake_thumb(url, name):
            path = tmp / "thumbs" / (name + ".png")
            if not path.is_file():
                raise IOError("HTTP Error 403: Forbidden")
            return path
        rc._thumb = fake_thumb
        try:
            ranked = rc.rank([dict(c) for c in cands], red, FakeClip())
        finally:
            rc._thumb = real
        names = [c["name"] for c in ranked]
        check("the picture decides over the label", names.index("red") < names.index("blue"), names)
        check("a blank preview is dropped", "blank" not in names, names)
        check("a removed model (403) is dropped", "gone" not in names, names)
        check("your own files stay in", "mine" in names, names)
        rc.CREDITS = tmp / "CREDITS.txt"
        m = {"source": "objaverse", "name": "Cage", "author": "Someone", "license": "CC-BY 4.0", "url": "https://x"}
        rc.credit(m)
        rc.credit(m)
        text = (tmp / "CREDITS.txt").read_text(encoding="utf-8")
        check("a used model is credited once", text.count("Cage by Someone") == 1, text)
        rc.credit({"source": "library", "name": "mine"})
        check("your own files need no credit", "mine" not in (tmp / "CREDITS.txt").read_text(encoding="utf-8"))
    finally:
        rc.ONLINE = saved
        rc.CREDITS = saved / "CREDITS.txt"
        shutil.rmtree(tmp, ignore_errors=True)


def test_trees_and_style():
    tall = {"bbox_frac": [0.0, 0.0, 0.26, 0.86]}
    short = {"bbox_frac": [0.1, 0.8, 0.2, 0.9]}
    check("a stump half the picture tall is a tree", ro.sensible_label("tree stump", "log", tall) == ("tree", "tree"))
    check("a real stump stays a stump", ro.sensible_label("tree stump", "log", short)[1] == "log")
    check("other labels are left alone", ro.sensible_label("tower", "tower", tall) == ("tower", "tower"))
    check("style: as painted is full resolution", near(rm.resolution(1.0, rm.GRID_LOW, rm.GRID_LONG), rm.GRID_LONG, 0.01))
    check("style: low poly is coarse", near(rm.resolution(0.0, rm.GRID_LOW, rm.GRID_LONG), rm.GRID_LOW, 0.01))
    middle = rm.resolution(0.5, rm.GRID_LOW, rm.GRID_LONG)
    check("style: the middle is really between", rm.GRID_LOW * 1.5 < middle < rm.GRID_LONG * 0.7, middle)
    pygame.display.init()
    tmp = Path(tempfile.mkdtemp(prefix="rwb_style_"))
    try:
        rgba = np.zeros((200, 120, 4), np.uint8)
        rgba[20:180, 30:90] = (120, 90, 60, 255)
        cutout = tmp / "cut.png"
        pygame.image.save(pygame.image.frombuffer(rgba.tobytes(), (120, 200), "RGBA"), str(cutout))
        cam = {"W": 768.0, "H": 548.0, "horizon": 335.0, "focal": 665.1}
        base = {"id": "a", "cutout": str(cutout), "category": "rock", "label": "rock", "bbox": [100, 100, 180, 400]}
        fine = rm.build_object({**base, "name": "fine", "detail": 1.0}, tmp, cam, 0)
        coarse = rm.build_object({**base, "name": "coarse", "detail": 0.0}, tmp, cam, 1)
        check("low poly is far fewer polygons", coarse["tris_raw"] * 4 < fine["tris_raw"], (coarse["tris_raw"], fine["tris_raw"]))
        check("low poly keeps the proportions", near(coarse["size"][2] / coarse["size"][0], fine["size"][2] / fine["size"][0], 0.6),
              (coarse["size"], fine["size"]))
        one = rm.build_all([{**base, "name": "x", "detail": 1.0}], tmp / "a", cam, use_blender=False)
        other = rm.build_all([{**base, "name": "x", "detail": 0.3}], tmp / "a", cam, use_blender=False)
        check("moving the slider rebuilds", one["key"] != other["key"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_surfaces():
    import reference_surfaces as rf
    check("green reads as grass", max(rf.colour_prior((70, 150, 60)), key=rf.colour_prior((70, 150, 60)).get) == "grass")
    check("clear blue reads as water", max(rf.colour_prior((60, 140, 220)), key=rf.colour_prior((60, 140, 220)).get) == "water")
    check("brown reads as dirt", max(rf.colour_prior((130, 90, 50)), key=rf.colour_prior((130, 90, 50)).get) == "dirt")
    check("grey reads as rock", max(rf.colour_prior((120, 120, 120)), key=rf.colour_prior((120, 120, 120)).get) == "rock")
    check("a dark bluish shadow isn't water", rf.colour_prior((30, 40, 60))["water"] < 0.2, rf.colour_prior((30, 40, 60)))
    check("muted teal-green paint is grass, not stone", max(rf.colour_prior((62, 70, 68)), key=rf.colour_prior((62, 70, 68)).get) == "grass",
          rf.colour_prior((62, 70, 68)))
    names = list(rf.SURFACES)
    say_dirt = np.zeros(len(names)); say_dirt[names.index("dirt")] = 1.0
    check("what it looks like can overrule colour", rf.name_groups([(95, 115, 80)], [say_dirt], names) == ["dirt"])
    check("without a second opinion colour decides", rf.name_groups([(70, 150, 60)], None, names) == ["grass"])
    centres, labels = rf.group_colours(np.array([[0, 200, 0]] * 500 + [[120, 80, 40]] * 500, np.float32), groups=2)
    check("two colours make two groups", labels[0] != labels[-1] and len(centres) == 2)
    filled = rf._fill_nearest(np.array([[-1, -1, 2], [-1, -1, -1], [1, -1, -1]]))
    check("ground under objects takes its surroundings", filled[0, 1] == 2 and filled[2, 0] == 1 and (filled >= 0).all(), filled)
    # picture -> surface, with the same camera as placement
    cam_obj = rwb_layout.PaintCamera(768, 548, 335, 60.0, 160.0, 8000.0)
    cam = {"W": 768.0, "H": 548.0, "horizon": 335.0, "focal": cam_obj.focal, "eye": 160.0, "player_y": cam_obj.player_y}
    u, v, _ = rf.picture_position(np.array([0.0]), np.array([cam_obj.player_y - 600.0]), cam)
    back = cam_obj.to_image(0.0, cam_obj.player_y - 600.0)
    check("the ground map uses the placement camera", near(float(u[0]), back[0], 1e-6) and near(float(v[0]), back[1], 1e-6))
    cells = np.zeros((10, 10), np.int64)
    cells[6:, :] = 1                                # lower part of the picture is the other surface
    nearest = rf.surface_at(cells, [0.1, 0.9], np.array([0.0]), np.array([cam_obj.player_y - 500.0]), cam)
    check("a near spot gets the surface painted low in the picture", nearest[0] == 1, nearest)
    behind = rf.surface_at(cells, [0.9, 0.1], np.array([0.0]), np.array([cam_obj.player_y + 300.0]), cam)
    check("beyond the view it's the ground's main surface", behind[0] == 0, behind)
    tmp = Path(tempfile.mkdtemp(prefix="rwb_surf_"))
    try:
        pygame.display.init()
        H, W = 200, 300
        rgb = np.full((H, W, 3), 0, np.uint8)
        rgb[:80] = (150, 180, 230)
        rgb[80:] = (70, 150, 60)
        rgb[140:, 150:] = (130, 90, 50)
        rng = np.random.default_rng(1)
        texture = np.clip(np.repeat(np.repeat(rgb, 2, 0), 2, 1).astype(np.int32) + rng.integers(-12, 12, (H * 2, W * 2, 3)), 0, 255).astype(np.uint8)
        rgb[185:, :60] = (20, 16, 18)                          # a dark framing trunk at the corner
        texture = np.clip(np.repeat(np.repeat(rgb, 2, 0), 2, 1).astype(np.int32) + rng.integers(-12, 12, (H * 2, W * 2, 3)), 0, 255).astype(np.uint8)
        result = rf.read_surfaces(rgb, texture, 80.0, tmp)
        kinds = {s["name"] for s in result["surfaces"]}
        check("the picture's two grounds are found", kinds == {"grass", "dirt"}, kinds)
        check("every cell has a surface, even under the dark trunk", min(min(r) for r in result["cells"]) >= 0)
        check("the panel can say so", rf.line(result).startswith("Ground: "), rf.line(result))
        grass = next(s for s in result["surfaces"] if s["name"] == "grass")
        tile = np.load(grass["tile"])
        check("a surface's tile is its own paint", tile[..., 1].mean() > tile[..., 0].mean() + 30, tile.mean((0, 1)))
        out = tmp / "ground.png"
        cam2 = dict(cam, horizon=80.0, H=200.0, W=300.0, focal=(150.0 / math.tan(math.radians(30))))
        rf.bake_ground({"camera": cam2, "world_size": 8000.0, "surfaces": result["surfaces"], "cells": result["cells"],
                        "heights": [[0.0, 0.0], [0.0, 0.0]], "water_z": None, "rock_color": [100, 100, 100],
                        "bake_size": 256, "out": str(out)})
        art = pygame.image.load(str(out))
        check("the terrain map is painted", art.get_size() == (256, 256) and art.get_at((128, 128))[:3] != (0, 0, 0))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_wading():
    import ast
    source = (HERE / "ReferenceWorldBuilder" / "Content" / "Python" / "world_builder.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "wading_speed")
    scope = {"WADING_SPEED": 0.45}
    exec(compile(ast.Module([node], []), "wading", "exec"), scope)
    speed = scope["wading_speed"]
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_water")
    text = ast.get_source_segment(source, build)
    check("water is passable in every build mode", "BlockAll" not in text and "NO_COLLISION" in text)
    check("wading follows wherever water is built", '_state["water_z"] = water_z' in text)
    check("no water, normal speed", speed(500.0, 10.0, None) == 500.0)
    check("above the water, normal speed", speed(500.0, 10.0, 0.0) == 500.0)
    check("ankle deep hardly slows you", speed(500.0, -10.0, 0.0) > 450.0)
    check("deep water slows you a lot", near(speed(500.0, -140.0, 0.0), 225.0, 1.0), speed(500.0, -140.0, 0.0))


def test_progress_and_speed():
    import rwb_progress
    tmp = Path(tempfile.mkdtemp(prefix="rwb_prog_"))
    saved = os.environ.get("RWB_PROGRESS")
    try:
        path = str(tmp / "p.txt")
        os.environ.pop("RWB_PROGRESS", None)
        rwb_progress.report(0.5, "ignored")
        check("no progress file without Unreal watching", not Path(path).exists())
        os.environ["RWB_PROGRESS"] = path
        check("nothing reported yet reads as nothing", rwb_progress.read(path) is None)
        rwb_progress.report(0.25, "Shaping 2 of 8 models")
        check("progress reads back", rwb_progress.read(path) == (0.25, "Shaping 2 of 8 models"), rwb_progress.read(path))
        rwb_progress.report(7.0, "clamped")
        check("progress can't pass 100%", rwb_progress.read(path)[0] == 1.0)
    finally:
        if saved is None:
            os.environ.pop("RWB_PROGRESS", None)
        else:
            os.environ["RWB_PROGRESS"] = saved
        shutil.rmtree(tmp, ignore_errors=True)
    entries = [{"name": str(i), "budget": b} for i, b in enumerate((9000, 600, 600, 600, 8000, 700, 650))]
    groups = rm._batches(entries, 3)
    loads = sorted(sum(e["budget"] for e in g) for g in groups)
    check("no model is lost between Blenders", sum(len(g) for g in groups) == len(entries))
    check("big models don't all land on one Blender", loads[-1] <= 9000, loads)
    check("fewer models than Blenders still works", len(rm._batches(entries[:2], 4)) == 2)
    check("workers stay within the machine", 1 <= rm.worker_count() <= 4)
    seen = []
    check("one by one keeps order", rm._in_parallel(abs, [-1, -2, -3], 1, seen.append) == [1, 2, 3] and seen == [1, 2, 3])
    seen = []
    check("in parallel keeps order", rm._in_parallel(abs, [-4, -5, -6, -7], 2, seen.append) == [4, 5, 6, 7])
    check("in parallel reports every one", sorted(seen) == [1, 2, 3, 4], seen)


def test_shape_choice():
    round_mask = blob()
    lopsided = np.zeros((60, 60), bool)
    lopsided[:, 28:32] = True                   # a post...
    lopsided[10:20, 30:58] = True               # ...with a sign out one side
    check("a symmetric shape scores high", rm.symmetry(round_mask) > 0.9, rm.symmetry(round_mask))
    check("a lopsided shape scores low", rm.symmetry(lopsided) < rm.ROUND_MIN_SYMMETRY, rm.symmetry(lopsided))
    pick = lambda mask, **obj: rm.choose_method(obj, mask, mask.shape[0] / float(mask.shape[1]))
    check("a round thing that is round spins", pick(round_mask, category="mushroom") == "lathe")
    check("no more vases: a lopsided 'round' thing doesn't spin", pick(lopsided, category="lantern") == "slab")
    check("buildings are built in parts", pick(round_mask, category="castle") == "composite")
    check("foliage is a soft mass", pick(round_mask, category="bush") == "cluster")
    check("an uncertain outline is a soft mass", pick(round_mask, category="rock", quality=0.5) == "cluster")
    check("a tangled outline is a soft mass", pick(round_mask, category="rock", roughness=40.0) == "cluster")
    check("a tree stays a tree even when unsure", pick(round_mask, category="tree", quality=0.4) == "inflate")
    check("a boat still lies down when unsure", pick(round_mask, category="boat", quality=0.4) == "ground")
    check("your choice beats the tool's", pick(round_mask, category="castle", method="slab") == "slab")
    # a tower with a branch: the branch mustn't widen the tower
    tower = np.zeros((80, 80), bool)
    tower[10:78, 30:50] = True
    tower[20:24, 5:30] = True
    core, limbs = rm.split_core(tower)
    check("the body is found", core[50, 40] and not core[22, 8], core.sum())
    check("the branch is a limb", len(limbs) == 1 and limbs[0][22, 8], len(limbs))
    verts, faces = rm.composite(tower)
    is_closed, _ = closed(faces)
    check("a building is closed", is_closed)
    check("the limb still reaches out", verts[:, 0].min() < 10.0, verts[:, 0].min())
    # the body itself stays narrow: check the back half's reach at mid-height
    mid = verts[(verts[:, 2] > 30) & (verts[:, 2] < 40)]
    check("the body stays the width it was painted", np.ptp(mid[:, 0]) < 24.0, np.ptp(mid[:, 0]))
    verts, faces = rm._ball(0.0, 0.0, 0.0, 2.0)
    is_closed, _ = closed(faces)
    check("a form is closed and faces out", is_closed and outward_share(verts, faces) > 0.95, outward_share(verts, faces))
    blobby = blob(60, 60)
    verts, faces = rm.cluster(blobby, seed=2)
    check("a mass holds several forms", len(faces) > 500, len(faces))
    check("a mass keeps its painted size", near(np.ptp(verts[:, 0]), 60.0, 12.0) and near(np.ptp(verts[:, 2]), 60.0, 12.0),
          (np.ptp(verts[:, 0]), np.ptp(verts[:, 2])))
    again, _ = rm.cluster(blobby, seed=2)
    check("the same object rebuilds the same", np.array_equal(verts, again))


def test_outlines():
    rod = np.zeros((100, 100), bool)
    rod[5:95, 50:52] = True
    check("smoothing leaves thin things alone", ro.smooth_outline(rod, 3).sum() == rod.sum())
    rough = np.zeros((100, 100), bool)
    rough[20:80, 20:80] = True
    for k in range(20, 80, 4):
        rough[k, 80:83] = True
    smooth = ro.smooth_outline(rough, 3)
    check("smoothing removes spurs", not smooth[40, 82] and smooth[50, 50])
    circle = np.fromfunction(lambda y, x: (y - 50) ** 2 + (x - 50) ** 2 < 900, (100, 100))
    check("a tangled outline measures rougher", ro.roughness(rough) > ro.roughness(circle))
    # a building gathers the fragments touching it
    def item(box, category, score=0.2, label=None):
        m = np.zeros((200, 200), bool)
        m[box[1]:box[3], box[0]:box[2]] = True
        return {"mask": m, "category": category, "score": score, "label": label or category, "place": ro.describe_mask(m)}
    tower = item((80, 20, 120, 150), "castle", 0.5)
    arch = item((110, 110, 132, 150), "bridge")
    root = item((60, 148, 90, 170), "prop_card")
    shroom = item((122, 130, 135, 150), "mushroom")
    sky = item((60, 0, 78, 14), "furniture")
    far = item((170, 100, 195, 150), "rock")
    kept = ro.merge_structures([tower, arch, root, shroom, sky, far])
    check("fragments join the building", arch.get("merged_away") and root.get("merged_away") and tower["mask"][160, 80])
    check("the building reports what it gathered", set(tower["parts"]) == {"bridge", "prop_card"}, tower.get("parts"))
    check("a mushroom beside it stays itself", not shroom.get("merged_away"))
    check("what's above it stays itself", not sky.get("merged_away"))
    check("what isn't touching stays itself", not far.get("merged_away"))
    check("the base comes down to the lowest part", near(tower["place"]["contact_v"], 170 / 200.0, 0.01), tower["place"])
    check("merged pieces leave the list", len(kept) == 4, len(kept))
    # the bug this guards: each merge widening the reach, so a chain swallowed the whole picture
    tower = item((80, 20, 120, 150), "castle", 0.5)
    links = [item((118 + 18 * n, 120, 138 + 18 * n, 150), "rock") for n in range(4)]
    big = item((30, 150, 110, 195), "boat")
    ro.merge_structures([tower] + links + [big])
    check("a chain of neighbours can't swallow the picture", not links[-1].get("merged_away"))
    check("something big next to a building stays itself", not big.get("merged_away"))
    # touching in the picture, but far behind in the world
    tower = item((80, 20, 120, 150), "castle", 0.5)
    tower["depth_eye"] = 8.0
    behind = item((118, 100, 128, 150), "furniture")
    behind["depth_eye"] = 110.0
    beside = item((70, 120, 82, 150), "prop_card")
    beside["depth_eye"] = 9.5
    unknown = item((100, 150, 110, 160), "rock")
    unknown["depth_eye"] = None
    ro.merge_structures([tower, behind, beside, unknown])
    check("a far-away thing behind a building stays itself", not behind.get("merged_away"))
    check("something at the building's depth joins", beside.get("merged_away"))
    check("beyond the depth map's range counts as far", not unknown.get("merged_away"))
    weak = item((80, 20, 120, 150), "castle", 0.1)
    near_it = item((118, 110, 140, 150), "bridge")
    ro.merge_structures([weak, near_it])
    check("a doubtful building gathers nothing", not near_it.get("merged_away"))


def test_water():
    import reference_inventory as ri
    tmp = Path(tempfile.mkdtemp(prefix="rwb_water_"))
    try:
        H, W = 200, 300
        rgb = np.full((H, W, 3), 70, np.uint8)
        rgb[:80] = (150, 180, 230)                              # sky: same hue as water, above the horizon
        pale, deep = (120, 190, 240), (40, 110, 200)            # water in two tones
        rgb[100:150, 40:260] = pale
        rgb[150:190, 40:260] = deep
        rgb[120:170, 130:170] = (70, 90, 60)                    # an island
        rgb[110:125, 5:20] = (60, 120, 220)                     # a blue flower, away from the water
        rgb[175:195, 262:290] = (20, 20, 25)                    # a dark trunk, inside a sloppy mask below
        mask = np.zeros((H, W), bool)
        mask[105:130, 60:120] = True                            # the detector only found a patch
        sloppy = np.zeros((H, W), bool)
        sloppy[160:195, 200:290] = True
        np.save(tmp / "a.npy", mask)
        np.save(tmp / "b.npy", sloppy)
        areas = [{"category": "water", "label": "pond", "score": 0.3, "mask": str(tmp / "a.npy")},
                 {"category": "water", "label": "pond", "score": 0.2, "mask": str(tmp / "b.npy")}]
        out = ri.refine_water(rgb, areas, 85.0)
        check("water grows from a patch to the whole lake", out[110, 250] and out[185, 60], out.mean())
        check("every tone of the water counts", out[160, 100] and out[120, 100])
        check("an island stays land", not out[145, 150])
        check("the sky isn't water", not out[:80].any())
        check("a blue flower isn't water", not out[117, 12])
        check("a dark trunk in the mask isn't water", not out[185, 275])
        grid = ri._grid_from_mask(out, 72)
        painted = np.random.default_rng(1).integers(90, 200, (H * 2, W * 2, 3)).astype(np.uint8)
        tile = ri.water_tile(painted, grid, tmp / "tile.png")
        check("a water swatch is made", tile and Path(tile).is_file())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_read_strokes():
    import reference_strokes as rs
    for wanted in (0.0, 45.0, 90.0, 135.0):
        stats = rs.read_strokes(stripes(wanted))
        check("reads strokes at %d degrees" % wanted, angle_gap(stats["angle"], wanted) < 6.0, stats)
        check("parallel marks read as directional (%d)" % wanted, stats["coherence"] > 0.6, stats)
    rng = np.random.default_rng(3)
    noise = rs.blur(rng.uniform(0, 255, (512, 512, 3)).astype(np.float32), 2) * 4.0
    check("noise doesn't read as directional", rs.read_strokes(noise)["coherence"] < 0.45, rs.read_strokes(noise))
    check("the panel line says which way", "vertical" in rs.describe({"angle": 90.0, "coherence": 0.7, "width": 4, "length": 30}))
    check("a patch holds a whole mark", rs.patch_size({"width": 5, "length": 60}) >= 120)
    check("a patch is never tiny", rs.patch_size({"width": 1, "length": 2}) == rs.MIN_PATCH)


def test_quilt():
    import reference_strokes as rs
    source = stripes(30.0, 300)
    tile = rs.quilt(source, 256, 64, seed=2)
    check("quilt has the size asked for", tile.shape == (256, 256, 3), tile.shape)
    check("quilt uses the source's colours", tile.min() >= source.min() - 1 and tile.max() <= source.max() + 1)
    check("quilt keeps the direction", angle_gap(rs.read_strokes(tile)["angle"], 30.0) < 10.0, rs.read_strokes(tile))
    wrap = np.abs(tile[0] - tile[-1]).mean()
    inside = np.abs(tile[100] - tile[101]).mean()
    check("the tile repeats without a line", wrap < inside * 3.0, (wrap, inside))
    again = rs.quilt(source, 256, 64, seed=2)
    check("same seed, same paint", np.array_equal(tile, again))
    check("too small to quilt says so", rs.quilt(source[:6, :6], 64, 64) is None)
    valid = np.zeros(source.shape[:2], bool)
    check("nothing usable says so", rs.quilt(source, 64, 32, valid) is None)
    # usable pixels only: patches can't come from the part of the picture ruled out
    half = source.copy()
    half[:, 150:] = 255.0
    valid = np.ones(source.shape[:2], bool)
    valid[:, 140:] = False
    check("ruled-out areas stay out", rs.quilt(half, 128, 32, valid, seed=1).max() < 250.0)
    detail, low, _usable = rs.extract_detail(source)
    check("marks and colour add back up (within clipping)", np.abs(detail + low - source).mean() < 15.0)
    check("marks average to nothing", abs(float(detail.mean())) < 2.0, detail.mean())


def _plate():
    """A one-quad-per-face double-sided plate in atlas halves: front (-Y) left, back (+Y) right."""
    positions = np.array([[0, -1, 0], [10, -1, 0], [10, -1, 10], [0, -1, 10],
                          [0, 1, 0], [10, 1, 0], [10, 1, 10], [0, 1, 10]], np.float32)
    normals = np.array([[0, -1, 0]] * 4 + [[0, 1, 0]] * 4, np.float32)
    tris = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], np.int32)
    left = [[0.02, 0.02], [0.48, 0.02], [0.48, 0.98], [0.02, 0.98]]
    right = [[0.52, 0.02], [0.98, 0.02], [0.98, 0.98], [0.52, 0.98]]
    uv = np.array(left + right, np.float32)
    return {"positions": positions, "normals": normals, "tris": tris, "tri_uvs": uv[tris]}


def test_bake():
    import reference_strokes as rs
    surface = _plate()
    # art: left half red, right half blue, with a white mark that must only show on the front
    art = np.zeros((64, 64, 3), np.float32)
    art[:, :32] = (200, 40, 40)
    art[:, 32:] = (40, 40, 200)
    art[31:33, 8:24] = 255.0                   # thin, like a real brush mark next to its object

    def uv_of(p):
        return np.stack([p[:, 0] / 10.0, 1.0 - p[:, 2] / 10.0], -1)

    size = 128
    rgb, direct = rs.bake(surface, art, None, uv_of, "slab", 6.4, 1.0, size)
    check("half the plate faces the painter", near(direct, 0.5, 0.05), direct)
    front, back = rgb[:, 8:56].astype(np.float32), rgb[:, 72:120].astype(np.float32)
    red = (200, 40, 40)
    check("the front is the painting, mark included", front.max() >= 250, front.max())
    check("the mark isn't mirrored onto the back", back.reshape(-1, 3).sum(1).max() < 620, back.max())
    # the back keeps the object's broad colour on the matching side
    left_side = back[:, :10].mean((0, 1))
    right_side = back[:, -10:].mean((0, 1))
    check("the back keeps the broad colours", left_side[0] > left_side[2] and right_side[2] > right_side[0],
          (left_side, right_side))
    flat, _ = rs.bake(surface, art, None, uv_of, "slab", 6.4, 0.0, size)
    detail = np.zeros((64, 64, 3), np.float32)
    detail[::8] = 40.0
    marked, _ = rs.bake(surface, art, detail, uv_of, "slab", 6.4, 1.0, size)
    check("strokes land on the back", np.abs(marked[:, 72:120].astype(int) - flat[:, 72:120]).max() > 20)
    check("strokes leave the painted front alone", np.abs(marked[:, 8:56].astype(int) - flat[:, 8:56]).max() < 4)
    strong, _ = rs.bake(surface, art, detail, uv_of, "slab", 6.4, 2.0, size)
    check("the slider makes strokes stronger",
          np.abs(strong[:, 72:120].astype(int) - flat[:, 72:120]).mean()
          > np.abs(marked[:, 72:120].astype(int) - flat[:, 72:120]).mean())
    # no black atlas gaps
    check("the empty atlas never shows black", rgb[:, 62:66].sum(-1).min() > 30, rgb[:, 62:66].sum(-1).min())
    normal = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    check("lying shapes face up", rs.facing_painter(normal, "ground")[0] == 1.0 and rs.facing_painter(normal, "ground")[1] == 0.0)
    check("standing shapes face front", rs.facing_painter(normal, "slab")[1] == 1.0 and rs.facing_painter(normal, "slab")[2] == 0.0)
    check("a round building's windows carry round the back", rs.facing_painter(normal, "composite", True)[2] == 1.0)
    # triplanar: same marks at the same scale on every axis
    tile = stripes(0.0, 64, 16)
    pos = np.array([[3.0, 0.0, 5.0]])
    value = rs.triplanar(tile, pos, np.array([[0.0, 1.0, 0.0]]), 2.0)
    check("marks wrap at the painting's scale", np.allclose(value[0], rs.sample(tile, np.array([6.0]), np.array([-10.0]), wrap=True)[0]))


# --------------------------------------------------------------------------- #
# Shape Sketch
# --------------------------------------------------------------------------- #
def test_sketch():
    import reference_draw as rd
    pygame.init()
    pygame.display.set_mode((16, 16))
    sketch = rd.Sketch((200, 100))
    check("starts empty", sketch.empty())
    sketch.checkpoint()
    sketch.fill_outline([(20, 20), (80, 20), (80, 80), (20, 80)])
    mask = sketch.mask()
    check("an outline fills", mask[50, 50] and not mask[50, 150])
    check("mask is rows by columns", mask.shape == (100, 200), mask.shape)
    sketch.checkpoint()
    sketch.stroke((50, 50), (50, 50), 10, erase=True)
    check("the eraser removes", not sketch.mask()[50, 50])
    sketch.undo()
    check("undo restores", sketch.mask()[50, 50])
    sketch.undo()
    check("undo all the way", sketch.empty())
    sketch.stroke((120, 10), (190, 90), 3)
    check("a fast brush stroke has no gaps", sketch.mask()[50, 155])
    tmp = Path(tempfile.mkdtemp(prefix="rwb_draw_"))
    try:
        H, W = 200, 300
        texture = np.zeros((400, 600, 3), np.uint8)
        texture[..., 0] = 200
        mask = np.zeros((100, 150), bool)
        mask[25:75, 75:120] = True
        job = rd.compose(mask, texture, (H, W), None, tmp, "lathe")
        check("the job is in the outbox", (tmp / "outbox" / (job["name"] + ".json")).is_file())
        check("no half-written job can be read", not list((tmp / "outbox").glob("*.tmp")))
        check("where you draw is where it goes", near(job["centre_u"], 97.5 / 150, 0.02) and near(job["contact_v"], 0.75, 0.02), job)
        check("the chosen shape travels", job["method"] == "lathe" and job["category"] == "drawn")
        art = pygame.image.load(job["cutout"])
        check("paint under the shape", art.get_at((art.get_width() // 2, art.get_height() // 2))[0] == 200)
        check("the cut-out is at full resolution", near(art.get_width(), 180, 6), art.get_size())
        tile = np.zeros((64, 64, 3), np.float32)
        tile[..., 2] = 220.0
        second = rd.compose(mask, texture, (H, W), tile, tmp, "inflate")
        art = pygame.image.load(second["cutout"])
        check("picked paint replaces what's under the shape", art.get_at((art.get_width() // 2, art.get_height() // 2))[2] == 220)
        check("shapes don't overwrite each other", second["name"] != job["name"])
        check("too small to build says so", rd.compose(np.zeros((100, 150), bool), texture, (H, W), None, tmp, "slab") is None)
        fixed = rd.compose(mask, texture, (H, W), None, tmp, "slab", replaces="cut_05")
        check("a redraw says what it replaces", fixed.get("replaces") == "cut_05")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# live: real models, real Blender
# --------------------------------------------------------------------------- #
def test_live(image):
    import reference_inventory
    import reference_scene
    pygame.display.init()
    tmp = Path(tempfile.mkdtemp(prefix="rwb_live_"))
    try:
        scene = reference_scene.prepare(str(image), tmp / "scene")
        check("depth model ran", bool(scene.get("depth")), scene.get("depth_error"))
        surf = reference_scene._surface(image)
        small = reference_scene._to_rgb(reference_scene._scaled(surf, 768))
        result = {"scene": scene, "width": surf.get_width(), "height": surf.get_height()}
        inv = reference_inventory.build(result, str(image), tmp / "inv", str(HERE / "rwb_layer_keywords.json"))
        check("recognition ran", inv["source"] == "detector", inv.get("info"))
        instances = [i for o in inv["objects"] for i in o["instances"]]
        print("  found %d objects in %d categories: %s" % (
            len(instances), len(inv["objects"]), ", ".join(sorted({o["category"] for o in inv["objects"]}))))
        check("found objects", len(instances) >= 3, len(instances))
        check("every object has an outline", all(Path(i["cutout"]).is_file() for i in instances))
        cam = rwb_layout.PaintCamera(inv["picture"]["W"], inv["picture"]["H"], inv["picture"]["horizon"], 60.0, 160.0, 8000.0)
        errors = []
        for obj in inv["objects"]:
            for inst in obj["instances"]:
                placed = cam.place(inst, obj["category"])
                column = cam.W / 2.0 + placed["x"] * cam.focal / placed["distance"]
                errors.append(abs(column - inst["centre_u"] * cam.W) / cam.W)
        check("objects stay at their painted left/right", max(errors) < 0.05, max(errors))
        jobs = [{"id": "m%02d" % n, "cutout": i["cutout"], "category": o["category"], "label": i.get("label"),
                 "bbox": i["bbox_px"], "area": i.get("area", 0.05)}
                for n, (o, i) in enumerate((o, i) for o in inv["objects"] for i in o["instances"])][:6]
        camera = {"W": cam.W, "H": cam.H, "horizon": cam.horizon, "focal": cam.focal}
        manifest = rm.build_all(jobs, tmp / "models", camera, strokes=inv.get("strokes"), workers=rm.worker_count())
        print("  " + manifest["message"])
        check("models built", len(manifest["models"]) == len(jobs), manifest.get("skipped"))
        if rm.find_blender():
            check("Blender refined them", all(m.get("refined") for m in manifest["models"]), manifest["message"])
            check("FBX written", all(m["mesh"].endswith(".fbx") for m in manifest["models"] if m.get("refined")))
            check("every model got a wrapped texture", all(m["art"].endswith("_wrap.png") for m in manifest["models"]),
                  manifest.get("bake_message"))
        else:
            print("  (Blender not installed: refinement not tested)")
        del small
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def default_image():
    """A small synthetic scene, so --live works without anyone's artwork."""
    tmp = Path(tempfile.mkdtemp(prefix="rwb_img_"))
    W, H = 900, 640
    surf = pygame.Surface((W, H))
    surf.fill((150, 190, 235))
    pygame.draw.rect(surf, (70, 130, 60), (0, 300, W, H - 300))
    for x, y, s in ((200, 420, 1.0), (620, 380, 0.7)):
        pygame.draw.rect(surf, (100, 70, 40), (x - 18 * s, y - 120 * s, 36 * s, 120 * s))
        pygame.draw.circle(surf, (30, 110, 40), (x, int(y - 170 * s)), int(85 * s))
    pygame.draw.rect(surf, (170, 60, 50), (380, 380, 170, 120))
    pygame.draw.polygon(surf, (90, 50, 40), ((360, 380), (465, 290), (570, 380)))
    path = tmp / "scene.png"
    pygame.image.save(surf, str(path))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="run the real vision models and Blender")
    parser.add_argument("--image", default=None, help="picture for --live")
    args = parser.parse_args()
    for test in (test_camera, test_placement, test_ground_placement, test_filtering, test_masks, test_labels,
                 test_psd_names, test_mesh, test_ground_shape, test_end_to_end_no_blender,
                 test_clicks_are_queued, test_catalog, test_trees_and_style, test_surfaces, test_wading, test_progress_and_speed, test_shape_choice, test_outlines, test_water,
                 test_read_strokes, test_quilt, test_bake, test_sketch):
        test()
    if args.live:
        test_live(Path(args.image) if args.image else default_image())
    print("%d checks, %d failed" % (count, len(failures)))
    if failures:
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
