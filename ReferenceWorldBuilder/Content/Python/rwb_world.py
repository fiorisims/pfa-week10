"""Reference World Builder - the default world: a fresh terrain, with the painting's objects in it.

Every analyzed image / PSD becomes:

1. a clean, seeded terrain (rolling ground, a pond basin where water was painted, hills, a raised
   rim, a flat clearing at the start), vertex-coloured from the painting's own ground colours;
2. every object found in the image, at the spot and size it was painted: the camera is solved
   (``rwb_layout``) so that from the start position the world lines up with the picture;
3. each object is a NEW 3D model built from its outline and painted with its own piece of the
   image (``reference_mesh``, refined in Blender).  Without an outline it falls back to a matching
   library model, shapes or a painted cut-out;
4. extra copies of repeated things may fill the map OUTSIDE the painted view, so the view of the
   picture stays exactly as it was painted.
"""

from __future__ import annotations

import json
import math
import re
import random
from pathlib import Path

import unreal

import rwb_generated
import rwb_layers
import rwb_layout
import rwb_library

# typical size in cm: ("h", height) or ("w", width for flat things)
PRIORS = {
    "tree": ("h", 1000), "pine": ("h", 1200), "bush": ("h", 150), "thorns": ("h", 260), "flower": ("h", 45),
    "grass": ("h", 70), "rock": ("h", 160), "cliff": ("h", 700), "log": ("h", 70), "mushroom": ("h", 70),
    "boat": ("w", 420), "lily": ("w", 110), "castle": ("h", 3000), "tower": ("h", 1300), "house": ("h", 700),
    "wall": ("h", 300), "fence": ("h", 110), "bridge": ("w", 700), "gate": ("h", 420), "pillar": ("h", 380),
    "lantern": ("h", 260), "furniture": ("h", 110), "character": ("h", 175), "creature": ("h", 110),
    "prop_card": ("h", 160), "unknown": ("h", 200),
}
BASE_COUNTS = {
    "tree": 16, "pine": 16, "bush": 12, "thorns": 10, "flower": 24, "grass": 30, "rock": 9, "cliff": 4,
    "log": 5, "mushroom": 10, "lily": 12, "fence": 4, "lantern": 4, "pillar": 3, "furniture": 3, "house": 4,
    "tower": 2, "wall": 3,
}
WATER_OBJECTS = {"boat", "lily"}
# when the library has no model for a type, borrow from a close relative
FALLBACK = {
    "thorns": ["bush"], "pine": ["tree"], "tree": ["pine"], "cliff": ["rock"], "rock": ["cliff"],
    "grass": ["bush"], "bush": ["thorns"], "log": [], "flower": [], "mushroom": [],
}
STRUCTURES = set(rwb_layers.STRUCTURES)
FOREST = {"tree", "pine"}
GRASS_LIFT = 2.0
GRASS_CAP = 240               # grass models on a grassy map (they're individual actors)
TOTAL_CAP = 650
MAX_GENERATED = 60            # models built per picture (largest first)
MAX_COUNT = 200               # copies of one model the panel may ask for
_world = {}                   # the last world built this session: lets the panel change it in place


def _wb():
    import world_builder
    return world_builder


# --------------------------------------------------------------------------- #
# terrain
# --------------------------------------------------------------------------- #
def _smooth01(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def _value_noise(n, cells, rng):
    g = [[rng.random() * 2 - 1 for _ in range(cells + 1)] for _ in range(cells + 1)]
    out = [[0.0] * n for _ in range(n)]
    for iy in range(n):
        fy = iy / float(n - 1) * cells
        y0 = min(cells - 1, int(fy))
        ty = _smooth01(fy - y0)
        for ix in range(n):
            fx = ix / float(n - 1) * cells
            x0 = min(cells - 1, int(fx))
            tx = _smooth01(fx - x0)
            a = g[y0][x0] * (1 - tx) + g[y0][x0 + 1] * tx
            b = g[y0 + 1][x0] * (1 - tx) + g[y0 + 1][x0 + 1] * tx
            out[iy][ix] = a * (1 - ty) + b * ty
    return out


def _blur(grid, passes=1):
    n = len(grid)
    for _ in range(passes):
        copy = [row[:] for row in grid]
        for y in range(n):
            for x in range(n):
                total, count = 0.0, 0
                for yy in (y - 1, y, y + 1):
                    for xx in (x - 1, x, x + 1):
                        if 0 <= yy < n and 0 <= xx < n:
                            total += copy[yy][xx]
                            count += 1
                grid[y][x] = total / count
    return grid


def _sample_grid(grid, u, v):
    if not grid:
        return 0.0
    size = len(grid)
    fx = max(0.0, min(size - 1.0, u * size - 0.5))
    fy = max(0.0, min(size - 1.0, v * size - 0.5))
    x0, y0 = int(fx), int(fy)
    x1, y1 = min(size - 1, x0 + 1), min(size - 1, y0 + 1)
    tx, ty = fx - x0, fy - y0
    a = grid[y0][x0] * (1 - tx) + grid[y0][x1] * tx
    b = grid[y1][x0] * (1 - tx) + grid[y1][x1] * tx
    return a * (1 - ty) + b * ty


def make_terrain(inv, frame, relief, rng, n=96, land=()):
    """Heights (row 0 = far edge), water level (or None) and a water mask.

    ``land`` = [(x, y, r)]: spots where something was painted standing, kept dry even if the water
    mask reaches them (a tower on an island).
    """
    size = frame.size
    amp = max(40.0, relief * 0.3)
    big, mid, small = _value_noise(n, 4, rng), _value_noise(n, 9, rng), _value_noise(n, 18, rng)
    water_cov = (inv.get("water") or {}).get("coverage")
    hill_cov = (inv.get("hills") or {}).get("coverage")
    has_water = bool((inv.get("water") or {}).get("present"))
    horizon_v = frame.horizon / frame.H
    heights = [[0.0] * n for _ in range(n)]
    water = [[0.0] * n for _ in range(n)]
    hills = [[0.0] * n for _ in range(n)]
    hill_columns = None
    if hill_cov:
        # mountains stand ON the horizon; their outline in the picture says which way they lie
        band = [row for i, row in enumerate(hill_cov) if abs((i + 0.5) / len(hill_cov) - horizon_v) < 0.2]
        if band:
            hill_columns = [max(row[c] for row in band) for c in range(len(band[0]))]
    for iy in range(n):
        y = -size / 2.0 + iy * size / (n - 1)
        for ix in range(n):
            x = -size / 2.0 + ix * size / (n - 1)
            # gentler ground inside the painted view, so objects stay where they were painted
            calm = 0.35 if frame.in_view(x, y, 300.0) else 1.0
            heights[iy][ix] = calm * amp * (0.65 * big[iy][ix] + 0.25 * mid[iy][ix] + 0.10 * small[iy][ix])
            img = frame.to_image(x, y)
            if img:
                u, v = img
                if water_cov and v > horizon_v + 0.035:
                    water[iy][ix] = _sample_grid(water_cov, u, v)
            if hill_columns and frame.in_view(x, y, size):
                distance = frame.player_y - y
                reach = _smooth01((distance - frame.far * 0.55) / (frame.far * 0.4))
                if reach > 0.0 and distance > 0.0:
                    u_far = 0.5 + x * frame.focal / (distance * frame.W)
                    if 0.0 <= u_far <= 1.0:
                        column = hill_columns[min(len(hill_columns) - 1, int(u_far * len(hill_columns)))]
                        hills[iy][ix] = column * reach
            edge = min(x + size / 2, size / 2 - x, y + size / 2, size / 2 - y) / (size * 0.12)
            heights[iy][ix] += amp * 3.0 * (1.0 - _smooth01(edge)) ** 2
    if hill_cov:
        _blur(hills, 3)
        for iy in range(n):
            for ix in range(n):
                heights[iy][ix] += amp * 7.0 * _smooth01(hills[iy][ix] * 1.4) * (0.75 + 0.25 * big[iy][ix])
    water_z = None
    if has_water:
        if sum(map(sum, water)) < n * n * 0.01:          # water seen but off-frame: a pond ahead
            cx, cy = frame.to_world(0.5, horizon_v + 0.35 * (1.0 - horizon_v))
            radius = size * 0.12
            for iy in range(n):
                y = -size / 2.0 + iy * size / (n - 1)
                for ix in range(n):
                    x = -size / 2.0 + ix * size / (n - 1)
                    d = math.hypot(x - cx, y - cy) / radius + 0.25 * mid[iy][ix]
                    water[iy][ix] = max(water[iy][ix], 1.0 - _smooth01(d - 0.6))
        for spot_x, spot_y, spot_r in land:
            for iy in range(n):
                y = -size / 2.0 + iy * size / (n - 1)
                if abs(y - spot_y) > spot_r * 1.6:
                    continue
                for ix in range(n):
                    x = -size / 2.0 + ix * size / (n - 1)
                    d = math.hypot(x - spot_x, y - spot_y)
                    if d < spot_r * 1.6:
                        water[iy][ix] *= _smooth01((d - spot_r) / (spot_r * 0.6))
        _blur(water, 2)
        flat = sorted(v for row in heights for v in row)
        water_z = flat[len(flat) // 2] - amp * 0.25
        for iy in range(n):
            for ix in range(n):
                w = _smooth01((water[iy][ix] - 0.15) / 0.35)
                h = heights[iy][ix]
                if w <= 0.0:
                    h = max(h, water_z + 30.0)
                else:
                    h = max(h, water_z + 30.0) * (1 - w) + (water_z - 140.0) * w
                heights[iy][ix] = h
                water[iy][ix] = w
    # flat clearing where the player starts
    py = frame.player_y
    ref = None
    for iy in range(n):
        y = -size / 2.0 + iy * size / (n - 1)
        for ix in range(n):
            x = -size / 2.0 + ix * size / (n - 1)
            d = math.hypot(x, y - py)
            if d < 1200.0:
                if ref is None:
                    ref = max(heights[iy][ix], (water_z + 40.0) if water_z is not None else -1e9)
                k = 1.0 - _smooth01((d - 500.0) / 700.0)
                heights[iy][ix] = heights[iy][ix] * (1 - k) + ref * k
                water[iy][ix] *= (1 - k)
    _blur(heights, 1)
    return heights, water_z, water


def _mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def build_terrain_actor(heights, water_z, water, inv, size, rng):
    wb = _wb()
    n = len(heights)
    pal = [wb._hex_rgb(c) for c in (inv.get("ground") or {}).get("colors") or ["#5F7A4A"]]
    while len(pal) < 3:
        pal.append(pal[-1])
    rock = wb._hex_rgb((inv.get("ground") or {}).get("rock") or "#6E6A64")
    shore = _mix(pal[1], (140, 123, 94), 0.35)
    tint1, tint2 = _value_noise(n, 6, rng), _value_noise(n, 13, rng)
    step = size / float(n - 1)
    verts, normals, uvs, colors, tangents = [], [], [], [], []
    for iy in range(n):
        for ix in range(n):
            x = -size / 2.0 + ix * step
            y = -size / 2.0 + iy * step
            z = heights[iy][ix]
            verts.append(unreal.Vector(x, y, z))
            left, right = heights[iy][max(0, ix - 1)], heights[iy][min(n - 1, ix + 1)]
            down, up = heights[max(0, iy - 1)][ix], heights[min(n - 1, iy + 1)][ix]
            normal = unreal.Vector(-(right - left) / (2 * step), -(up - down) / (2 * step), 1.0)
            normal.normalize()
            normals.append(normal)
            uvs.append(unreal.Vector2D(ix / float(n - 1), iy / float(n - 1)))
            tangents.append(unreal.ProcMeshTangent())
            c = _mix(pal[0], pal[1], 0.5 + 0.5 * tint1[iy][ix])
            c = _mix(c, pal[2], max(0.0, tint2[iy][ix]) * 0.6)
            c = _mix(c, rock, (0.85 - normal.z) * 4.0)
            if water_z is not None:
                c = _mix(c, shore, 1.0 - _smooth01((z - water_z) / 60.0))
                if z < water_z:
                    c = _mix(c, (c[0] * 0.55, c[1] * 0.6, c[2] * 0.65), 1.0)
            colors.append(unreal.LinearColor(*[rwb_scene_srgb(v) for v in c], 1.0))
    triangles = []
    for iy in range(n - 1):
        for ix in range(n - 1):
            a = iy * n + ix
            # counter-clockwise seen from above: procedural meshes cull the other way round
            triangles.extend([a, a + n + 1, a + 1, a, a + n, a + n + 1])
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.Actor, unreal.Vector(0, 0, 0))
    component = unreal.new_object(unreal.ProceduralMeshComponent, outer=actor, name="RWBTerrain")
    actor.set_editor_property("root_component", component)
    component.set_editor_property("mobility", unreal.ComponentMobility.STATIC)
    component.create_mesh_section_linear_color(0, verts, triangles, normals, uvs, [], [], [], colors, tangents, True)
    component.set_material(0, wb._terrain_material())
    component.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS)
    wb._tag_actor(actor, wb.GENERATED_TAG, wb.TERRAIN_TAG, label="RWB_Terrain")
    return actor


def rwb_scene_srgb(channel):
    v = max(0.0, min(255.0, float(channel))) / 255.0
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


# --------------------------------------------------------------------------- #
# placement without overlaps
# --------------------------------------------------------------------------- #
class Placer:
    def remove(self, x, y, r):
        """Free a spot (a model's count went down)."""
        cell = self.grid.get((int(x // self.cell), int(y // self.cell)), [])
        if (x, y, r) in cell:
            cell.remove((x, y, r))
        if (x, y, r) in self.items:
            self.items.remove((x, y, r))

    def __init__(self, size, heights, water_z, water, player):
        self.size = size
        self.heights = heights
        self.water_z = water_z
        self.water = water
        self.player = player
        self.items = []                 # (x, y, r)
        self.corridors = []             # (x0, y0, x1, y1, half_width)
        self.cell = 500.0
        self.grid = {}
        self.max_r = 0.0

    def uv(self, x, y):
        return x / self.size + 0.5, y / self.size + 0.5

    def height(self, x, y):
        u, v = self.uv(x, y)
        return _wb()._sample_height(self.heights, max(0.0, min(1.0, u)), max(0.0, min(1.0, v)))

    def wet(self, x, y):
        if self.water_z is None:
            return 0.0
        u, v = self.uv(x, y)
        return _sample_grid(self.water, u, v) if self.water else 0.0

    def ground(self, x, y, r):
        pts = [(x, y), (x + r * 0.6, y), (x - r * 0.6, y), (x, y + r * 0.6), (x, y - r * 0.6)]
        hs = [self.height(px, py) for px, py in pts]
        return min(hs), max(hs) - min(hs)

    def _near(self, x, y, r=0.0):
        cx, cy = int(x // self.cell), int(y // self.cell)
        reach = int(math.ceil((r + self.max_r + 60.0) / self.cell)) + 1      # covers the biggest footprint
        for gx in range(cx - reach, cx + reach + 1):
            for gy in range(cy - reach, cy + reach + 1):
                for item in self.grid.get((gx, gy), ()):
                    yield item

    def free(self, x, y, r, tall=False, in_water=False, gap=40.0):
        lim = self.size / 2.0 - 250.0 - r
        if abs(x) > lim or abs(y) > lim:
            return False
        if math.hypot(x - self.player[0], y - self.player[1]) < 650.0 + r:
            return False
        for px, py, pr in self._near(x, y, r):
            if (x - px) ** 2 + (y - py) ** 2 < (r + pr + gap) ** 2:
                return False
        wet = self.wet(x, y)
        if in_water:
            if self.water_z is None or wet < 0.7 or self.height(x, y) > self.water_z - 40.0:
                return False
        elif self.water_z is not None and (wet > 0.05 or self.height(x, y) < self.water_z + 15.0):
            return False
        if not in_water:
            _low, spread = self.ground(x, y, r)
            if spread > max(60.0, r * 0.7):          # too steep under the footprint
                return False
        if tall:
            for x0, y0, x1, y1, hw in self.corridors:
                if _seg_dist(x, y, x0, y0, x1, y1) < hw + r:
                    return False
        return True

    def painted_spot(self, x, y, r):
        """Stay on the painted spot; only step aside to separate real collisions.

        Painted objects may stand close, so only about half their width counts here.  If no clear
        spot is near, the object stays where it was painted: position beats a slight overlap.
        """
        core = r * 0.5
        lim = self.size / 2.0 - 250.0
        x, y = max(-lim, min(lim, x)), max(-lim, min(lim, y))

        def clear(px, py):
            if abs(px) > lim or abs(py) > lim:
                return False
            return all((px - ix) ** 2 + (py - iy) ** 2 >= (core + ir * 0.5) ** 2 for ix, iy, ir in self._near(px, py, core))

        if clear(x, y):
            return x, y
        step = max(30.0, core * 0.35)
        for ring in range(1, 5):
            for k in range(8):
                a = math.pi * k / 4.0
                px, py = x + math.cos(a) * step * ring, y + math.sin(a) * step * ring
                if clear(px, py):
                    return px, py
        return x, y

    def add(self, x, y, r):
        self.max_r = max(self.max_r, r)
        self.items.append((x, y, r))
        self.grid.setdefault((int(x // self.cell), int(y // self.cell)), []).append((x, y, r))

    def reserve_view(self, x, y, r):
        self.corridors.append((self.player[0], self.player[1], x, y, max(200.0, r * 0.6)))

    def spiral(self, x, y, r, **kw):
        if self.free(x, y, r, **kw):
            return x, y
        for ring in range(1, 14):
            dist = ring * max(120.0, r * 0.8)
            for k in range(8 + ring * 2):
                a = 2 * math.pi * k / (8 + ring * 2)
                px, py = x + math.cos(a) * dist, y + math.sin(a) * dist
                if self.free(px, py, r, **kw):
                    return px, py
        return None


def _seg_dist(px, py, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    length = dx * dx + dy * dy
    t = 0.0 if length == 0 else max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


# --------------------------------------------------------------------------- #
# objects
# --------------------------------------------------------------------------- #
def _pick_models(models, word):
    exact = [m for m in models if m["word"] == word]
    return exact or models


def _scale_for(model, category, size_mult, jitter):
    axis, value = PRIORS.get(category, ("h", 200))
    target = value * size_mult * jitter
    native = model["width"] if axis == "w" else model["height"]
    return max(0.03, min(300.0, target / native)), target


def _spawn_model(ctx, model, x, y, scale, colors, label, yaw, in_water=False):
    wb = _wb()
    r = model["width"] * scale * 0.5
    if in_water:
        z = ctx["placer"].water_z + 2.0
    else:
        low, _spread = ctx["placer"].ground(x, y, r)
        z = low - model["height"] * scale * 0.03
    z -= model["origin_z"] * scale
    big = model["height"] * scale > 120.0
    actor, component = rwb_layers._spawn(ctx, model["mesh"], unreal.Vector(x, y, z), (scale, scale, scale), label,
                                         yaw=yaw, collision=big and not in_water)
    rwb_layers._register(label, (scale, scale, scale))
    rwb_layers._dress(ctx, [(actor, component, model["mesh"], None)], None, colors or {}, "unused")
    return actor


def _spawn_kit(ctx, category, x, y, w, h, colors, label):
    parts = rwb_layers._structure_parts(category, w, h)
    low, _spread = ctx["placer"].ground(x, y, max(w, h * 0.3) * 0.5)
    base = low - h * 0.02
    spawned = []
    for n, (shape, centre, size, rot, role) in enumerate(parts):
        mesh = ctx["shapes"].get(shape)
        if not mesh:
            continue
        loc = unreal.Vector(x + centre[0], y + centre[1], base + centre[2])
        if not rot:
            origin = mesh.get_bounds().origin
            loc = unreal.Vector(loc.x - origin.x * size[0] / 100.0, loc.y - origin.y * size[1] / 100.0,
                                loc.z - origin.z * size[2] / 100.0)
        actor, component = rwb_layers._spawn(ctx, mesh, loc, (size[0] / 100.0, size[1] / 100.0, size[2] / 100.0),
                                             "%s_%d" % (label, n), roll=rot.get("roll", 0.0), pitch=rot.get("pitch", 0.0))
        rwb_layers._register("%s_%d" % (label, n), None)
        spawned.append((actor, component, mesh, role))
    if spawned:
        rwb_layers._dress(ctx, spawned, None, colors or {}, "unused")


def _spawn_card(ctx, obj, inst, x, y, label, cache_dir):
    import rwb_scene
    texture_path = inst.get("texture")
    if not texture_path:
        return False
    tex = rwb_scene._import_texture(texture_path, ctx["folder"] + "/Art",
                                    "T_%s_%s" % (ctx["psd"], rwb_layers._safe(obj["label"], 20)), cache_dir)
    if not tex:
        return False
    _axis, h = PRIORS.get(obj["category"], ("h", 175))
    h *= ctx["size"]
    w = h / max(0.25, min(4.0, inst.get("aspect", 2.0)))
    low, _spread = ctx["placer"].ground(x, y, w * 0.5)
    cube = ctx["shapes"]["cube"]
    actor, component = rwb_layers._spawn(ctx, cube, unreal.Vector(x, y, low + h * 0.5 - h * 0.02),
                                         (w / 100.0, 0.04, h / 100.0), label, collision=False)
    rwb_layers._register(label, (w / 100.0, 0.04, h / 100.0))
    rwb_layers._dress(ctx, [(actor, component, cube, "dominant")], tex, inst.get("colors") or {},
                      "%s_%s" % (ctx["psd"], rwb_layers._safe(obj["label"], 20)), card=True)
    return True


def _target_count(obj, density, area):
    base = BASE_COUNTS.get(obj["category"], 4)
    return int(max(obj["count"], min(220, round(base * (0.35 + 1.6 * density) * area))))


def _picture_camera(inv, analysis, values, size):
    """The painting's camera.  Object size scales the eye height, and so everything with it."""
    picture = inv.get("picture") or {}
    W = float(picture.get("W") or analysis.get("width") or 768)
    H = float(picture.get("H") or analysis.get("height") or 512)
    horizon = float(picture.get("horizon") or float(inv.get("horizon", 0.4)) * H)
    eye = 160.0 * float(values.get("object_scale", 1.0))
    return rwb_layout.PaintCamera(W, H, horizon, float(values.get("fov", 60.0)), eye, size)


def _place_for(camera, inst, category, method=None):
    # a found model of a lying thing (a canoe) sits where the painted one lay, sized by its width
    lying = method == "ground" or (method in (None, "adopt") and category in rwb_layout.FLAT_CATEGORIES)
    if lying and camera.lies_flat(inst):
        return camera.place_ground(inst)
    return camera.place(inst, category)


def _plan(inv, camera):
    """One record per painted object, in the order they should be claimed (largest first)."""
    plan = []
    for obj in inv.get("objects") or []:
        if obj.get("mass"):
            continue
        category = obj["category"]
        for inst in obj.get("instances") or []:
            if "contact_v" not in inst:
                continue
            placed = _place_for(camera, inst, category, inst.get("method"))
            plan.append({"obj": obj, "inst": inst, "category": category, "place": placed})
    plan.sort(key=lambda p: -p["place"]["width"])
    return plan


def _job_objects(plan, camera, detail=1.0):
    """What the mesh builder needs, largest objects first, within the model budget."""
    jobs = []
    for number, record in enumerate(plan):
        inst = record["inst"]
        if not inst.get("cutout"):
            continue
        record["model_id"] = "m%03d" % number
        jobs.append({
            "id": record["model_id"], "cutout": inst["cutout"], "category": record["category"],
            "label": inst.get("label") or record["category"], "bbox": inst.get("bbox_px"),
            "area": inst.get("area", 0.05), "quality": inst.get("quality"), "roughness": inst.get("roughness"),
            "detail": detail,
        })
        if len(jobs) >= MAX_GENERATED:
            break
    return jobs


def _put_mesh(ctx, model, x, y, base, scale, yaw, label, collision, colors=None):
    """Spawn a generated mesh with its base at height ``base``, painted."""
    z = base - model["origin_z"] * scale[2]
    loc = unreal.Vector(x - model["origin_x"] * scale[0], y - model["origin_y"] * scale[1], z)
    actor, component = rwb_layers._spawn(ctx, model["mesh"], loc, scale, label, yaw=yaw, collision=collision)
    rwb_layers._register(label, scale)
    rwb_generated.dress(ctx, component, model, colors or {}, "%s_%s" % (ctx["psd"], model["name"]))
    return actor


def _spawn_generated(ctx, model, record, label):
    """A new model made from this object's outline, on its painted spot."""
    placer = ctx["placer"]
    place = record["place"]
    category = record["category"]
    x, y = place["x"], place["y"]
    flat = model["method"] == "ground"
    lying_found = model["method"] == "adopt" and "scale_xy" in place
    if flat:
        s = place["scale_xy"] / max(1e-6, model["native_cm_per_unit"])
        scale = (s, s * place.get("scale_y", 1.0), s)
        wide = model["width"] * s
    elif lying_found:
        # its length follows the painted width: a real model's own depth is right, unlike a projection
        s = place["width"] / max(1e-6, max(model["width"], model["depth"]))
        scale = (s, s, s)
        wide = max(model["width"], model["depth"]) * s
    else:
        s = place["height"] / model["height"]
        scale = (s, s, s)
        wide = max(model["width"], model["depth"]) * s
    r = max(15.0, wide * 0.5)
    x, y = placer.painted_spot(x, y, r)
    in_water = category in WATER_OBJECTS and placer.water_z is not None and placer.wet(x, y) > 0.3
    if in_water:
        base = placer.water_z + 3.0
    else:
        low, _spread = placer.ground(x, y, r)
        base = low + place["z"]
        if not place["airborne"] and not flat:
            base -= model["height"] * s * 0.015             # settled into the ground, not on tiptoe
    yaw = 0.0 if flat else place.get("yaw", 0.0)
    tall = model["height"] * scale[2] > 120.0
    actor = _put_mesh(ctx, model, x, y, base, scale, yaw, label,
                      tall and not in_water and not place["airborne"], record["inst"].get("colors"))
    placer.add(x, y, r)
    record.update({"actor": actor, "spot": (x, y, r), "scale": scale, "label": label, "model": model})
    return actor


# --------------------------------------------------------------------------- #
# how many of each model: the panel's +/- buttons
# --------------------------------------------------------------------------- #
def record_key(record):
    """Stable across rebuilds: the name of the object's cut-out."""
    return Path(record["inst"]["cutout"]).stem


def _counts_path(cache_dir):
    return Path(cache_dir) / "model_counts.json"


def load_counts(cache_dir):
    try:
        return {str(k): int(v) for k, v in json.loads(_counts_path(cache_dir).read_text(encoding="utf-8")).items()}
    except Exception:
        return {}


def save_counts(cache_dir, counts):
    _counts_path(cache_dir).write_text(json.dumps(counts, indent=1), encoding="utf-8")


def _copy_spot(world, record, rng, r, in_water, airborne):
    """A free place for a copy: next to the original, or anywhere outside the painted view."""
    placer, camera = world["ctx"]["placer"], world["camera"]
    ox, oy, or_ = record["spot"]
    lim = camera.size / 2.0 - 250.0 - r
    for attempt in range(160):
        if world.get("keep_out"):
            x, y = rng.uniform(-lim, lim), rng.uniform(-lim, lim)
            if not rwb_layout.scatter_allowed(camera, x, y):
                continue
        else:
            angle = rng.uniform(0.0, 2.0 * math.pi)
            reach = (or_ + r) * (1.15 + attempt * 0.06) + rng.uniform(0.0, or_)
            x, y = ox + math.cos(angle) * reach, oy + math.sin(angle) * reach
        if airborne:
            if abs(x) < lim and abs(y) < lim and y < camera.player_y - 100.0:
                return x, y
            continue
        if placer.free(x, y, r, in_water=in_water, gap=12.0):
            return x, y
    return None


def _spawn_copy(world, record, number):
    """One more of this model, the same every rebuild (the spot is seeded by key + number)."""
    ctx, model, place = world["ctx"], record["model"], record["place"]
    placer = ctx["placer"]
    key = record_key(record)
    rng = random.Random("%s|%d" % (key, number))
    jitter = rng.uniform(0.85, 1.15)
    scale = tuple(v * jitter for v in record["scale"])
    r = record["spot"][2] * jitter
    category = record["category"]
    in_water = category in WATER_OBJECTS and placer.water_z is not None
    spot = _copy_spot(world, record, rng, r, in_water, place["airborne"])
    if spot is None and in_water:                       # no pond left: a boat on the bank beats nothing
        in_water = False
        spot = _copy_spot(world, record, rng, r, False, place["airborne"])
    if spot is None:
        return None
    x, y = spot
    if in_water:
        base = placer.water_z + 3.0
    else:
        low, _spread = placer.ground(x, y, r)
        base = low + place["z"] * jitter
    flat = model["method"] == "ground"
    label = "RWB_copy_%s_%03d" % (rwb_layers._safe(key, 20), number)
    actor = _put_mesh(ctx, model, x, y, base, scale, rng.uniform(0.0, 360.0), label,
                      model["height"] * scale[2] > 120.0 and not in_water and not place["airborne"] and not flat,
                      record["inst"].get("colors"))
    placer.add(x, y, r)
    return {"actor": actor, "spot": (x, y, r), "number": number}


# --------------------------------------------------------------------------- #
# fixing one model: a different shape type, or a redrawn outline
# --------------------------------------------------------------------------- #
SHAPE_NAMES = {None: "Auto", "inflate": "Puffy", "lathe": "Round", "slab": "Flat front", "ground": "Lying down",
               "cluster": "Leafy mass", "composite": "Building", "adopt": "Found model"}
SHAPE_ORDER = [None, "inflate", "lathe", "slab", "cluster", "composite", "ground"]


def _shapes_path(cache_dir):
    return Path(cache_dir) / "model_shapes.json"


def load_shapes(cache_dir):
    try:
        return json.loads(_shapes_path(cache_dir).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fix_model(args, record, method):
    """Build just this model with a chosen shape.  Cached in its own folder."""
    wb = _wb()
    inst = record["inst"]
    key = record_key(record)
    name = "%s_%s" % (key, method)
    models, lines = rwb_generated.build_models(
        [{"id": name, "name": name, "cutout": inst["cutout"], "category": record["category"], "method": method,
          "label": inst.get("label") or record["category"], "bbox": inst.get("bbox_px") or inst.get("bbox"),
          "area": inst.get("area", 0.05), "detail": args.get("detail", 1.0)}],
        args["camera_data"], args["name"], str(Path(args["cache_dir"]).resolve() / "fixes"), wb._run_helper, args["script"],
        use_blender=args["use_blender"], strokes=args["strokes"], stroke_strength=args["strength"],
        subdir="fix_" + name)
    return models.get(name), lines


def _remove_all(record):
    placer = _world["ctx"]["placer"]
    for item in record.get("copies", []):
        _destroy(item["actor"])
        placer.remove(*item["spot"])
    record["copies"] = []
    if record.get("actor"):
        _destroy(record["actor"])
        placer.remove(*record["spot"])
        record["actor"] = None


def set_shape(key, method):
    """Panel: rebuild one model with a different shape type (None = the tool's own choice)."""
    record = _record_for(key)
    if record is None:
        raise RuntimeError("Generate the world first.")
    count = (1 if record.get("actor") else 0) + len(record.get("copies", []))
    if method is None:
        model = record.get("auto_model") or record["model"]
        found = load_found(_world["cache_dir"])
        if found.pop(key, None) is not None:
            _found_path(_world["cache_dir"]).write_text(json.dumps(found, indent=1), encoding="utf-8")
    else:
        record.setdefault("auto_model", record["model"])
        model, lines = _fix_model(_world["build_args"], record, method)
        if not model:
            raise RuntimeError("; ".join(line for line in lines if line) or "that shape could not be built")
    _remove_all(record)
    record["model"] = model
    record["place"] = _place_for(_world["camera"], record["inst"], record["category"], model["method"])
    _spawn_generated(_world["ctx"], model, record, record.get("label") or "RWB_%s" % key)
    _apply_count(record, count or 1)
    shapes = load_shapes(_world["cache_dir"])
    if method is None:
        shapes.pop(key, None)
    else:
        shapes[key] = method
    _shapes_path(_world["cache_dir"]).write_text(json.dumps(shapes, indent=1), encoding="utf-8")
    record["shape"] = method
    return SHAPE_NAMES.get(method, method)


def _rebuild_with_shape(record, method):
    """Restore a shape chosen earlier, while the world is being built."""
    model, _lines = _fix_model(_world["build_args"], record, method)
    if not model:
        return
    record["auto_model"] = record["model"]
    _remove_all(record)
    record["model"] = model
    record["place"] = _place_for(_world["camera"], record["inst"], record["category"], model["method"])
    record["shape"] = method
    _spawn_generated(_world["ctx"], model, record, record["label"])


# --------------------------------------------------------------------------- #
# describe it: find a real model for one object
# --------------------------------------------------------------------------- #
def _found_path(cache_dir):
    return Path(cache_dir) / "model_found.json"


def load_found(cache_dir):
    try:
        return json.loads(_found_path(cache_dir).read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_models(key, query):
    """Search for models matching a description, ranked against this object's painted shape."""
    wb = _wb()
    record = _record_for(key)
    if record is None:
        raise RuntimeError("Generate the world first.")
    out = Path(_world["cache_dir"]).resolve() / "found" / ("search_%s.json" % key)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb._run_helper(Path(_world["build_args"]["script"]).with_name("reference_catalog.py"),
                   ["search", "--query", query, "--cutout", record["inst"]["cutout"], "--out", str(out)],
                   timeout=600, message="Searching model libraries for '%s'" % query)
    result = json.loads(out.read_text(encoding="utf-8"))
    _world.setdefault("searches", {})[key] = result
    return result


def use_match(key, index, own_look=False):
    """Replace one object with a found model, fitted to what was painted."""
    record = _record_for(key)
    result = (_world.get("searches") or {}).get(key)
    if record is None or not result or index >= len(result["matches"]):
        raise RuntimeError("Search first.")
    match = result["matches"][index]
    model, lines = _adopt_model(_world["build_args"], record, match, own_look)
    if not model:
        raise RuntimeError("; ".join(line for line in lines if line) or "that model couldn't be used")
    _swap_model(record, model, "adopt")
    found = load_found(_world["cache_dir"])
    found[key] = {"match": match, "own_look": bool(own_look), "query": result["query"]}
    _found_path(_world["cache_dir"]).write_text(json.dumps(found, indent=1), encoding="utf-8")
    return match


def forget_match(key):
    """Back to the model made from the painting."""
    found = load_found(_world["cache_dir"])
    found.pop(key, None)
    _found_path(_world["cache_dir"]).write_text(json.dumps(found, indent=1), encoding="utf-8")
    return set_shape(key, None)


def _adopt_model(args, record, match, own_look):
    wb = _wb()
    inst = record["inst"]
    key = record_key(record)
    # short names: Windows refuses paths over 260 characters, and these folders sit deep in Saved/
    import hashlib
    tag = hashlib.md5(("%s|%s|%s|%s" % (key, match.get("source"), match.get("id"), bool(own_look))).encode()).hexdigest()[:8]
    name = "f_" + tag
    models, lines = rwb_generated.build_models(
        [{"id": name, "name": name, "cutout": inst["cutout"], "category": record["category"], "source": match,
          "own_look": bool(own_look), "label": match.get("name"), "area": inst.get("area", 0.05),
          "detail": args.get("detail", 1.0), "bbox": inst.get("bbox_px") or inst.get("bbox")}],
        args["camera_data"], args["name"], str(Path(args["cache_dir"]).resolve() / "found"), wb._run_helper, args["script"],
        use_blender=args["use_blender"], strokes=args["strokes"], stroke_strength=args["strength"],
        subdir=name)
    return models.get(name), lines


def _swap_model(record, model, shape):
    """Put a different model in this object's place, keeping its count."""
    count = (1 if record.get("actor") else 0) + len(record.get("copies", []))
    record.setdefault("auto_model", record["model"])
    _remove_all(record)
    record["model"] = model
    record["place"] = _place_for(_world["camera"], record["inst"], record["category"], model["method"])
    _spawn_generated(_world["ctx"], model, record, record.get("label") or "RWB_%s" % record_key(record))
    _apply_count(record, count or 1)
    record["shape"] = shape


def next_shape(key):
    record = _record_for(key)
    current = record.get("shape") if record else None
    return SHAPE_ORDER[(SHAPE_ORDER.index(current) + 1) % len(SHAPE_ORDER)] if current in SHAPE_ORDER else None


def redraw_args(key):
    """What the sketch window needs to start from this model's current outline."""
    record = _record_for(key)
    if record is None:
        raise RuntimeError("Generate the world first.")
    inst = record["inst"]
    return {"cutout": inst["cutout"], "bbox": inst.get("bbox_px") or inst.get("bbox"), "key": key}


def _destroy(actor):
    try:
        unreal.EditorLevelLibrary.destroy_actor(actor)
    except Exception:
        pass


def _record_for(key):
    return next((r for r in _world.get("records", []) if record_key(r) == key), None)


def _apply_count(record, wanted):
    """Make the world hold ``wanted`` of this model.  Returns how many it really holds."""
    placer = _world["ctx"]["placer"]
    copies = record.setdefault("copies", [])
    wanted = max(0, min(MAX_COUNT, int(wanted)))
    # the painted original is the first; copies come and go after it
    while len(copies) > max(0, wanted - 1):
        item = copies.pop()
        _destroy(item["actor"])
        placer.remove(*item["spot"])
    if wanted == 0 and record.get("actor"):
        _destroy(record["actor"])
        placer.remove(*record["spot"])
        record["actor"] = None
    if wanted > 0 and not record.get("actor"):
        _spawn_generated(_world["ctx"], record["model"], record, record["label"])
    while len(copies) < wanted - 1:
        number = (copies[-1]["number"] + 1) if copies else 1
        item = _spawn_copy(_world, record, number)
        if item is None:
            break                                       # no room: report the real number
        copies.append(item)
    return (1 if record.get("actor") else 0) + len(copies)


def set_count(key, wanted):
    """Panel: change one model's count in place.  Returns the count it could reach."""
    record = _record_for(key)
    if record is None:
        raise RuntimeError("Generate the world first.")
    got = _apply_count(record, wanted)
    _world["counts"][key] = int(wanted) if got >= wanted else got
    save_counts(_world["cache_dir"], _world["counts"])
    return got


def set_keep_out(value):
    if _world:
        _world["keep_out"] = bool(value)


def model_rows():
    """One row per generated model for the panel, biggest first."""
    rows = []
    for record in _world.get("records", []):
        rows.append({
            "key": record_key(record),
            "title": record["inst"].get("label") or record["category"],
            "category": record["category"],
            "count": (1 if record.get("actor") else 0) + len(record.get("copies", [])),
            "thumb": record["model"].get("thumb"),
            "shape": SHAPE_NAMES.get(record.get("shape"), record.get("shape")),
            "search": (_world.get("searches") or {}).get(record_key(record)),
        })
    return rows


def select_model(key):
    """Select this model and all its copies in the level."""
    record = _record_for(key)
    if record is None:
        return 0
    actors = [record["actor"]] if record.get("actor") else []
    actors += [item["actor"] for item in record.get("copies", [])]
    unreal.get_editor_subsystem(unreal.EditorActorSubsystem).set_selected_level_actors(actors)
    return len(actors)


def forget():
    """The world was cleared."""
    _world.clear()


# --------------------------------------------------------------------------- #
# shapes drawn in the Shape Sketch window
# --------------------------------------------------------------------------- #
def drawn_dir(cache_dir):
    return Path(cache_dir).resolve() / "drawn"


def _drawn_records(cache_dir, camera):
    shapes = []
    try:
        shapes = json.loads((drawn_dir(cache_dir) / "shapes.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    return [_drawn_record(job, camera) for job in shapes if Path(job["cutout"]).is_file()]


def _drawn_record(job, camera):
    lying = job.get("method") == "ground" and camera.lies_flat(job)
    place = camera.place_ground(job) if lying else camera.place(job, "drawn")
    return {"obj": {"category": "drawn", "label": job["label"], "individual": True}, "inst": job,
            "category": "drawn", "place": place, "drawn": True}


def _build_drawn_model(world_or_args, record):
    """Mesh, Blender and bake for one drawn shape.  Cached by its own folder."""
    wb = _wb()
    args = world_or_args
    job = record["inst"]
    models, lines = rwb_generated.build_models(
        [{"id": job["id"], "name": job["name"], "cutout": job["cutout"], "category": "drawn", "method": job.get("method"),
          "label": job["label"], "bbox": job["bbox"], "area": job.get("area", 0.05), "detail": args.get("detail", 1.0)}],
        args["camera_data"], args["name"], str(drawn_dir(args["cache_dir"])), wb._run_helper, args["script"],
        use_blender=args["use_blender"], strokes=args["strokes"], stroke_strength=args["strength"],
        subdir="models_" + job["id"])
    return models.get(job["id"]), lines


def add_drawn(job):
    """Unreal side of Send to Unreal: build the shape and put it where it was drawn."""
    if not _world:
        raise RuntimeError("Generate the world first.")
    record = _drawn_record(job, _world["camera"])
    model, lines = _build_drawn_model(_world["build_args"], record)
    replaced = job.get("replaces")
    if not model:
        raise RuntimeError("; ".join(line for line in lines if line) or "the shape could not be modelled")
    folder = drawn_dir(_world["cache_dir"])
    path = folder / "shapes.json"
    try:
        shapes = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        shapes = []
    shapes = [s for s in shapes if s.get("id") != job["id"]] + [job]
    path.write_text(json.dumps(shapes, indent=1), encoding="utf-8")
    if replaced and _record_for(replaced):
        set_count(replaced, 0)                          # the new outline takes its place
    _spawn_generated(_world["ctx"], model, record, "RWB_%s" % rwb_layers._safe(job["id"], 20))
    _world["records"].append(record)
    return job["label"]


def poll_outbox(cache_dir):
    """Jobs the sketch window has sent and Unreal hasn't handled yet."""
    box = drawn_dir(cache_dir) / "outbox"
    if not box.is_dir():
        return []
    jobs = []
    for path in sorted(box.glob("*.json")):
        try:
            jobs.append(json.loads(path.read_text(encoding="utf-8")))
        finally:
            path.unlink(missing_ok=True)
    return jobs


def _spawn_fallback(ctx, record, models, label, cache_dir, rng):
    """No outline to model from: the closest library model, shapes, or a painted cut-out."""
    placer = ctx["placer"]
    place, inst, obj = record["place"], record["inst"], record["obj"]
    category = record["category"]
    title = obj["label"] if obj.get("individual") else category
    in_water = category in WATER_OBJECTS and placer.water_z is not None
    if models:
        model = min(models, key=lambda m: abs(math.log(max(0.05, m["height"] / m["width"]) /
                                                        max(0.05, inst.get("aspect", 1.0)))))
        scale = max(0.03, min(300.0, place["height"] / model["height"]))
        r = model["width"] * scale * 0.5
        x, y = placer.painted_spot(place["x"], place["y"], r)
        _spawn_model(ctx, model, x, y, scale, inst.get("colors") or {}, label, place.get("yaw", 0.0), in_water)
        placer.add(x, y, r)
        return "%s: library model %s" % (title, model["name"])
    if category in STRUCTURES:
        w, h = place["width"], place["height"]
        r = max(w, h * 0.3) * 0.6
        x, y = placer.painted_spot(place["x"], place["y"], r)
        _spawn_kit(ctx, category, x, y, w, h, inst.get("colors") or {}, label)
        placer.add(x, y, r)
        return "%s: built from simple shapes" % title
    if inst.get("texture"):
        x, y = placer.painted_spot(place["x"], place["y"], place["width"] * 0.35)
        if _spawn_card(ctx, obj, inst, x, y, label, cache_dir):
            placer.add(x, y, place["width"] * 0.35)
            return "%s: painted cut-out" % title
    return "%s: nothing to build it from - add a model named '%s' to ModelLibrary" % (title, obj.get("word") or category)


def _scatter_extras(ctx, obj, models, generated, camera, density, area, rng, total):
    """Extra copies of a repeated thing, outside the painted view.  Returns how many were made."""
    placer = ctx["placer"]
    category = obj["category"]
    in_water = category in WATER_OBJECTS
    size = camera.size
    want = min(_target_count(obj, density, area) - obj["count"], TOTAL_CAP - total)
    made = tries = 0
    palettes = [i.get("colors") or {} for i in (obj.get("instances") or [])] or [{}]
    while made < want and tries < want * 25:
        tries += 1
        x = rng.uniform(-size / 2, size / 2)
        y = rng.uniform(-size / 2, size / 2)
        if not rwb_layout.scatter_allowed(camera, x, y):
            continue
        label = "RWB_%s_extra_%03d" % (rwb_layers._safe(category, 16), made)
        if models:
            model = rng.choice(models)
            scale, target = _scale_for(model, category, ctx["size"], rng.uniform(0.8, 1.2))
            r = model["width"] * scale * 0.5 * (0.75 if category in ("tree", "pine") else 1.0)
            if not placer.free(x, y, r, tall=False, in_water=in_water, gap=15.0 if target < 120 else 40.0):
                continue
            _spawn_model(ctx, model, x, y, scale, palettes[made % len(palettes)], label, rng.uniform(0, 360), in_water)
        else:
            model = rng.choice(generated)
            if model["method"] == "ground":
                return made
            _axis, value = PRIORS.get(category, ("h", 200))
            s = value * ctx["size"] * rng.uniform(0.8, 1.2) / model["height"]
            r = max(model["width"], model["depth"]) * s * 0.5
            if not placer.free(x, y, r, gap=40.0):
                continue
            low, _spread = placer.ground(x, y, r)
            loc = unreal.Vector(x, y, low - model["origin_z"] * s - model["height"] * s * 0.015)
            actor, component = rwb_layers._spawn(ctx, model["mesh"], loc, (s, s, s), label, yaw=rng.uniform(0, 360),
                                                 collision=model["height"] * s > 120.0)
            rwb_layers._register(label, (s, s, s))
            rwb_generated.dress(ctx, component, model, palettes[made % len(palettes)],
                                "%s_%s" % (ctx["psd"], model["name"]))
        placer.add(x, y, r)
        made += 1
    return made


def _scatter_painted_mass(ctx, obj, models, camera, density, rng):
    """PSD flower / grass layers: plants scattered only where they were painted."""
    placer = ctx["placer"]
    coverage = obj.get("coverage")
    if not models or not coverage:
        return 0
    category = obj["category"]
    want = int(30 + 120 * density)
    made = tries = 0
    palettes = [i.get("colors") or {} for i in (obj.get("instances") or [])] or [{}]
    span = camera.far
    while made < want and tries < want * 30:
        tries += 1
        y = camera.player_y - rng.uniform(rwb_layout.NEAREST_CM, span)
        half = (camera.W / 2.0) * (camera.player_y - y) / camera.focal
        x = rng.uniform(-half, half)
        img = camera.to_image(x, y)
        if not img or rng.random() > _sample_grid(coverage, img[0], img[1]) * 1.5:
            continue
        model = rng.choice(models)
        scale, _target = _scale_for(model, category, ctx["size"], rng.uniform(0.75, 1.25))
        r = model["width"] * scale * 0.5
        if not placer.free(x, y, r, gap=8.0):
            continue
        _spawn_model(ctx, model, x, y, scale, palettes[made % len(palettes)],
                     "RWB_%s_%03d" % (rwb_layers._safe(category, 16), made), rng.uniform(0, 360))
        placer.add(x, y, r)
        made += 1
    return made


def _camera_data(camera):
    return {"W": camera.W, "H": camera.H, "horizon": camera.horizon, "focal": camera.focal, "eye": camera.eye,
            "player_y": camera.player_y}


def _paint_ground(ctx, terrain_actor, surfaces, camera, heights, water_z, inv, cache_dir, root_dir):
    """Bake the top-down surface map (outside Unreal: it needs numpy) and lay it on the terrain."""
    wb = _wb()
    folder = Path(cache_dir).resolve()
    job = folder / "ground_job.json"
    out = folder / "ground_paint.png"
    rock = wb._hex_rgb((inv.get("ground") or {}).get("rock") or "#6E6A64")
    job.write_text(json.dumps({
        "camera": _camera_data(camera), "world_size": camera.size, "surfaces": surfaces["surfaces"],
        "cells": surfaces["cells"], "heights": heights, "water_z": water_z, "rock_color": list(rock),
        "out": str(out)}), encoding="utf-8")
    wb._run_helper(Path(root_dir) / "reference_surfaces.py", ["--job", str(job)], timeout=300,
                   message="Painting the ground from your picture")
    component = (terrain_actor.get_component_by_class(unreal.ProceduralMeshComponent)
                 or terrain_actor.get_editor_property("root_component"))
    rwb_generated.paint_ground(ctx, component, str(out), cache_dir)


def _surface_here(surfaces, camera, x, y):
    """The surface painted at a ground position.  Outside the picture, the ground's main surface."""
    kinds = surfaces.get("surfaces") or []
    cells = surfaces.get("cells")
    if not kinds or not cells:
        return None
    main = max(kinds, key=lambda s: s["share"])
    img = camera.to_image(x, y)
    if not img:
        return main["name"]
    rows, cols = len(cells), len(cells[0])
    return kinds[cells[min(rows - 1, int(img[1] * rows))][min(cols - 1, int(img[0] * cols))]]["name"]


GRASS_SURFACES = {"grass", "meadow"}


def _grow_grass(ctx, surfaces, camera, library, density, area, rng):
    """Grass and flower models where the picture's ground is grassy, in its own colours."""
    placer = ctx["placer"]
    kinds = surfaces.get("surfaces") or []
    grassy = [s for s in kinds if s["name"] in GRASS_SURFACES]
    if not grassy:
        return 0
    grass_models = list(library.get("grass", []))
    flower_models = list(library.get("flower", []))
    if not grass_models and not flower_models:
        return 0
    size = camera.size
    # a model's flat tint reads far darker than the same colour in paint: lift it, or dark painted
    # grass turns into black tufts
    colours = {s["name"]: "#%02X%02X%02X" % tuple(min(255, int(c * GRASS_LIFT + 25)) for c in s["color"]) for s in grassy}
    want = int(min(GRASS_CAP, (60 + 240 * density) * area))
    made = tries = 0
    while made < want and tries < want * 12:
        tries += 1
        x, y = rng.uniform(-size / 2, size / 2), rng.uniform(-size / 2, size / 2)
        name = _surface_here(surfaces, camera, x, y)
        if name not in GRASS_SURFACES:
            continue
        # a meadow is the flowery kind
        pool = flower_models if name == "meadow" and flower_models and rng.random() < 0.5 else (grass_models or flower_models)
        model = rng.choice(pool)
        category = "flower" if model in flower_models else "grass"
        scale, _target = _scale_for(model, category, ctx["size"], rng.uniform(0.7, 1.3))
        r = model["width"] * scale * 0.5
        if not placer.free(x, y, r, gap=4.0):
            continue
        hexed = colours[name]
        _spawn_model(ctx, model, x, y, scale, {"dominant": hexed, "top": hexed, "bottom": hexed, "accent": hexed},
                     "RWB_%s_%03d" % (category, made), rng.uniform(0, 360))
        placer.add(x, y, r)
        made += 1
    return made


def build_world(analysis, values, root_dir, project_dir, keywords_path):
    wb = _wb()
    inv = analysis.get("inventory") or {}
    if inv.get("error"):
        raise RuntimeError("the object inventory failed: " + inv["error"])
    seed = int(values.get("seed", 1))
    rng = random.Random(seed)
    size = max(4000.0, float(values["world_size"]) * 2.0)
    camera = _picture_camera(inv, analysis, values, size)
    messages = rwb_library.ensure_imported(root_dir, project_dir)
    library = rwb_library.index(keywords_path)
    name = rwb_layers._safe(analysis.get("name", "ref"), 24)
    cache_dir = (analysis.get("scene") or {}).get("cache") or analysis.get("layer_dir") or "."
    plan = _plan(inv, camera) + _drawn_records(cache_dir, camera)

    # ---- new 3D models from the painting
    generated = {}
    camera_data = {"W": camera.W, "H": camera.H, "horizon": camera.horizon, "focal": camera.focal}
    strokes = inv.get("strokes") if not (inv.get("strokes") or {}).get("error") else None
    strength = float(values.get("stroke_strength", 1.0))
    use_blender = bool(values.get("use_blender", True))
    detail = float(values.get("model_detail", 1.0))
    build_args = {"camera_data": camera_data, "name": name, "cache_dir": cache_dir, "use_blender": use_blender,
                  "detail": detail,
                  "strokes": strokes, "strength": strength, "script": Path(root_dir) / "reference_mesh.py"}
    if values.get("generate_models", True):
        jobs = _job_objects([r for r in plan if not r.get("drawn")], camera, detail)
        if jobs:
            try:
                generated, lines = rwb_generated.build_models(
                    jobs, camera_data, name, cache_dir, wb._run_helper, build_args["script"], use_blender=use_blender,
                    strokes=strokes, stroke_strength=strength)
                messages += [line for line in lines if line]
            except Exception as error:
                messages.append("could not build new models (%s): using library models" % str(error)[:160])
        for record in plan:
            if not record.get("drawn"):
                continue
            try:
                model, lines = _build_drawn_model(build_args, record)
            except Exception as error:
                model, lines = None, ["could not rebuild %s (%s)" % (record["inst"].get("label"), str(error)[:120])]
            if model:
                record["model_id"] = record["inst"]["id"]
                generated[record["model_id"]] = model
            else:
                messages += [line for line in lines if line]

    # ---- terrain, keeping painted things that stand on land dry
    wb._progress_message("Shaping the terrain and water", 90.0)
    land = []
    for record in plan:
        if record["category"] not in WATER_OBJECTS and not record["place"]["airborne"]:
            # enough for its base; a wide building mustn't erase the lake it stands in
            land.append((record["place"]["x"], record["place"]["y"], min(380.0, max(120.0, record["place"]["width"] * 0.45))))
    heights, water_z, water = make_terrain(inv, camera, float(values["height"]), rng, n=128, land=land)
    terrain_actor = build_terrain_actor(heights, water_z, water, inv, size, rng)
    water_actor = None
    if water_z is not None:
        water_actor = wb._build_water(size, size, water_z, (inv.get("water") or {}).get("color") or "#3E7FA6")
    overrides = {k: v for k, v in (("sky", inv.get("sky")), ("fog", inv.get("fog")), ("accent", inv.get("accent")),
                                     ("water", (inv.get("water") or {}).get("color"))) if v}
    wb._lighting(analysis, size, size, heights, overrides)
    ctx = {
        "count": 0, "psd": name, "folder": rwb_layers.LAYER_ROOT + "/" + name, "paint": False, "blend": 0.0,
        "art_blend": float(values.get("art_blend", 1.0)) if values.get("paint_art", True) else 0.0,
        "size": float(values.get("object_scale", 1.0)),
        "masters": rwb_layers._masters(False),
        "shapes": {k: unreal.load_asset(p) for k, p in rwb_layers.SHAPES.items()},
        "placer": Placer(size, heights, water_z, water, (0.0, camera.player_y)),
    }
    surfaces = inv.get("surfaces") or {}
    if surfaces.get("surfaces"):
        try:
            _paint_ground(ctx, terrain_actor, surfaces, camera, heights, water_z, inv, cache_dir, root_dir)
        except Exception as error:
            messages.append("ground keeps plain colours (%s)" % str(error)[:140])
        if surfaces.get("line"):
            messages.append(surfaces["line"])
    water_tile = (inv.get("water") or {}).get("tile")
    if water_actor and water_tile and Path(water_tile).is_file():
        try:
            rwb_generated.paint_water(ctx, water_actor, water_tile, cache_dir)
        except Exception as error:
            messages.append("water keeps a flat colour (%s)" % str(error)[:120])
    density = float(values.get("density", 0.55))
    area = (size / 8000.0) ** 2
    report = []
    total = 0
    tally = {}
    records = []

    # ---- every painted object, where it was painted
    for number, record in enumerate(plan):
        wb._progress_message("Placing %d of %d objects where you painted them" % (number + 1, len(plan)),
                             92.0 + 7.0 * number / max(1, len(plan)))
        category = record["category"]
        title = record["obj"]["label"] if record["obj"].get("individual") else (record["inst"].get("label") or category)
        label = "RWB_%s_%02d" % (rwb_layers._safe(title, 22), number)
        model = generated.get(record.get("model_id"))
        if model:
            _spawn_generated(ctx, model, record, label)
            records.append(record)
            tally[title] = tally.get(title, 0) + 1
        else:
            models = list(library.get(category, []))
            for other in FALLBACK.get(category, []):
                if models:
                    break
                models = list(library.get(other, []))
            report.append(_spawn_fallback(ctx, record, models, label, cache_dir, rng))
        total += 1
    if tally:
        report.insert(0, "New 3D models, placed as painted: " + ", ".join(
            "%s x%d" % (k, v) if v > 1 else k for k, v in sorted(tally.items())))

    # ---- remember this world, and restore the counts chosen for each model
    _world.clear()
    _world.update({"ctx": ctx, "camera": camera, "records": records, "cache_dir": cache_dir, "name": name,
                   "build_args": build_args, "keep_out": bool(values.get("keep_copies_out", False)),
                   "counts": load_counts(cache_dir)})
    short = []
    shapes = load_shapes(cache_dir)
    found = load_found(cache_dir)
    for record in records:
        if record_key(record) in found:
            item = found[record_key(record)]
            try:
                model, _lines = _adopt_model(build_args, record, item["match"], item.get("own_look"))
                if model:
                    _swap_model(record, model, "adopt")
                    report.append("%s: using %s (%s)" % (record_key(record), item["match"].get("name"), item["match"].get("license")))
                    continue
            except Exception as error:
                short.append("%s: kept the painted model (%s)" % (record_key(record), str(error)[:100]))
        chosen = shapes.get(record_key(record))
        if chosen:
            try:
                _rebuild_with_shape(record, chosen)
            except Exception as error:
                short.append("%s: kept the automatic shape (%s)" % (record_key(record), str(error)[:100]))
        wanted = _world["counts"].get(record_key(record))
        if wanted is None:
            continue
        got = _apply_count(record, wanted)
        total += max(0, got - 1)
        if got < wanted:
            short.append("%s: room for %d of %d" % (record["inst"].get("label") or record["category"], got, wanted))
    report += short

    # ---- grass grows where the ground is grassy
    if values.get("grow_grass", True):
        grown = _grow_grass(ctx, surfaces, camera, library, density, area, rng)
        total += grown
        if grown:
            report.append("grass: %d plants on the grassy ground" % grown)

    # ---- painted masses, then extra copies outside the view
    by_category = {}
    for record in plan:
        model = generated.get(record.get("model_id"))
        if model:
            by_category.setdefault(record["category"], []).append(model)
    for obj in inv.get("objects") or []:
        category = obj["category"]
        models = list(library.get(category, []))
        for other in FALLBACK.get(category, []):
            if models:
                break
            models = list(library.get(other, []))
        if obj.get("mass"):
            made = _scatter_painted_mass(ctx, obj, models, camera, density, rng)
            total += made
            report.append("%s: %d plants where they were painted" % (category, made) if made
                          else "%s: painted, but no 3D model - add one named '%s' to ModelLibrary" % (category, category))
            continue
        # a forest needs more than the trees it was painted with, even if only one was recognised
        if not values.get("extra_copies", True) or (obj["unique"] and category not in FOREST):
            continue
        extras = by_category.get(category) or []
        if not models and not extras:
            continue
        made = _scatter_extras(ctx, obj, models, extras, camera, density, area, rng, total)
        total += made
        if made:
            report.append("%s: seen %d times, %d extra copies outside the painted view" % (category, obj["count"], made))

    try:
        import rwb_scene
        rwb_scene._grade(analysis)
    except Exception:
        pass
    _player_start(camera, heights, size)
    try:
        unreal.EditorAssetLibrary.save_directory(ctx["folder"], only_if_is_dirty=True, recursive=True)
    except Exception:
        pass
    stroke_line = (inv.get("strokes") or {}).get("line")
    return {"report": ([stroke_line] if stroke_line else []) + messages + report, "count": ctx["count"],
            "info": inv.get("info", "")}


def _player_start(frame, heights, size):
    wb = _wb()
    x, y = 0.0, frame.player_y
    z = wb._sample_height(heights, 0.5, y / size + 0.5) + 120.0
    location = unreal.Vector(x, y, z)
    starts = [a for a in wb._all_actors() if isinstance(a, unreal.PlayerStart)]
    if starts:
        starts[0].set_actor_location(location, False)
        starts[0].set_actor_rotation(unreal.Rotator(yaw=-90.0), False)
        return starts[0]
    start = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PlayerStart, location, unreal.Rotator(yaw=-90.0))
    wb._tag_actor(start, wb.GENERATED_TAG, wb.PLAYER_TAG, label="RWB_PlayerStart")
    return start
