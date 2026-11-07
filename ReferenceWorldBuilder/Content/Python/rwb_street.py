"""Streets and rooms: walls built from the picture itself.  (Reference World Builder)

A street of buildings or a room is mostly facades and walls, and those are too big to cut out as
objects.  Instead the picture is projected onto two side walls and an end wall from the painter's
eye (the start position), so from there the walls look exactly like the painting, and walking down
the street they keep its shop fronts, signs and windows where they were painted.

The geometry below is plain Python (tested offline); ``build`` makes the Unreal actors.
"""

import math

WALL_COLS = 48                # segments along a side wall (the projection is not linear: subdivide)
WALL_ROWS = 16                # segments up a wall
MIN_WALL_CM = 700.0
MAX_WALL_CM = 4500.0
BEHIND_CM = 400.0             # the side walls start this far behind the start position
NIGHT_SELF_LIT = 1.1          # neon and lit windows glow at night
DAY_SELF_LIT = 0.6
LIGHT_OFF_WALL = 60.0         # a sign's light hangs this far in front of its wall


def wall_uv(camera, x, y, z):
    """Where a point on a wall shows up in the picture (u, v), clamped to the picture.

    Points beside or behind the start position are outside the picture; they take its edge.
    """
    distance = max(60.0, camera.player_y - y)
    u = 0.5 + x * camera.focal / (distance * camera.W)
    v = (camera.horizon - (z - camera.eye) * camera.focal / distance) / camera.H
    return min(1.0, max(0.0, u)), min(1.0, max(0.0, v))


def wall_top(camera, y, top_row=0.0):
    """How tall a wall is at a distance: up to where the picture's buildings end (the sky)."""
    distance = max(60.0, camera.player_y - y)
    seen = camera.eye + (camera.horizon - float(top_row)) * distance / camera.focal
    return max(MIN_WALL_CM, min(MAX_WALL_CM, seen))


def _grid(points, cols, rows):
    """Triangles for a (cols+1) x (rows+1) grid, both sides (a wall is seen from either side)."""
    tris = []
    for c in range(cols):
        for r in range(rows):
            a = c * (rows + 1) + r
            b = a + rows + 1
            tris += [a, b, a + 1, a + 1, b, b + 1]          # one side
            tris += [a, a + 1, b, a + 1, b + 1, b]          # the other
    return tris


def side_wall(camera, x, top_row=0.0, cols=WALL_COLS, rows=WALL_ROWS):
    """A wall along the street at ``x`` (cm), from behind the start to the far end.

    Returns (verts [(x,y,z)], uvs [(u,v)], triangles).
    """
    y_near = camera.player_y + BEHIND_CM
    y_far = camera.player_y - camera.far
    verts, uvs = [], []
    for c in range(cols + 1):
        # denser near the start, where the projection changes fastest
        t = (c / float(cols)) ** 1.6
        y = y_near + (y_far - y_near) * t
        top = wall_top(camera, y, top_row)
        for r in range(rows + 1):
            z = top * r / float(rows)
            verts.append((x, y, z))
            uvs.append(wall_uv(camera, x, y, z))
    return verts, uvs, _grid(verts, cols, rows)


def end_wall(camera, half, top_row=0.0, cols=16, rows=WALL_ROWS):
    """The far end of the street: what the painting shows down the middle."""
    y = camera.player_y - camera.far
    top = wall_top(camera, y, top_row)
    verts, uvs = [], []
    for c in range(cols + 1):
        x = -half + 2.0 * half * c / float(cols)
        for r in range(rows + 1):
            z = top * r / float(rows)
            verts.append((x, y, z))
            uvs.append(wall_uv(camera, x, y, z))
    return verts, uvs, _grid(verts, cols, rows)


def light_spot(camera, spot, half):
    """Where a glowing patch of the picture (u, v) lands on the street's walls: (x, y, z) or None."""
    du = float(spot["u"]) - 0.5
    if abs(du) < 1e-3:
        distance = camera.far
    else:
        distance = half * camera.focal / (abs(du) * camera.W)
    distance = max(300.0, min(camera.far, distance))
    x = du * camera.W * distance / camera.focal
    x = max(-half + LIGHT_OFF_WALL, min(half - LIGHT_OFF_WALL, x))
    z = camera.eye + (camera.horizon - float(spot["v"]) * camera.H) * distance / camera.focal
    if z < 30.0:
        return None                        # a reflection on the wet street, not a sign
    return x, camera.player_y - distance, min(z, MAX_WALL_CM)


def street_half_width(places, default=650.0, low=450.0, high=1600.0):
    """Half the width of a street (cm) from where its painted things stand.

    places: [(x, airborne)] ground positions in cm.  People and things on the ground stand in the
    street, so it is at least as wide as they are spread; signs on the facades don't count.
    """
    xs = sorted(abs(x) for x, airborne in places if not airborne)
    if not xs:
        return default
    spread = xs[min(len(xs) - 1, int(math.ceil(0.85 * (len(xs) - 1))))]
    return float(max(low, min(high, max(default * 0.85, spread * 1.15 + 120.0))))


ON_FACADES = {"sign"}          # these hang on the buildings, wherever their lowest point is


def on_wall(camera, inst, half):
    """Placement for a sign painted on a facade: where its picture position meets the wall.

    Like PaintCamera.place, but the distance comes from the wall, not the ground or the depth map.
    """
    u = float(inst.get("centre_u", inst.get("contact_u", 0.5)))
    du = u - 0.5
    distance = camera.far if abs(du) < 1e-3 else half * camera.focal / (abs(du) * camera.W)
    distance = max(400.0, min(camera.far, distance))
    per_pixel = distance / camera.focal
    contact_row = float(inst["contact_v"]) * camera.H
    top_row = float(inst.get("top_v", inst["contact_v"])) * camera.H
    x = du * camera.W * per_pixel
    return {"x": max(-half, min(half, x)), "y": camera.player_y - distance, "distance": distance,
            "z": max(0.0, camera.eye + (camera.horizon - contact_row) * per_pixel),
            "width": max(4.0, float(inst.get("width_frac", 0.1)) * camera.W * per_pixel),
            "height": max(8.0, (contact_row - top_row) * per_pixel),
            "yaw": 0.0, "airborne": True}


def replace_in_street(camera, record, half):
    """Where a painted thing goes in a street.

    Depth maps of street photos are unreliable (signs and glass), so ground things are placed by
    where they meet the ground alone, and signs by the facade they hang on.
    """
    inst = dict(record["inst"], depth_eye=None)
    if record["category"] in ON_FACADES:
        record["place"] = on_wall(camera, inst, half)
    elif "contact_v" in inst and record["place"].get("scale_xy") is None:
        record["place"] = camera.place(inst, record["category"])
    return keep_in_street(record["place"], half)


def keep_in_street(place, half, margin=40.0):
    """Painted things stay inside the street (a facade sign hangs just in front of its wall)."""
    reach = max(10.0, half - margin - place.get("width", 0.0) * 0.25)
    place["x"] = max(-reach, min(reach, place["x"]))
    return place


# --------------------------------------------------------------------------- #
# individual buildings along the street (from reference_facades)
# --------------------------------------------------------------------------- #
BUILDING_DEPTH_CM = 1400.0
MAX_SETBACK_CM = 45.0
MIN_BUILDING_CM_HIGH = 600.0
MAX_BUILDING_CM_HIGH = 6000.0


def image_uv(camera, x, y, z):
    """Where a point shows up in the picture (u, v), not clamped."""
    distance = max(1.0, camera.player_y - y)
    return (0.5 + x * camera.focal / (distance * camera.W),
            (camera.horizon - (z - camera.eye) * camera.focal / distance) / camera.H)


def street_distance(camera, half, u):
    """Distance along the street seen at picture column u on a side facade."""
    du = abs(float(u) - 0.5)
    return camera.far if du < 1e-3 else half * camera.focal / (du * camera.W)


def building_blocks(camera, half, sides, seed=1):
    """One block per building reference_facades found.

    Returns [{side, number, y_near, y_far, x_front, x_back, height, color, open_top}], nearest first.
    """
    import random
    rng = random.Random(seed)
    out = []
    for side in ("left", "right"):
        segments = (sides or {}).get(side) or []
        sign = -1.0 if side == "left" else 1.0
        previous = -BEHIND_CM                     # the first building starts behind the start position
        for number, seg in enumerate(segments):
            if previous >= camera.far - 150.0:
                break
            d_near = previous
            d_far = min(camera.far, street_distance(camera, half, seg["u_far"]))
            if number == len(segments) - 1:
                d_far = camera.far                # the last one runs to the end of the street
            if d_far - d_near < 150.0:
                continue
            previous = d_far
            heights = []
            for u, v in seg.get("roof") or []:
                if v * camera.H <= 1.0:
                    continue                      # its roof is above the frame there
                d = max(200.0, min(camera.far, street_distance(camera, half, u)))
                heights.append(camera.eye + (camera.horizon - v * camera.H) * d / camera.focal)
            seen = camera.eye + camera.horizon * max(200.0, d_far) / camera.focal   # the frame's top, at its far end
            if seg.get("open_top") or not heights:
                height = seen
            else:
                heights.sort()
                height = heights[len(heights) // 2]
            height = max(MIN_BUILDING_CM_HIGH, min(MAX_BUILDING_CM_HIGH, height))
            setback = 0.0 if number == 0 else rng.uniform(0.0, MAX_SETBACK_CM)
            x_front = sign * (half + setback)
            out.append({"side": side, "number": len([b for b in out if b["side"] == side]) + 1,
                        "y_near": camera.player_y - d_near, "y_far": camera.player_y - d_far,
                        "x_front": x_front, "x_back": x_front + sign * BUILDING_DEPTH_CM, "height": height,
                        "color": list(seg.get("color") or (128, 128, 128)), "open_top": bool(seg.get("open_top"))})
    return out


def _quad_grid(corners, cols, rows):
    """Points of a bilinear grid over four corners (a, b, c, d), column by column from a-d to b-c."""
    a, b, c, d = corners
    pts = []
    for i in range(cols + 1):
        s = i / float(cols)
        bottom = [a[k] + (b[k] - a[k]) * s for k in range(3)]
        top = [d[k] + (c[k] - d[k]) * s for k in range(3)]
        for j in range(rows + 1):
            t = j / float(rows)
            pts.append(tuple(bottom[k] + (top[k] - bottom[k]) * t for k in range(3)))
    return pts


def _two_sided(quads):
    verts, tris = [], []
    for quad in quads:
        base = len(verts)
        verts += list(quad)
        tris += [base, base + 1, base + 2, base, base + 2, base + 3, base, base + 2, base + 1, base, base + 3, base + 2]
    return verts, tris


def block_mesh(camera, block, cols=12, rows=10):
    """A building's faces: the front painted from the picture, roof and ends in its own colour.

    Returns {"front": (verts, uvs, tris), "plain": (verts, tris)}.  Both windings, like the walls.
    """
    x0, x1 = block["x_front"], block["x_back"]
    y0, y1, top = block["y_near"], block["y_far"], block["height"]
    verts = _quad_grid([(x0, y0, 0.0), (x0, y1, 0.0), (x0, y1, top), (x0, y0, top)], cols, rows)
    uvs = [tuple(min(1.0, max(0.0, v)) for v in image_uv(camera, *p)) for p in verts]
    front = (verts, uvs, _grid(verts, cols, rows))
    plain = _two_sided([
        [(x0, y0, top), (x0, y1, top), (x1, y1, top), (x1, y0, top)],         # roof
        [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y0, top), (x0, y0, top)],         # near end
        [(x0, y1, 0.0), (x1, y1, 0.0), (x1, y1, top), (x0, y1, top)],         # far end
    ])
    return {"front": front, "plain": plain}


# --------------------------------------------------------------------------- #
# signs as boards you can read
# --------------------------------------------------------------------------- #
BLADE_ASPECT = 1.6           # a sign this much taller than wide sticks out from its wall, facing down the street
CENTRAL = 0.22               # a sign this close to the middle hangs across the street, facing you
BOARD_OFF_WALL = 25.0
BOARD_THICK = {"facade": 12.0, "blade": 16.0, "front": 12.0}


def sign_kind(box_frac, W, H):
    u0, v0, u1, v1 = box_frac
    aspect = (v1 - v0) * H / max(1e-6, (u1 - u0) * W)
    if aspect >= BLADE_ASPECT:
        return "blade"
    if abs((u0 + u1) / 2.0 - 0.5) < CENTRAL:
        return "front"
    return "facade"


def board_corners(camera, box_frac, half=None, place=None):
    """The board's corners (bottom-left, bottom-right, top-right, top-left as you face it), kind, normal.

    In a street (half given) the distance comes from the wall the sign hangs on; elsewhere (``place``
    from PaintCamera.place) it stands where it was placed, turned to face the start.
    """
    u0, v0, u1, v1 = box_frac
    W, H, f = camera.W, camera.H, camera.focal

    def at(u, v, d):
        return ((u - 0.5) * W * d / f, camera.player_y - d, camera.eye + (camera.horizon - v * H) * d / f)

    if half is None:
        p = place
        nx, ny = -p["x"], camera.player_y - p["y"]
        length = math.hypot(nx, ny) or 1.0
        nx, ny = nx / length, ny / length
        rx, ry = ny, -nx
        w, h, z = p["width"] / 2.0, p["height"], p.get("z", 0.0)
        bl = (p["x"] - rx * w, p["y"] - ry * w, z)
        br = (p["x"] + rx * w, p["y"] + ry * w, z)
        return [bl, br, (br[0], br[1], z + h), (bl[0], bl[1], z + h)], "front", (nx, ny, 0.0)
    kind = sign_kind(box_frac, W, H)
    if kind == "facade":
        side = -1.0 if (u0 + u1) < 1.0 else 1.0
        x = side * (half - BOARD_OFF_WALL)
        ends = []
        for u in (u0, u1):
            d = max(300.0, min(camera.far, street_distance(camera, half - BOARD_OFF_WALL, u)))
            ends.append((d, camera.eye + (camera.horizon - v1 * H) * d / f, camera.eye + (camera.horizon - v0 * H) * d / f))
        (da, za0, za1), (db, zb0, zb1) = ends
        corners = [(x, camera.player_y - da, za0), (x, camera.player_y - db, zb0),
                   (x, camera.player_y - db, zb1), (x, camera.player_y - da, za1)]
        return corners, kind, (-side, 0.0, 0.0)
    # a blade or a sign across the street: faces the start, where its outer edge meets a wall
    outer = max(abs(u0 - 0.5), abs(u1 - 0.5))
    d = camera.far if outer < 1e-3 else half * f / (outer * W)
    d = max(300.0, min(camera.far, d))
    return [at(u0, v1, d), at(u1, v1, d), at(u1, v0, d), at(u0, v0, d)], kind, (0.0, 1.0, 0.0)


def board_mesh(camera, board, half=None, place=None, cols=6, rows=6):
    """A sign board: its face (and, for a blade, its back) shows the sign's own crop, read the right way.

    Returns ({"face": (verts, uvs, tris), "frame": (verts, tris)}, kind).
    """
    corners, kind, normal = board_corners(camera, board["box_frac"], half, place)
    thick = BOARD_THICK[kind]
    nx, ny, _nz = normal
    front = [(p[0] + nx * thick / 2.0, p[1] + ny * thick / 2.0, p[2]) for p in corners]
    back = [(p[0] - nx * thick / 2.0, p[1] - ny * thick / 2.0, p[2]) for p in corners]
    cu0, cv0, cu1, cv1 = board["box_frac"]

    def crop_uv(p):
        u, v = image_uv(camera, *p)
        return (min(1.0, max(0.0, (u - cu0) / max(1e-6, cu1 - cu0))),
                min(1.0, max(0.0, (v - cv0) / max(1e-6, cv1 - cv0))))

    # picture positions come from the board's middle plane, so the crop lines up with the sign
    mid = _quad_grid(corners, cols, rows)
    face_v = _quad_grid(front, cols, rows)
    face_uv = [crop_uv(p) for p in mid]
    tris = _grid(face_v, cols, rows)
    if kind == "blade":
        # printed on both sides: the back is mirrored back, so it reads from behind as well
        n = len(face_v)
        face_v = face_v + _quad_grid(back, cols, rows)
        face_uv = face_uv + [(1.0 - u, v) for u, v in face_uv]
        tris = tris + [t + n for t in tris]
    rim = [(front[i], front[(i + 1) % 4], back[(i + 1) % 4], back[i]) for i in range(4)]
    if kind != "blade":
        rim.append(tuple(back))
    return {"face": (face_v, face_uv, tris), "frame": _two_sided(rim)}, kind


# --------------------------------------------------------------------------- #
# Unreal
# --------------------------------------------------------------------------- #
def build(ctx, camera, half, picture_path, cache_dir, top_row=0.0, night=False, room=False, walls=True):
    """Spawn the walls, painted with the picture.  walls=False: only the far end (buildings line the sides)."""
    import unreal
    import rwb_generated
    import rwb_layers
    import rwb_scene
    import world_builder as wb

    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.Actor, unreal.Vector(0, 0, 0))
    component = unreal.new_object(unreal.ProceduralMeshComponent, outer=actor, name="RWBStreet")
    actor.set_editor_property("root_component", component)
    component.set_editor_property("mobility", unreal.ComponentMobility.STATIC)
    parts = ([side_wall(camera, -half, top_row), side_wall(camera, half, top_row)] if walls else []) + \
        [end_wall(camera, half, top_row)]
    for section, (verts, uvs, tris) in enumerate(parts):
        normal = unreal.Vector(0.0, 1.0, 0.0) if section == len(parts) - 1 else \
            unreal.Vector(1.0, 0.0, 0.0) if section == 0 else unreal.Vector(-1.0, 0.0, 0.0)
        component.create_mesh_section_linear_color(
            section, [unreal.Vector(*v) for v in verts], tris, [normal] * len(verts),
            [unreal.Vector2D(u, v) for u, v in uvs], [], [], [], [unreal.LinearColor(1, 1, 1, 1)] * len(verts),
            [unreal.ProcMeshTangent()] * len(verts), True)
    instance = street_material(ctx, picture_path, cache_dir, night)
    if instance:
        for section in range(len(parts)):
            component.set_material(section, instance)
    component.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS)
    wb._tag_actor(actor, wb.GENERATED_TAG, label="RWB_Room" if room else "RWB_Street")
    return actor


def add_lights(camera, half, spots, brightness=1.0):
    """A coloured point light at each glowing sign.  Returns how many were made."""
    import unreal
    import world_builder as wb
    made = 0
    for number, spot in enumerate(spots):
        where = light_spot(camera, spot, half)
        if not where:
            continue
        light = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PointLight, unreal.Vector(*where))
        component = light.get_component_by_class(unreal.PointLightComponent)
        if component:
            r, g, b = [c / 255.0 for c in spot["color"]]
            component.set_light_color(unreal.LinearColor(r, g, b, 1.0))
            component.set_editor_property("intensity", 450.0 * brightness * (0.5 + float(spot.get("strength", 0.3))))
            component.set_editor_property("attenuation_radius", 900.0)
            try:
                component.set_editor_property("cast_shadows", False)
            except Exception:
                pass
        wb._tag_actor(light, wb.GENERATED_TAG, wb.LIGHT_TAG, label="RWB_Neon_%02d" % number)
        made += 1
    return made


def _linear(rgb):
    import unreal

    def lin(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return unreal.LinearColor(lin(rgb[0]), lin(rgb[1]), lin(rgb[2]), 1.0)


def street_material(ctx, picture_path, cache_dir, night=False):
    """The picture as a material (made once per build)."""
    import unreal
    import rwb_generated
    import rwb_layers
    import rwb_scene
    if ctx.get("street_material"):
        return ctx["street_material"]
    folder = ctx["folder"] + "/Street"
    texture = rwb_scene._import_texture(picture_path, folder, "T_%s_street" % ctx["psd"], str(cache_dir))
    if not texture:
        return None
    for prop in ("address_x", "address_y"):
        texture.set_editor_property(prop, unreal.TextureAddress.TA_CLAMP)
    ctx["street_material"] = rwb_layers._mic(folder, "MI_%s_street" % ctx["psd"], rwb_generated.model_material(),
                                             scalars={"SelfLit": NIGHT_SELF_LIT if night else DAY_SELF_LIT, "ArtBlend": 1.0},
                                             textures={"Art": texture})
    return ctx["street_material"]


def plain_material(ctx, name, rgb, self_lit=0.35):
    import rwb_generated
    import rwb_layers
    return rwb_layers._mic(ctx["folder"] + "/Street", name, rwb_generated.model_material(),
                           vectors={"Tint": _linear(rgb)}, scalars={"ArtBlend": 0.0, "SelfLit": self_lit})


def _section(component, index, verts, tris, uvs=None):
    import unreal
    component.create_mesh_section_linear_color(
        index, [unreal.Vector(*v) for v in verts], tris, [unreal.Vector(0.0, 0.0, 1.0)] * len(verts),
        [unreal.Vector2D(u, v) for u, v in (uvs or [(0.0, 0.0)] * len(verts))], [], [], [],
        [unreal.LinearColor(1, 1, 1, 1)] * len(verts), [unreal.ProcMeshTangent()] * len(verts), True)


def _mesh_actor(name):
    import unreal
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.Actor, unreal.Vector(0, 0, 0))
    component = unreal.new_object(unreal.ProceduralMeshComponent, outer=actor, name=name)
    actor.set_editor_property("root_component", component)
    component.set_editor_property("mobility", unreal.ComponentMobility.STATIC)
    return actor, component


def build_buildings(ctx, camera, blocks, picture_path, cache_dir, night=False):
    """One actor per building: its front painted from the picture, roof and ends in its colour."""
    import unreal
    import world_builder as wb
    art = street_material(ctx, picture_path, cache_dir, night)
    made = []
    for block in blocks:
        mesh = block_mesh(camera, block)
        tag = "%s%02d" % ("L" if block["side"] == "left" else "R", block["number"])
        actor, component = _mesh_actor("RWBBuilding")
        verts, uvs, tris = mesh["front"]
        _section(component, 0, verts, tris, uvs)
        _section(component, 1, *mesh["plain"])
        if art:
            component.set_material(0, art)
        shade = [int(c * 0.72) for c in block["color"]]
        component.set_material(1, plain_material(ctx, "MI_%s_bldg_%s" % (ctx["psd"], tag), shade))
        component.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS)
        wb._tag_actor(actor, wb.GENERATED_TAG, label="RWB_Building_%s" % tag)
        made.append(actor)
    return made


SIGN_MATERIAL = "M_RWB_SignPaint_v1"


def _build_sign_paint(material):
    """A sign's own picture, lit from inside so its lettering stays readable in any light."""
    import unreal
    import rwb_layers
    x = rwb_layers._x
    link = rwb_layers._link
    mel = unreal.MaterialEditingLibrary
    tex = x(material, unreal.MaterialExpressionTextureSampleParameter2D, -600, 0)
    tex.set_editor_property("parameter_name", "Art")
    default = (unreal.load_asset("/Engine/EngineResources/WhiteSquareTexture")
               or unreal.load_asset("/Engine/EngineResources/DefaultTexture"))
    if default:
        tex.set_editor_property("texture", default)
    mel.connect_material_property(tex, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
    glow = x(material, unreal.MaterialExpressionMultiply, -300, 200)
    link(tex, "RGB", glow, "A")
    link(rwb_layers._scalar_param(material, "Glow", 0.9, -600, 300), "", glow, "B")
    mel.connect_material_property(glow, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    rough = x(material, unreal.MaterialExpressionConstant, -300, 400)
    rough.set_editor_property("r", 0.6)
    mel.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)


def build_boards(ctx, camera, records, cache_dir, half=None, night=False):
    """Each sign as its own board model.  records: plan records whose inst carries "board".

    Returns {kind: count}.
    """
    import unreal
    import rwb_layers
    import rwb_scene
    import world_builder as wb
    master = rwb_layers._material(SIGN_MATERIAL, _build_sign_paint)
    frame = plain_material(ctx, "MI_%s_signframe" % ctx["psd"], (38, 36, 42), 0.1)
    folder = ctx["folder"] + "/Signs"
    kinds = {}
    for number, record in enumerate(records):
        board = record["inst"].get("board")
        if not board:
            continue
        mesh, kind = board_mesh(camera, board, half, None if half is not None else record["place"])
        label = "".join(ch if ch.isalnum() else "_" for ch in (record["inst"].get("label") or "sign"))[:20]
        texture = rwb_scene._import_texture(board["texture"], folder, "T_%s_sign%02d" % (ctx["psd"], number), str(cache_dir))
        actor, component = _mesh_actor("RWBSign")
        verts, uvs, tris = mesh["face"]
        _section(component, 0, verts, tris, uvs)
        _section(component, 1, *mesh["frame"])
        if texture:
            for prop in ("address_x", "address_y"):
                texture.set_editor_property(prop, unreal.TextureAddress.TA_CLAMP)
            component.set_material(0, rwb_layers._mic(folder, "MI_%s_sign%02d" % (ctx["psd"], number), master,
                                                      scalars={"Glow": 1.6 if night else 0.9},
                                                      textures={"Art": texture}))
        component.set_material(1, frame)
        component.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS)
        wb._tag_actor(actor, wb.GENERATED_TAG, label="RWB_Sign_%s_%02d" % (label, number))
        record["actor"] = actor
        kinds[kind] = kinds.get(kind, 0) + 1
    return kinds
