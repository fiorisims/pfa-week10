"""Where things go: the painting's camera, solved so the world matches the picture.

Pure math with no ``unreal`` import, so it can be tested in any Python.

The player starts where the painter's eye was, looking into the picture.  For an object
whose base touches the ground at picture row ``r``, a pinhole camera at height ``eye``
above flat ground with its horizon at row ``hr`` says::

    distance = eye * focal / (r - hr)
    size in the world = size in pixels * distance / focal

So lower in the picture = closer, the same pixel height means bigger when further away,
and from the start position every object covers the part of the frame it was painted in.

Coordinates are Unreal's: +X = right of the picture, -Y = into the picture, +Z = up.
"""

from __future__ import annotations

import math

NEAREST_CM = 260.0            # nothing closer than this to the start
FAR_SHARE = 0.85              # share of the map used in front of the player
MAX_GROUND_ASPECT = 2.6       # a lying object at most this much longer than it is wide
# hidden feet: when the depth map says an object is this much nearer than where its visible
# bottom would put it, something is covering its base and the depth map is right
OCCLUDED_RATIO = 1.6
HIDDEN_BEHIND = 1.1           # a thing with hidden feet stands at least this much further than what hides them
FLAT_CATEGORIES = {"boat", "lily", "bridge"}  # modelled lying down (reference_mesh.GROUND)
HORIZON_MARGIN = 0.012        # of the picture height: closer to the horizon than this is "at the far edge"


class PaintCamera:
    def __init__(self, width, height, horizon, fov, eye, world_size):
        self.W = float(width)
        self.H = float(height)
        self.horizon = float(horizon)
        self.fov = max(20.0, min(120.0, float(fov)))
        self.focal = (self.W / 2.0) / math.tan(math.radians(self.fov) / 2.0)
        self.eye = float(eye)
        self.size = float(world_size)
        self.player_y = self.size / 2.0 - 700.0
        self.far = min(self.size * FAR_SHARE, self.player_y + self.size / 2.0 - 300.0)
        self.limit = self.size / 2.0 - 300.0

    # ---- picture <-> ground --------------------------------------------------
    def row_distance(self, row):
        """Distance to the ground seen at a picture row, clamped to the map."""
        below = float(row) - self.horizon
        if below <= HORIZON_MARGIN * self.H:
            return self.far
        return max(NEAREST_CM, min(self.far, self.eye * self.focal / below))

    def to_world(self, u, v):
        """Ground position (x, y) of a point in the picture (u, v = 0..1, top-left origin)."""
        distance = self.row_distance(v * self.H)
        x = (u - 0.5) * self.W * distance / self.focal
        return max(-self.limit, min(self.limit, x)), self.player_y - distance

    def to_image(self, x, y):
        """Picture (u, v) that sees a ground position, or None if it is out of the frame."""
        distance = self.player_y - y
        if distance <= 1.0:
            return None
        u = 0.5 + x * self.focal / (distance * self.W)
        v = (self.horizon + self.eye * self.focal / distance) / self.H
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return None
        return u, v

    def in_view(self, x, y, margin=0.0):
        """True when a ground spot is inside the picture's wedge (margin in cm)."""
        distance = self.player_y - y
        if distance <= 0.0:
            return False
        half = (self.W / 2.0) * distance / self.focal
        return abs(x) <= half + margin

    # ---- objects -------------------------------------------------------------
    def place(self, inst, category="", airborne=("floating", "creature", "lantern", "prop_card", "drawn", "sign")):
        """Everything a builder needs for an object seen in the picture.

        inst keys (fractions of the picture): contact_u, contact_v, top_v, width_frac,
        centre_u (optional), depth_eye (optional distance from the depth map in camera
        heights), band_t (optional FG/MG/BG override, 0 = near .. 1 = far).
        Returns {x, y, z, distance, width, height, yaw, airborne}; z is the object's base
        above the ground under it.
        """
        contact_row = float(inst["contact_v"]) * self.H
        top_row = float(inst.get("top_v", inst["contact_v"])) * self.H
        depth_eye = inst.get("depth_eye")
        depth_cm = float(depth_eye) * self.eye if depth_eye else None
        above = contact_row < self.horizon - 0.05 * self.H
        is_airborne = above and category in airborne
        if inst.get("band_t") is not None:
            # the artist said how far: FG/MG/BG tags beat anything the picture implies
            t = max(0.0, min(1.0, float(inst["band_t"])))
            distance = NEAREST_CM + (self.far - NEAREST_CM) * t ** 1.5
            is_airborne = False
        elif is_airborne:
            distance = depth_cm if depth_cm else self.row_distance(self.H * 0.9)
        else:
            distance = self.row_distance(contact_row)
            near = None
            if inst.get("feet_hidden") and inst.get("behind_v") is not None:
                near = self.row_distance(float(inst["behind_v"]) * self.H) * HIDDEN_BEHIND
            if near is not None and near < distance:
                # its foot is hidden behind something nearer (reference_layering): it stands behind that
                # thing, and no further than its visible bottom says; the depth map picks the spot between
                distance = min(distance, max(near, depth_cm)) if depth_cm else math.sqrt(near * distance)
            elif depth_cm and depth_cm * OCCLUDED_RATIO < distance:
                distance = depth_cm
        distance = max(NEAREST_CM, min(self.far, distance))
        per_pixel = distance / self.focal
        width = float(inst.get("width_frac", 0.1)) * self.W * per_pixel
        if is_airborne:
            z = self.eye + (self.horizon - contact_row) * per_pixel
            height = (contact_row - top_row) * per_pixel
        elif inst.get("band_t") is not None:
            z = 0.0
            height = (contact_row - top_row) * per_pixel
        else:
            z = 0.0
            # the top's height above the ground, so hidden feet can't shrink the object
            height = self.eye + (self.horizon - top_row) * per_pixel
            ground_row = self.horizon + self.eye * self.focal / distance
            if contact_row > ground_row:
                height = max(height, (contact_row - top_row) * per_pixel)
        height = max(8.0, height)
        centre_u = inst.get("centre_u", inst.get("contact_u", 0.5))
        x = (float(centre_u) - 0.5) * self.W * per_pixel
        x = max(-self.limit, min(self.limit, x))
        return {
            "x": x, "y": self.player_y - distance, "z": max(0.0, z), "distance": distance,
            "width": max(4.0, width), "height": height,
            # turn to face the start position, so the silhouette stays the one that was painted
            "yaw": math.degrees(math.atan2(x, distance)),
            "airborne": is_airborne,
        }

    def lies_flat(self, inst):
        """A shape that crosses the horizon is standing up, whatever its label says.

        reference_mesh.build_object applies the same rule when it chooses how to model it.
        """
        box = inst.get("bbox_px")
        return bool(box) and box[1] >= self.horizon + 0.03 * self.H

    def place_ground(self, inst):
        """Placement for a lying object: position from its outline's centre on the ground.

        Returns {x, y, z, distance, scale_xy, scale_y, airborne}.  ``scale_xy`` converts the
        mesh's camera-height units to cm; ``scale_y`` shortens objects that perspective has
        made absurdly long (a row near the horizon stretches enormously on the ground).
        """
        x0, y0, x1, y1 = inst["bbox_px"]
        near = self.row_distance(y1)
        far = self.row_distance(y0)
        centre_u = (x0 + x1) / 2.0 / self.W
        wide = max(1.0, (x1 - x0)) * ((near + far) / 2.0) / self.focal
        long_now = max(1.0, far - near)
        squash = min(1.0, MAX_GROUND_ASPECT * wide / long_now)
        length = long_now * squash
        centre_d = near + length / 2.0
        x = (centre_u - 0.5) * self.W * centre_d / self.focal
        return {
            "x": max(-self.limit, min(self.limit, x)), "y": self.player_y - centre_d, "z": 0.0,
            "distance": centre_d, "scale_xy": self.eye, "scale_y": squash, "airborne": False,
            "width": wide, "height": max(wide * 0.16, 10.0),
        }


def footprint(place):
    """Radius used for spacing objects apart."""
    return max(20.0, place.get("width", 100.0) * 0.5)


def scatter_allowed(camera, x, y, margin=350.0):
    """Extra copies must stay out of the painted view, so it still matches the picture."""
    return not camera.in_view(x, y, margin)
