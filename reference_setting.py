"""What kind of place is this picture?  (Reference World Builder)

Everything after recognition depends on the answer: a city street needs buildings, a road and no
grass; a desert needs dunes and sand; a forest needs trees and meadows.  CLIP compares the whole
picture with a few descriptions of each kind of place.

Each SETTING says what the rest of the tool should do there:
* ``queries``  - extra things the detector looks for (phrase -> category)
* ``surfaces`` - which kinds of ground can exist (see reference_surfaces.SURFACES)
* ``terrain``  - how hilly the ground is (1 = the default rolling ground, 0 = flat)
* ``grass``    - whether grass may grow on grassy ground
* ``forest``   - whether extra trees fill the map outside the painted view
* ``box``      - whether the place is enclosed by walls (a street of buildings, a room): the walls are
                 built from the picture itself (reference_streetbox)
"""

from __future__ import annotations

import numpy as np

SETTINGS = {
    "nature": {
        "phrases": ["a painting of a forest", "a landscape with trees and grass", "a garden with plants",
                    "a fantasy woodland", "a meadow by a lake"],
        "queries": {}, "surfaces": ["grass", "meadow", "dirt", "sand", "rock", "water"],
        "terrain": 1.0, "grass": True, "forest": True, "box": False,
    },
    "city": {
        "phrases": ["a city street with buildings", "a street lined with shops and signs", "a busy downtown street at night",
                    "a cityscape with skyscrapers", "an alley between buildings"],
        "queries": {"a building": "building", "a skyscraper": "building", "a storefront": "building",
                    "a neon sign": "sign", "a shop sign": "sign", "a billboard": "sign", "a street lamp": "lantern",
                    "a traffic light": "pillar", "a utility pole": "pillar", "a car": "vehicle", "a bicycle": "vehicle",
                    "a scooter": "vehicle", "a vending machine": "furniture", "a trash can": "furniture"},
        "surfaces": ["road", "pavement", "grass", "water"],
        "terrain": 0.0, "grass": False, "forest": False, "box": True,
    },
    "interior": {
        "phrases": ["the inside of a room", "an interior with furniture", "a hallway inside a building",
                    "a bedroom", "a kitchen"],
        "queries": {"a chair": "furniture", "a table": "furniture", "a bed": "furniture", "a sofa": "furniture",
                    "a cabinet": "furniture", "a shelf": "furniture", "a lamp": "lantern", "a door": "gate",
                    "a window": "prop_card", "a rug": "lily", "a plant in a pot": "bush"},
        "surfaces": ["floorboards", "tiles", "carpet"],
        "terrain": 0.0, "grass": False, "forest": False, "box": True,
    },
    "desert": {
        "phrases": ["a desert with sand dunes", "a dry rocky canyon", "a sandy wasteland under the sun"],
        "queries": {"a cactus": "pine", "a dune": "mountain", "a palm tree": "tree", "a camel": "creature",
                    "a tent": "house", "a ruin": "wall"},
        "surfaces": ["sand", "dirt", "rock", "water"],
        "terrain": 2.2, "grass": False, "forest": False, "box": False,
    },
    "snow": {
        "phrases": ["a snowy landscape", "snow covered mountains", "a frozen winter forest"],
        "queries": {"a snowman": "prop_card", "an igloo": "house", "a pine tree covered in snow": "pine"},
        "surfaces": ["snow", "ice", "rock", "water"],
        "terrain": 1.8, "grass": False, "forest": True, "box": False,
    },
    "coast": {
        "phrases": ["a beach by the sea", "a seaside coast with waves", "a tropical island shore"],
        "queries": {"a palm tree": "tree", "a sailboat": "boat", "a pier": "bridge", "a beach umbrella": "prop_card",
                    "a lighthouse": "tower"},
        "surfaces": ["sand", "grass", "rock", "water"],
        "terrain": 0.8, "grass": True, "forest": False, "box": False,
    },
}
ORDER = ["nature", "city", "interior", "desert", "snow", "coast"]
MIN_CONFIDENCE = 0.6       # a weaker guess builds the default (nature): walls in the wrong place are worse
INDOOR = {"interior"}      # no sky: a dim room isn't night


def classify(rgb, clip=None):
    """{"setting", "scores" {setting: probability}} for a picture (HxWx3 0-255)."""
    if clip is None:
        import reference_catalog
        clip = reference_catalog.Clip()
    image = clip.image(rgb)
    phrases, owners = [], []
    for name in ORDER:
        for phrase in SETTINGS[name]["phrases"]:
            phrases.append(phrase)
            owners.append(name)
    sims = clip.texts(phrases) @ image * 100.0
    probs = np.exp(sims - sims.max())
    probs /= probs.sum()
    scores = {name: 0.0 for name in ORDER}
    for owner, p in zip(owners, probs):
        scores[owner] += float(p)
    best = max(scores, key=scores.get)
    unsure = scores[best] < MIN_CONFIDENCE
    return {"setting": "nature" if unsure else best, "unsure": unsure,
            "scores": {k: round(v, 3) for k, v in scores.items()}}


def get(name):
    """The settings for a kind of place (unknown names fall back to nature)."""
    return SETTINGS.get(name) or SETTINGS["nature"]


def line(result, chosen=None):
    """One line for the panel."""
    chosen = chosen or result["setting"]
    ranked = sorted(result["scores"].items(), key=lambda kv: -kv[1])[:3]
    note = " - unsure, so the default" if result.get("unsure") and chosen == result["setting"] else ""
    return "Setting: %s (%s)%s" % (chosen, ", ".join("%s %d%%" % (k, round(100 * v)) for k, v in ranked), note)


# --------------------------------------------------------------------------- #
# night and neon
# --------------------------------------------------------------------------- #
NIGHT_LUMINANCE = 0.3      # mean brightness below this, with a dark sky, is a night picture
NEON_MIN_SAT = 0.45        # neon: saturated ...
NEON_MIN_VAL = 0.75        # ... and bright
MAX_LIGHTS = 12


def light_of(rgb, horizon=None):
    """{"night", "luminance", "sky_luminance"} for a picture (HxWx3 0-255)."""
    x = np.asarray(rgb, np.float32) / 255.0
    lum = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    H = lum.shape[0]
    top = lum[: max(1, int((horizon if horizon is not None else 0.35 * H) * 0.5))]
    mean, sky = float(lum.mean()), float(top.mean())
    return {"night": bool(mean < NIGHT_LUMINANCE and sky < 0.35), "luminance": round(mean, 3),
            "sky_luminance": round(sky, 3)}


def neon_spots(rgb, count=MAX_LIGHTS, cells=24):
    """Bright, saturated patches (lit signs): [{"u", "v", "color" [r,g,b], "strength"}], strongest first.

    The picture is cut into a grid; each cell scores how much glowing colour it holds, and the best
    cells that are not next to an already chosen one become lights.
    """
    import reference_surfaces
    rgb = np.asarray(rgb)
    H, W = rgb.shape[:2]
    _hue, sat, val = reference_surfaces.hsv(rgb)
    glow = (sat >= NEON_MIN_SAT) & (val >= NEON_MIN_VAL)
    rows = max(1, int(round(cells * H / float(max(H, W)))))
    cols = max(1, int(round(cells * W / float(max(H, W)))))
    score = np.zeros((rows, cols), np.float32)
    for r in range(rows):
        for c in range(cols):
            score[r, c] = glow[r * H // rows:(r + 1) * H // rows, c * W // cols:(c + 1) * W // cols].mean()
    out = []
    taken = np.zeros_like(score, bool)
    for flat in np.argsort(-score, axis=None):
        r, c = divmod(int(flat), cols)
        if score[r, c] < 0.08 or len(out) >= count:
            break
        if taken[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any():
            continue
        taken[r, c] = True
        block = (slice(r * H // rows, (r + 1) * H // rows), slice(c * W // cols, (c + 1) * W // cols))
        lit = glow[block]
        colour = rgb[block][lit].mean(0)
        out.append({"u": round((c + 0.5) / cols, 4), "v": round((r + 0.5) / rows, 4),
                    "color": [int(v) for v in colour], "strength": round(float(score[r, c]), 3)})
    return out

