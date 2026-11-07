"""Is that really water?  (Reference World Builder)

Water you can wade into changes the whole terrain, so a wrong guess spoils a world: blue-lit pavement, a
neon-lit road or a blue wall all look like water when their paint is judged on its own.  Every water
candidate (a detector pond, a bluish colour group on the ground, the colour fallback) is checked here,
with CLIP looking at the candidate **in its surroundings** and at the whole picture:

* in context, a lake reads as a lake (0.8-0.9 on the test paintings); a blue-lit sidewalk reads as a
  sidewalk (0.1-0.2), because the kerb, shop fronts and people around it are in the crop;
* the whole picture is asked "is there water here?" as a second opinion (it can be fooled by line
  sketches, so it never decides alone);
* water is coloured: grey paint (sketches, line art) is never water.
"""

from __future__ import annotations

import numpy as np

WET = ["a lake", "a pond", "a river", "the sea", "a swimming pool", "a stream"]
DRY_SCENES = ["a city street", "a sidewalk", "an indoor floor", "a grassy field", "a dirt path", "a desert",
              "a snowy field", "a road at night", "a garden", "a forest floor", "a room"]
# what a candidate could be instead, per kind of ground (reference_surfaces names)
DRY_GROUND = {"road": "an asphalt road", "pavement": "a sidewalk", "grass": "grass", "meadow": "a meadow",
              "dirt": "bare dirt", "sand": "sand", "rock": "rocks", "snow": "snow", "ice": "ice",
              "floorboards": "a wooden floor", "tiles": "a tiled floor", "carpet": "a carpet"}
LOOKALIKES = ["a wet street reflecting lights", "a shiny floor", "a blue wall", "a glass window", "a blue sky"]
CONTEXT_PAD = 0.15          # of the picture's long side, kept around a candidate so CLIP sees where it is
SURE = 0.5                  # in context, this sure is enough
LIKELY = 0.35               # ... this sure, if the whole picture agrees this much
BUILT_UP_EXTRA = 0.1        # streets and rooms need more proof: their shiny floors look wet
MIN_SATURATION = 0.12       # water has colour; grey paint (a sketch) isn't water
MIN_AREA = 0.004            # of the picture

_clip = {}


def _model():
    if "clip" not in _clip:
        import reference_catalog
        _clip["clip"] = reference_catalog.Clip()
    return _clip["clip"]


def _softmax(sims):
    p = np.exp(sims - sims.max())
    return p / p.sum()


def crop_around(rgb, mask, pad=CONTEXT_PAD):
    """The candidate's bounding box plus its surroundings."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    p = int(pad * max(H, W))
    return rgb[max(0, ys.min() - p):min(H, ys.max() + p + 1), max(0, xs.min() - p):min(W, xs.max() + p + 1)]


def region_probability(rgb, mask, allowed=(), clip=None):
    """How sure CLIP is that this part of the picture, seen in its surroundings, is water (0..1)."""
    clip = clip or _model()
    dry = [DRY_GROUND[n] for n in allowed if n in DRY_GROUND] or list(DRY_GROUND.values())
    phrases = ["a picture of %s" % p for p in WET] + ["a picture of %s" % p for p in dry + LOOKALIKES]
    probs = _softmax(clip.texts(phrases) @ clip.image(crop_around(rgb, mask)) * 100.0)
    return float(probs[:len(WET)].sum())


def scene_probability(rgb, clip=None):
    """How sure CLIP is that the picture has water in it at all (0..1)."""
    clip = clip or _model()
    phrases = ["a picture with %s" % p for p in WET] + ["a picture of %s with no water" % p for p in DRY_SCENES]
    probs = _softmax(clip.texts(phrases) @ clip.image(rgb) * 100.0)
    return float(probs[:len(WET)].sum())


def saturation(rgb, mask):
    x = np.asarray(rgb, np.float32)[mask] / 255.0
    if not len(x):
        return 0.0
    high, low = x.max(-1), x.min(-1)
    return float(np.mean(np.where(high > 0, (high - low) / np.maximum(high, 1e-6), 0.0)))


def decide(region, scene, sat, area, built_up=False):
    """(keep, reason) from the evidence.  Pure: the rules, without the models."""
    if area < MIN_AREA:
        return False, "too small to be a pond"
    if sat < MIN_SATURATION:
        return False, "grey paint, not water"
    extra = BUILT_UP_EXTRA if built_up else 0.0
    if region >= SURE + extra:
        return True, "reads as water in its surroundings"
    if region >= LIKELY + extra and scene >= LIKELY:
        return True, "looks like water, and the picture has water in it"
    return False, "reads as dry ground in its surroundings"


def confirm(rgb, mask, allowed=(), setting="nature", scene=None, clip=None):
    """Check one water candidate.  Returns a JSON-friendly verdict:
    {"keep", "reason", "region", "scene", "saturation", "area"}.  ``scene``: a precomputed scene_probability.
    """
    mask = np.asarray(mask, bool)
    area = float(mask.mean()) if mask.size else 0.0
    sat = saturation(rgb, mask) if area > 0 else 0.0
    region = 0.0
    if area >= MIN_AREA and sat >= MIN_SATURATION:
        try:
            region = region_probability(rgb, mask, allowed, clip)
            if scene is None:
                scene = scene_probability(rgb, clip)
        except Exception as error:                     # no CLIP: only an unmistakable pond counts
            return {"keep": False, "reason": "could not check (%s)" % str(error)[:80], "region": 0.0,
                    "scene": 0.0, "saturation": round(sat, 3), "area": round(area, 4)}
    keep, reason = decide(region, scene or 0.0, sat, area, built_up=setting in ("city", "interior"))
    return {"keep": keep, "reason": reason, "region": round(region, 3), "scene": round(scene or 0.0, 3),
            "saturation": round(sat, 3), "area": round(area, 4)}


def line(verdicts):
    """One line for the panel about the water."""
    kept = [v for v in verdicts if v["keep"]]
    if kept:
        v = max(kept, key=lambda v: v["region"])
        return "Water: yes (%s, %d%% sure) - you can wade into it" % (v.get("source", "found"), round(100 * v["region"]))
    if verdicts:
        v = max(verdicts, key=lambda v: v["region"])
        return "Water: none (a %s looked wet, but %s)" % (v.get("source", "patch"), v["reason"])
    return "Water: none"
