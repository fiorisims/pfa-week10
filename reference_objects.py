"""See what is in a painting and cut each object out (Reference World Builder).

Two small vision models, both CPU-only through onnxruntime:

* **OWLv2** (zero-shot detector, ~155 MB) finds objects by name, so the word list
  can be edited without retraining.  Unlike the older OWL-ViT it stays confident
  on stylised paintings (peak score 0.41 vs 0.04 on the Sleeping Beauty concept).
* **SlimSAM** (segmentation, ~40 MB) turns each box into the object's real outline.
  Those outlines give the true ground-contact point for placement and are the
  silhouette ``reference_mesh`` builds new 3D objects from.

Models download once from Hugging Face, or put the files in ``models/owlv2`` and
``models/slimsam``.  Runs in the system Python; never imports ``unreal``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402

import rwb_progress  # noqa: E402

ROOT = Path(__file__).resolve().parent
OWL_REPO = "Xenova/owlv2-base-patch16-ensemble"
OWL_MODEL = "onnx/model_quantized.onnx"
OWL_TOKENIZER = "tokenizer.json"
SAM_REPO = "Xenova/slimsam-77-uniform"
SAM_ENCODER = "onnx/vision_encoder.onnx"
SAM_DECODER = "onnx/prompt_encoder_mask_decoder.onnx"
CLIP_REPO = "Xenova/clip-vit-base-patch32"
CLIP_VISION = "onnx/vision_model_quantized.onnx"
CLIP_TEXT = "onnx/text_model_quantized.onnx"
CLIP_TOKENIZER = "tokenizer.json"
RECOGNIZER_VERSION = "rwb-objects-12"
OWL_SIZE = 960
SAM_SIZE = 1024
SCORE_MIN = 0.10          # below this OWLv2 is guessing, even on paintings
MAX_BOX_AREA = 0.65       # a box this big is the whole picture, not an object
MAX_OBJECT_AREA = 0.42    # an outline's bounding box this big is framing or background
MIN_OBJECT_AREA = 0.0006  # smaller outlines are specks of paint
PART_OF = 0.6             # this share of a smaller outline inside a bigger one = a part of it
MERGE_MIN_SCORE = 0.3     # only a confidently seen building gathers up the fragments around it
MERGE_REACH = 0.2         # a fragment's centre may be this share of the building's width beyond its sides
MERGE_TOUCH = 5           # pixels: how close counts as touching
MERGE_MAX_SIZE = 0.45     # a fragment is smaller than this share of the building
MERGE_MAX_DEPTH = 2.2     # a fragment further than this many times the building's distance is behind it
MERGE_NEVER = {"mushroom", "floating", "character", "creature", "lily", "pine", "lantern"}
MAX_OBJECTS = 30
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
SAM_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
SAM_STD = np.array([0.229, 0.224, 0.225], np.float32)

# phrase -> category.  Categories decide real-world size and how the mesh is built.
# Extend without editing code: add to "detector" in rwb_layer_keywords.json.
QUERIES = {
    "a tree": "tree", "a tree trunk": "tree", "a pine tree": "pine", "a bush": "bush",
    "thorny vines": "thorns", "flowers": "flower", "reeds": "grass", "cattails": "grass",
    "a rock": "rock", "a boulder": "rock", "a cliff": "cliff", "a log": "log", "a tree stump": "log",
    "a mushroom": "mushroom", "a boat": "boat", "a canoe": "boat", "a lily pad": "lily",
    "a castle": "castle", "a tower": "tower", "a gazebo": "tower", "a greenhouse": "house",
    "a house": "house", "a cottage": "house", "a stone wall": "wall", "a fence": "fence",
    "a bridge": "bridge", "an archway": "gate", "a stone column": "pillar", "a lamp post": "lantern",
    "a lantern": "lantern", "a bench": "furniture", "a barrel": "furniture", "a bucket": "furniture",
    "a birdcage": "furniture", "a ladder": "furniture", "a fishing rod": "prop_card",
    "a statue": "prop_card", "a signpost": "prop_card", "a crystal": "floating",
    "a person": "character", "a bird": "creature", "an animal": "creature", "a butterfly": "floating",
    "a pond": "water", "a river": "water", "a lake": "water", "a waterfall": "water",
    "a mountain": "mountain",
}
# area kinds shape the terrain instead of becoming objects
AREA_CATEGORIES = {"water", "mountain"}
# these lie on the ground or water, so they cannot be above the horizon
GROUNDED_CATEGORIES = {"boat", "lily", "bridge"}
# these may hang or fly above the ground; anything else above the horizon is canopy or backdrop
AIRBORNE_CATEGORIES = {"floating", "creature", "lantern", "prop_card", "sign"}
AIRBORNE_MAX_AREA = 0.03  # share of the picture: a floater is small
HORIZON_SLACK = 0.08      # standing objects may meet the ground this far above the horizon (hidden feet)


def _resize(a, w, h):
    import reference_scene
    return reference_scene._resize(a, w, h)


def _file(repo, name, local_dir):
    local = ROOT / "models" / local_dir / Path(name).name
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo, filename=name))


def _session(path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def load_queries(keywords_path=None):
    """Built-in phrases plus any the artist added under "detector" in the vocabulary."""
    table = dict(QUERIES)
    if keywords_path and Path(keywords_path).is_file():
        try:
            data = json.loads(Path(keywords_path).read_text(encoding="utf-8-sig"))
            for phrase, category in (data.get("detector") or {}).items():
                table[str(phrase).strip().lower()] = str(category).strip().lower()
        except Exception:
            pass
    return table


def short_label(phrase):
    for article in ("a ", "an "):
        if phrase.startswith(article):
            return phrase[len(article):]
    return phrase


# --------------------------------------------------------------------------- #
# boxes
# --------------------------------------------------------------------------- #
def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _contains(big, small):
    ix = max(0.0, min(big[2], small[2]) - max(big[0], small[0]))
    iy = max(0.0, min(big[3], small[3]) - max(big[1], small[1]))
    area = max(1e-6, (small[2] - small[0]) * (small[3] - small[1]))
    return ix * iy / area > 0.85


def filter_boxes(scores, boxes, phrases, table, W, H, side, score_min=SCORE_MIN):
    """Turn raw OWLv2 output into one labelled box per object.

    scores: patches x queries (0-1).  boxes: patches x 4, cx/cy/w/h as fractions of
    the padded square of edge ``side``.  Returns [{label, category, score, box}].
    """
    best_q = scores.argmax(1)
    best = scores.max(1)
    cands = []
    for p in np.nonzero(best >= score_min)[0]:
        cx, cy, w, h = [float(v) * side for v in boxes[p]]
        box = [max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(float(W), cx + w / 2), min(float(H), cy + h / 2)]
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw < 4 or bh < 4 or bw * bh > MAX_BOX_AREA * W * H:
            continue
        phrase = phrases[int(best_q[p])]
        cands.append({"label": short_label(phrase), "category": table[phrase], "phrase": phrase,
                      "score": float(best[p]), "box": box, "owl": scores[p]})
    cands.sort(key=lambda c: -c["score"])
    kept = []
    for c in cands:
        if any((k["category"] == c["category"] and iou(k["box"], c["box"]) > 0.35) or iou(k["box"], c["box"]) > 0.6
               for k in kept):
            continue
        kept.append(c)
    # a box holding two or more of its own type is a "group" box, not one object
    out = []
    for c in kept:
        inner = [k for k in kept if k is not c and k["category"] == c["category"] and _contains(c["box"], k["box"])]
        if len(inner) < 2:
            out.append(c)
    return out


def _owl_run(rgb, ids, mask, session):
    """Raw OWLv2 on one image: (scores, boxes as fractions of the padded square, square edge)."""
    H, W = rgb.shape[:2]
    side = max(H, W)
    pad = np.full((side, side, 3), 127.5, np.float32)         # OWLv2 expects a grey-padded square
    pad[:H, :W] = rgb
    x = _resize(pad / 255.0, OWL_SIZE, OWL_SIZE)
    x = ((x - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)[None].astype(np.float32)
    out = session.run(None, {"pixel_values": x, "input_ids": ids, "attention_mask": mask})
    names = [o.name for o in session.get_outputs()]
    logits = np.asarray(out[names.index("logits")][0], np.float32)
    boxes = np.asarray(out[names.index("pred_boxes")][0], np.float32)
    return 1.0 / (1.0 + np.exp(-logits)), boxes, side


def tile_boxes(W, H, grid=2, overlap=0.25):
    """Overlapping crops [x0, y0, x1, y1] covering the picture, for finding smaller objects."""
    tw, th = W / (grid - (grid - 1) * overlap), H / (grid - (grid - 1) * overlap)
    tiles = []
    for gy in range(grid):
        for gx in range(grid):
            x0, y0 = gx * tw * (1 - overlap), gy * th * (1 - overlap)
            tiles.append([int(x0), int(y0), int(min(W, x0 + tw)), int(min(H, y0 + th))])
    return tiles


def detect(rgb, table, session=None, score_min=SCORE_MIN, tiles=True):
    """OWLv2 over the whole image and (optionally) overlapping tiles: one box per object."""
    from tokenizers import Tokenizer
    phrases = list(table)
    tok = Tokenizer.from_file(str(_file(OWL_REPO, OWL_TOKENIZER, "owlv2")))
    tok.no_padding()
    tok.no_truncation()
    ids = np.zeros((len(phrases), 16), np.int64)
    mask = np.zeros_like(ids)
    for i, phrase in enumerate(phrases):
        encoded = list(tok.encode(phrase).ids)[:16]
        ids[i, :len(encoded)] = encoded
        mask[i, :len(encoded)] = 1
    session = session or _session(_file(OWL_REPO, OWL_MODEL, "owlv2"))
    H, W = rgb.shape[:2]
    scores, boxes, side = _owl_run(rgb, ids, mask, session)
    all_scores, all_boxes = [scores], [boxes * side]
    for x0, y0, x1, y1 in (tile_boxes(W, H) if tiles else []):
        tile_scores, tile_box, tile_side = _owl_run(rgb[y0:y1, x0:x1], ids, mask, session)
        # a box cut off by the tile edge is only part of an object: leave those to other views
        px = tile_box * tile_side
        left, right = px[:, 0] - px[:, 2] / 2, px[:, 0] + px[:, 2] / 2
        top, bottom = px[:, 1] - px[:, 3] / 2, px[:, 1] + px[:, 3] / 2
        margin = 0.01 * tile_side
        whole = ((left > margin) | (x0 == 0)) & ((right < x1 - x0 - margin) | (x1 >= W)) \
            & ((top > margin) | (y0 == 0)) & ((bottom < y1 - y0 - margin) | (y1 >= H))
        px[:, 0] += x0
        px[:, 1] += y0
        all_scores.append(tile_scores[whole])
        all_boxes.append(px[whole])
    # filter_boxes works in fractions of a square; use the whole picture's
    dets = filter_boxes(np.concatenate(all_scores), np.concatenate(all_boxes) / side,
                        phrases, table, W, H, side, score_min)
    for d in dets:
        d["phrases"] = phrases
    return dets


def classify_crop(rgb, table, session=None):
    """Best (label, category, score) for a single object's picture, or None.

    Used for PSD layers whose names don't say what they are.
    """
    dets = detect(rgb, table, session=session, score_min=0.06, tiles=False)
    dets = [d for d in dets if d["category"] not in AREA_CATEGORIES]
    if not dets:
        return None
    top = max(dets, key=lambda d: d["score"])
    return top["label"], top["category"], top["score"]


# --------------------------------------------------------------------------- #
# outlines
# --------------------------------------------------------------------------- #
class Segmenter:
    """SlimSAM with one image encoded once and any number of box prompts."""

    def __init__(self, rgb):
        self.encoder = _session(_file(SAM_REPO, SAM_ENCODER, "slimsam"))
        self.decoder = _session(_file(SAM_REPO, SAM_DECODER, "slimsam"))
        self.H, self.W = rgb.shape[:2]
        self.scale = SAM_SIZE / float(max(self.H, self.W))
        self.nw = max(1, int(round(self.W * self.scale)))
        self.nh = max(1, int(round(self.H * self.scale)))
        small = _resize(rgb.astype(np.float32) / 255.0, self.nw, self.nh)
        canvas = np.zeros((SAM_SIZE, SAM_SIZE, 3), np.float32)
        canvas[:self.nh, :self.nw] = (small - SAM_MEAN) / SAM_STD
        out = self.encoder.run(None, {"pixel_values": canvas.transpose(2, 0, 1)[None]})
        names = [o.name for o in self.encoder.get_outputs()]
        self.embeddings = out[names.index("image_embeddings")]
        self.positional = out[names.index("image_positional_embeddings")]

    def _decode(self, points, labels):
        out = self.decoder.run(None, {
            "input_points": np.array([[points]], np.float32),
            "input_labels": np.array([[labels]], np.int64),
            "image_embeddings": self.embeddings, "image_positional_embeddings": self.positional,
        })
        names = [o.name for o in self.decoder.get_outputs()]
        return np.asarray(out[names.index("pred_masks")])[0, 0], np.asarray(out[names.index("iou_scores")])[0, 0]

    def box(self, box):
        """(bool HxW mask, predicted quality) for a box in image pixels."""
        x0, y0, x1, y1 = [float(v) * self.scale for v in box]
        corners = [[x0, y0], [x1, y1]]
        prompts = [
            (corners, [2, 3]),                                       # 2 / 3 = box corners in SAM's encoding
            (corners + [[(x0 + x1) / 2.0, (y0 + y1) / 2.0]], [2, 3, 1]),   # 1 = "this point is on the object"
        ]
        # SAM offers whole / part / sub-part.  Its own quality score alone can prefer a crisp
        # sliver at the edge of the box, so also ask how much of the box the outline explains.
        best, best_rank, best_quality = None, -1.0, 0.0
        for points, labels in prompts:
            masks, quality = self._decode(points, labels)
            size = masks.shape[-1]
            to_map = size / float(SAM_SIZE)
            prompt = [x0 * to_map, y0 * to_map, x1 * to_map, y1 * to_map]
            box_area = max(1.0, (prompt[2] - prompt[0]) * (prompt[3] - prompt[1]))
            for i in range(masks.shape[0]):
                on = masks[i] > 0.0
                ys, xs = np.nonzero(on)
                if not len(xs):
                    continue
                fit = iou(prompt, [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
                inside = on[int(prompt[1]):int(prompt[3]) + 1, int(prompt[0]):int(prompt[2]) + 1].sum()
                fill = min(1.0, float(inside) / box_area / 0.5)     # a solid object fills half its box or more
                rank = float(quality[i]) * (0.1 + fit) * (0.25 + fill)
                if rank > best_rank:
                    best, best_rank, best_quality = masks[i], rank, float(quality[i])
        cut_w = int(round(self.nw * best.shape[-1] / float(SAM_SIZE)))
        cut_h = int(round(self.nh * best.shape[-2] / float(SAM_SIZE)))
        logits = best[:max(1, cut_h), :max(1, cut_w)]
        return _resize(logits, self.W, self.H) > 0.0, best_quality


    def points(self, positive, negative=()):
        """[(bool HxW mask, quality)] for every mask SAM offers for point prompts (image pixels)."""
        pts = [[x * self.scale, y * self.scale] for x, y in list(positive) + list(negative)]
        labels = [1] * len(positive) + [0] * len(negative)          # 0 = "this point is not on it"
        masks, quality = self._decode(pts, labels)
        out = []
        for i in range(masks.shape[0]):
            cut_w = int(round(self.nw * masks.shape[-1] / float(SAM_SIZE)))
            cut_h = int(round(self.nh * masks.shape[-2] / float(SAM_SIZE)))
            logits = masks[i][:max(1, cut_h), :max(1, cut_w)]
            out.append((_resize(logits, self.W, self.H) > 0.0, float(quality[i])))
        return out


FLOOR_VERSION = "rwb-floor-5"


def floor_rank(mask, quality, horizon):
    """How much a mask looks like the floor: it covers the bottom of the picture and stays below the horizon."""
    H = mask.shape[0]
    if not mask.any():
        return 0.0
    bottom = float(mask[int(0.93 * H):].mean())
    above = float(mask[:max(1, int(horizon - 0.03 * H))].sum()) / float(mask.sum())
    return quality * bottom * max(0.0, 1.0 - 2.5 * above)


def close_columns(floor, rgb=None, max_gap=0.25, min_run=6, max_colour=0.22):
    """Seen from a standing eye the floor runs unbroken up each column of the picture until it ends, so a
    gap with floor above and below it is floor too (a lit patch, a road marking) - if its paint matches
    the floor on either side of it.  A shop front between the sidewalk and a sliver of floor doesn't.
    Objects standing there still hide it later.
    """
    H, W = floor.shape
    out = floor.copy()
    limit = int(max_gap * H)
    x = None if rgb is None else np.asarray(rgb, np.float32) / 255.0
    for c in range(W):
        rows = np.nonzero(floor[:, c])[0]
        if len(rows) < 2 * min_run:
            continue
        for g in np.nonzero(np.diff(rows) > 1)[0]:
            top, bottom = rows[g], rows[g + 1]
            above = np.sum((rows <= top) & (rows > top - min_run * 2))
            if bottom - top - 1 > limit or above < min_run:
                continue
            if x is not None:
                edge = np.concatenate([x[max(0, top - 4):top + 1, c], x[bottom:bottom + 5, c]])
                gap = x[top + 1:bottom, c]
                if np.linalg.norm(np.median(gap, 0) - np.median(edge, 0)) > max_colour:
                    continue
            out[top:bottom, c] = True
    return out


def grow_floor(rgb, floor, allowed, groups=6, owned=0.55, step=2):
    """SAM outlines the near floor well and stops early further back.  Colour groups that are mostly floor
    where SAM saw it are floor wherever they connect to it (below the horizon).
    """
    import reference_scene
    import reference_surfaces
    if floor.sum() < 50:
        return floor
    zone = allowed
    centres, labels = reference_surfaces.group_colours(rgb[zone], groups=groups)
    label_map = -np.ones(floor.shape, np.int64)
    label_map[zone] = labels
    floorish = np.zeros(floor.shape, bool)
    for i in range(len(centres)):
        members = label_map == i
        if members.any() and float((members & floor).sum()) / float(members.sum()) >= owned:
            floorish |= members
    candidates = (floorish | floor)[::step, ::step]
    seeds = floor[::step, ::step]
    out = floor.copy()
    for cells in reference_scene._components(candidates):
        rows, cols = zip(*cells)
        if seeds[list(rows), list(cols)].any():
            grown = np.zeros_like(candidates)
            grown[list(rows), list(cols)] = True
            out |= np.repeat(np.repeat(grown, step, 0), step, 1)[:floor.shape[0], :floor.shape[1]] & zone
    return out


def floor_mask(rgb, horizon, cache_dir=None, stamp="", segmenter=None):
    """The floor (ground, street, sidewalk, room floor) as SAM outlines it, or None.

    Prompted with points along the bottom of the picture and "not this" points on the sky and the
    upper walls.  The floor may be in pieces (a road and its sidewalk), so each bottom point also asks
    on its own, and good pieces are joined.
    """
    H, W = rgb.shape[:2]
    cache = Path(cache_dir).resolve() / "floor.npy" if cache_dir else None
    key = "%s|%s|%d" % (FLOOR_VERSION, stamp, int(horizon))
    if cache is not None and cache.is_file() and cache.with_suffix(".json").is_file():
        try:
            if json.loads(cache.with_suffix(".json").read_text(encoding="utf-8")).get("key") == key:
                return np.load(cache)
        except Exception:
            pass
    seg = segmenter or Segmenter(rgb)
    bottom = [(W * f, H * 0.965) for f in (0.12, 0.5, 0.88)]
    lift = max(0.0, horizon - 0.14 * H)
    negative = [(W * 0.5, H * 0.04), (W * 0.06, lift), (W * 0.94, lift)]
    best, best_rank = None, 0.0
    for mask, quality in seg.points(bottom, negative):
        rank = floor_rank(mask, quality, horizon)
        if rank > best_rank:
            best, best_rank = mask, rank
    if best is None:
        return None
    out = best.copy()
    for point in bottom:
        for mask, quality in seg.points([point], negative):
            above = float(mask[:max(1, int(horizon - 0.03 * H))].sum()) / max(1.0, float(mask.sum()))
            if quality >= 0.8 and above < 0.08 and mask[int(0.9 * H):].any():
                out |= mask
    below = np.zeros_like(out)
    below[int(max(0, horizon - 0.02 * H)):] = True
    out = close_columns(grow_floor(rgb, fill_holes(out & below), below), rgb)
    if cache is not None:
        np.save(cache, out)
        cache.with_suffix(".json").write_text(json.dumps({"key": key, "share": float(out.mean())}), encoding="utf-8")
    return out


def fill_holes(mask, max_share=0.06):
    """Close small gaps inside an outline (windows, gaps between leaves) so a mesh has no tunnels.

    A big enclosed region is left open: it is usually another object seen through this one.
    """
    import reference_scene
    h, w = mask.shape
    out = mask.copy()
    limit = max(4.0, mask.sum() * max_share)
    for cells in reference_scene._components(~mask):
        if len(cells) > limit:
            continue
        rows, cols = zip(*cells)
        if min(rows) == 0 or min(cols) == 0 or max(rows) == h - 1 or max(cols) == w - 1:
            continue                                    # open to the outside: not a hole
        out[list(rows), list(cols)] = True
    return out


def clean_mask(mask, box=None, pad=0.08):
    """Keep the mask inside its box and drop specks.  Returns a bool mask (maybe empty)."""
    import reference_scene
    out = mask.copy()
    if box is not None:
        H, W = mask.shape
        x0, y0, x1, y1 = box
        mx, my = (x1 - x0) * pad, (y1 - y0) * pad
        keep = np.zeros_like(out)
        keep[int(max(0, y0 - my)):int(min(H, y1 + my + 1)), int(max(0, x0 - mx)):int(min(W, x1 + mx + 1))] = True
        out &= keep
    if not out.any():
        return out
    step = max(1, int(max(out.shape) / 320))
    small = out[::step, ::step]
    comps = reference_scene._components(small)
    largest = max(len(c) for c in comps)
    remove = [c for c in comps if len(c) < largest * 0.04]
    for comp in remove:
        for r, c in comp:
            out[r * step:(r + 1) * step, c * step:(c + 1) * step] = False
    if box is not None:
        # work on the box only: flood-filling a whole image per object would be slow
        ys, xs = np.nonzero(out)
        if len(xs):
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            step = max(1, int(max(y1 - y0, x1 - x0) / 300))
            crop = out[y0:y1, x0:x1]
            filled = fill_holes(np.pad(crop[::step, ::step], 1))[1:-1, 1:-1]
            if step > 1:
                filled = np.repeat(np.repeat(filled, step, 0), step, 1)[:y1 - y0, :x1 - x0]
            out[y0:y1, x0:x1] = crop | filled
    return out


def describe_mask(mask):
    """Where a mask sits: bounding box, ground-contact point and coverage."""
    ys, xs = np.nonzero(mask)
    H, W = mask.shape
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    # contact = the middle of the lowest ~6% of the shape: where it meets the ground
    band = ys >= y1 - max(1.0, (y1 - y0) * 0.06)
    return {"bbox": [x0, y0, x1, y1], "bbox_frac": [x0 / float(W), y0 / float(H), x1 / float(W), y1 / float(H)],
            "contact_u": float(np.median(xs[band])) / W, "contact_v": y1 / float(H),
            "top_v": y0 / float(H), "width_frac": (x1 - x0) / float(W), "area": float(mask.mean())}


def fill_holes_in_box(mask):
    """Fill small holes, working only on the shape's box (the seams left by merging pieces)."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return mask
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    out = mask.copy()
    out[y0:y1, x0:x1] = fill_holes(np.pad(mask[y0:y1, x0:x1], 1))[1:-1, 1:-1]
    return out


def smooth_outline(mask, radius=None):
    """Round off jagged, spurred edges.  Thin things (rods, posts) are left alone: they'd vanish."""
    import reference_strokes as rs
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return mask
    size = max(xs.max() - xs.min(), ys.max() - ys.min())
    radius = radius or max(1, int(size * 0.008))
    smooth = rs.blur(mask.astype(np.float32), radius) > 0.5
    return smooth if smooth.sum() >= 0.75 * mask.sum() else mask


def roughness(mask):
    """1 = a perfect circle; higher = a more tangled, less definite outline."""
    edges = (mask[:-1, :] != mask[1:, :]).sum() + (mask[:, :-1] != mask[:, 1:]).sum()
    area = max(1, int(mask.sum()))
    return float(edges) ** 2 / (4.0 * np.pi * area)


def sensible_label(label, category, place):
    """A label the shape contradicts.  (A stump can't fill half the picture's height.)"""
    x0, y0, x1, y1 = place["bbox_frac"]
    if category == "log" and (y1 - y0) > 0.3 and (y1 - y0) > (x1 - x0):
        return "tree", "tree"
    return label, category


def merge_structures(items):
    """Buildings get read as many fragments (arches, roots, steps, each with its own label).

    Fold every fragment touching a building into it, so it becomes one model at one distance.
    items: dicts with mask, category, place.  Mutates and returns the kept items.
    """
    from reference_mesh import ARCHITECTURE
    structures = sorted((d for d in items if d["category"] in ARCHITECTURE and d["score"] >= MERGE_MIN_SCORE),
                        key=lambda d: -int(d["mask"].sum()))
    for structure in structures[:1 + len(structures) // 2]:
        if structure.get("merged_away"):
            continue
        # the reach stays the building's ORIGINAL footprint: if each piece widened it, a chain of
        # neighbours could swallow half the picture
        x0, y0, x1, y1 = structure["place"]["bbox"]
        reach = (x1 - x0) * MERGE_REACH
        own_area = int(structure["mask"].sum())
        changed = True
        while changed:
            changed = False
            grown = _grow(structure["mask"], MERGE_TOUCH)
            for other in items:
                if other is structure or other.get("merged_away") or other["category"] in MERGE_NEVER:
                    continue
                ox0, oy0, ox1, oy1 = other["place"]["bbox"]
                centre = (ox0 + ox1) / 2.0
                # beside or below the building, touching it, and smaller than it
                if not (x0 - reach <= centre <= x1 + reach) or (oy0 + oy1) / 2.0 < y0 + (y1 - y0) * 0.4:
                    continue
                if _far_behind(other, structure):
                    continue
                if other["mask"].sum() > own_area * MERGE_MAX_SIZE or not (grown & other["mask"]).any():
                    continue
                structure["mask"] = structure["mask"] | other["mask"]
                structure["place"] = describe_mask(structure["mask"])
                structure.setdefault("parts", []).append(other["label"])
                other["merged_away"] = True
                changed = True
                break
    return [d for d in items if not d.get("merged_away")]


def _far_behind(other, structure):
    """Touching in the picture doesn't mean touching in the world: ask the depth map."""
    if "depth_eye" not in other or not structure.get("depth_eye"):
        return False
    return other["depth_eye"] is None or other["depth_eye"] > structure["depth_eye"] * MERGE_MAX_DEPTH


def _grow(mask, radius):
    import reference_strokes as rs
    return rs.blur(mask.astype(np.float32), radius) > 0.02


def relative_depth(disp, mask, fit, horizon, focal):
    """Distance in camera heights from the depth map, or None without a usable fit."""
    if disp is None or not fit:
        return None
    a = fit[0] * focal
    b = fit[0] * horizon + fit[1]
    if a <= 0:
        return None
    inverse = (float(np.median(disp[mask])) - b) / a
    return 1.0 / inverse if inverse > 1e-3 else None


# --------------------------------------------------------------------------- #
# second opinion on each outline
# --------------------------------------------------------------------------- #
def _on_grey(rgb, mask):
    """The masked object on neutral grey, square, ready for CLIP."""
    import pygame
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = int(max(x1 - x0, y1 - y0) * 1.1) + 1
    canvas = np.full((side, side, 3), 128, np.uint8)
    ox, oy = (side - (x1 - x0)) // 2, (side - (y1 - y0)) // 2
    crop = rgb[y0:y1, x0:x1]
    keep = mask[y0:y1, x0:x1]
    canvas[oy:oy + (y1 - y0), ox:ox + (x1 - x0)][keep] = crop[keep]
    surf = pygame.image.frombuffer(canvas.tobytes(), (side, side), "RGB")
    surf = pygame.transform.smoothscale(surf, (224, 224))
    return np.frombuffer(pygame.image.tobytes(surf, "RGB"), np.uint8).reshape(224, 224, 3).astype(np.float32) / 255.0


def clip_opinions(rgb, masks, phrases):
    """objects x phrases similarity with each word's own bias removed, as probabilities."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(_file(CLIP_REPO, CLIP_TOKENIZER, "clip")))
    tok.no_padding()
    tok.no_truncation()
    ids = np.zeros((len(phrases), 77), np.int64)
    for i, phrase in enumerate(phrases):
        encoded = list(tok.encode("a painting of " + phrase).ids)[:77]
        ids[i, :len(encoded)] = encoded
    text = _session(_file(CLIP_REPO, CLIP_TEXT, "clip")).run(None, {"input_ids": ids})[0]
    text = text / np.linalg.norm(text, axis=-1, keepdims=True)
    vision = _session(_file(CLIP_REPO, CLIP_VISION, "clip"))
    images = []
    for mask in masks:
        x = (_on_grey(rgb, mask) - CLIP_MEAN) / CLIP_STD
        emb = vision.run(None, {"pixel_values": x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        images.append(emb / np.linalg.norm(emb))
    sims = np.stack(images) @ text.T * 100.0
    # thin shapes all look like "cattails" to CLIP: subtract what each word scores on everything
    if len(masks) >= 4:
        sims = sims - sims.mean(0, keepdims=True)
    probs = np.exp(sims - sims.max(1, keepdims=True))
    return probs / probs.sum(1, keepdims=True)


def fused_label(owl, clip, phrases, table, above_horizon):
    """Best phrase by both models; things that lie on the ground can't be above the horizon."""
    # OWLv2 saw the object in context; CLIP only breaks ties and fixes clear mistakes
    blended = np.maximum(owl, 1e-6) ** 0.75 * np.maximum(clip, 1e-6) ** 0.25
    for i in np.argsort(-blended):
        phrase = phrases[int(i)]
        category = table[phrase]
        if category in AREA_CATEGORIES or (above_horizon and category in GROUNDED_CATEGORIES):
            continue
        return phrase, float(blended[int(i)])
    return None, 0.0


def is_scenery(place, category, horizon_frac):
    """True for outlines that are the setting rather than a thing standing in it."""
    x0, y0, x1, y1 = place["bbox_frac"]
    edge = 0.012
    touches = sum((x0 < edge, y0 < edge, x1 > 1 - edge, y1 > 1 - edge))
    wide = x1 - x0
    # the ground itself: runs off the bottom of the frame and is wider than it is tall
    if y1 > 1 - edge and wide > 0.3 and (y1 - y0) < wide and category not in GROUNDED_CATEGORIES:
        return True
    # a big dark mass running off two sides, unless it is a tree framing the shot
    if touches >= 2 and category not in ("tree", "pine"):
        return True
    if horizon_frac is not None and place["contact_v"] < horizon_frac - HORIZON_SLACK:
        # its lowest point is in the sky: canopy and backdrop, unless it is small and may hang or fly
        if category not in AIRBORNE_CATEGORIES or place["area"] > AIRBORNE_MAX_AREA:
            return True
    return False


def drop_parts(items):
    """Keep whole objects: an outline mostly inside a bigger one is a part of it.

    items: dicts with a bool "mask", in any order; returns them without the parts.
    """
    ordered = sorted(items, key=lambda d: -int(d["mask"].sum()))
    kept = []
    for item in ordered:
        area = int(item["mask"].sum())
        if any(int((item["mask"] & big["mask"]).sum()) > PART_OF * area for big in kept):
            continue
        kept.append(item)
    return kept


# --------------------------------------------------------------------------- #
# cut-outs
# --------------------------------------------------------------------------- #
def save_cutout(texture, mask, bbox, path, work_size):
    """Crop the object from the full-resolution painting with the mask as alpha."""
    import pygame
    TH, TW = texture.shape[:2]
    H, W = work_size
    sx, sy = TW / float(W), TH / float(H)
    x0, y0, x1, y1 = int(bbox[0] * sx), int(bbox[1] * sy), int(np.ceil(bbox[2] * sx)), int(np.ceil(bbox[3] * sy))
    x1, y1 = max(x0 + 2, min(TW, x1)), max(y0 + 2, min(TH, y1))
    alpha = _resize(mask.astype(np.float32), TW, TH)[y0:y1, x0:x1]
    crop = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    crop[..., :3] = texture[y0:y1, x0:x1, :3]
    crop[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    surf = pygame.image.frombuffer(crop.tobytes(), (x1 - x0, y1 - y0), "RGBA")
    pygame.image.save(surf, str(path))
    return str(path)


def recognize(rgb, texture, cache_dir, stamp, keywords_path=None, disp=None, fit=None, horizon=None, focal=None,
              extra_queries=None):
    """Detect, outline and cut out everything in a painting (cached).

    Returns (objects, areas, info).  ``objects`` carry their cut-out and placement data;
    ``areas`` carry a mask file for terrain-shaping things (water, mountains).
    """
    table = load_queries(keywords_path)
    # what this kind of place holds (reference_setting): a street has buildings and cars
    for phrase, category in (extra_queries or {}).items():
        table.setdefault(phrase, category)
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = "%s|%s|%s" % (RECOGNIZER_VERSION, stamp, hashlib.md5(json.dumps(table, sort_keys=True).encode()).hexdigest())
    cache = cache_dir / "recognized.json"
    if cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("key") == key and all(Path(o["cutout"]).is_file() for o in data["objects"]):
                return data["objects"], data["areas"], data["info"]
        except Exception:
            pass
    for old in list(cache_dir.glob("cut_*.png")) + list(cache_dir.glob("area_*.npy")):
        old.unlink()
    rwb_progress.report(0.3, "Looking for objects")
    dets = detect(rgb, table)
    H, W = rgb.shape[:2]
    objects, areas, candidates = [], [], []
    if dets:
        rwb_progress.report(0.5, "Outlining %d possible objects" % len(dets))
        segmenter = Segmenter(rgb)
        dets.sort(key=lambda d: -d["score"])
        for number, d in enumerate(dets):
            rwb_progress.report(0.5 + 0.3 * number / float(len(dets)), "Outlining %d of %d" % (number + 1, len(dets)))
            mask, quality = segmenter.box(d["box"])
            mask = clean_mask(mask, d["box"])
            if mask.mean() < MIN_OBJECT_AREA:
                continue
            d["quality"] = round(quality, 3)
            d["mask"] = mask
            if d["category"] in AREA_CATEGORIES:
                name = cache_dir / ("area_%02d.npy" % len(areas))
                np.save(name, mask)
                areas.append({"category": d["category"], "label": d["label"], "score": round(d["score"], 3),
                              "mask": str(name), "box": [round(v, 1) for v in d["box"]]})
                continue
            place = describe_mask(mask)
            bx0, by0, bx1, by1 = place["bbox"]
            if (bx1 - bx0) * (by1 - by0) > MAX_OBJECT_AREA * W * H:
                continue
            d["place"] = place
            candidates.append(d)
    candidates = drop_parts(candidates)
    if disp is not None and fit:
        for d in candidates:
            d["depth_eye"] = relative_depth(disp, d["mask"], fit, horizon, focal)
    candidates = merge_structures(candidates)
    for d in candidates:
        d["mask"] = smooth_outline(fill_holes_in_box(d["mask"]))
        d["place"] = describe_mask(d["mask"])
    candidates.sort(key=lambda d: -d["score"])
    candidates = candidates[:MAX_OBJECTS]
    info = "OWLv2 + SlimSAM"
    if candidates:
        rwb_progress.report(0.82, "Double-checking what each object is")
        phrases = candidates[0]["phrases"]
        try:
            opinions = clip_opinions(rgb, [d["mask"] for d in candidates], phrases)
            info += " + CLIP"
        except Exception:
            opinions = None                               # keep the detector's own labels
        for i, d in enumerate(candidates):
            d["detected"] = d["category"]
            above = horizon is not None and d["place"]["contact_v"] * H < horizon
            if opinions is not None:
                phrase, _blended = fused_label(d["owl"], opinions[i], phrases, table, above)
                if phrase:
                    d["label"], d["category"] = short_label(phrase), table[phrase]
            d["label"], d["category"] = sensible_label(d["label"], d["category"], d["place"])
        horizon_frac = None if horizon is None else horizon / float(H)
        # the detector saw the whole scene: a relabel from the close-up can't make something scenery
        candidates = [d for d in candidates if not (is_scenery(d["place"], d["category"], horizon_frac)
                                                    and is_scenery(d["place"], d.get("detected", d["category"]), horizon_frac))]
        for d in candidates:
            mask = d["mask"]
            place = d["place"]
            box_area = max(1.0, (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
            cut = save_cutout(texture, mask, place["bbox"], cache_dir / ("cut_%02d.png" % len(objects)), (H, W))
            objects.append({
                "centre_u": (place["bbox"][0] + place["bbox"][2]) / 2.0 / W,
                "label": d["label"], "category": d["category"], "score": round(d["score"], 3),
                "quality": d["quality"], "coverage": round(float(mask.sum() / box_area), 3),
                "roughness": round(roughness(mask), 2), "parts": d.get("parts", []),
                "box": [round(v, 1) for v in d["box"]], "cutout": cut,
                "depth_eye": relative_depth(disp, mask, fit, horizon, focal), **place,
            })
    cache.write_text(json.dumps({"key": key, "objects": objects, "areas": areas, "info": info}), encoding="utf-8")
    return objects, areas, info
