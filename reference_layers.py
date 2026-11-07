"""Read a layered Photoshop file one layer at a time for the Reference World Builder.

Each visible layer (and layer group) is exported by Photoshop as a full-canvas
PNG, so its position in the painting is preserved.  For every layer we then:

* work out WHAT it is from its name (``rwb_layer_keywords.json`` - editable),
* find WHERE it is and HOW BIG it is (the bounding box of its pixels, split
  into separate pieces when one layer holds several trees or rocks),
* measure its COLOURS (overall, top, bottom, darkest, most saturated),
* crop its artwork to a texture so Unreal can paint it onto the object,
* and, for ground / hills / water / paths, record a coverage grid so the
  terrain can follow the painted shapes.

Runs in the normal system Python (pygame).  Never imports ``unreal``.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import pygame

WORK_SIZE = 384          # long side of the working copy used for shape analysis
TEXTURE_SIZE = 1024      # long side of the exported artwork crops
ALPHA_ON = 96            # alpha (0-255) that counts as "painted"
MAX_LAYERS = 160
CACHE_VERSION = "rwb-layers-7"

# --------------------------------------------------------------------------- #
# name -> object vocabulary
# --------------------------------------------------------------------------- #
DEFAULT_KEYWORDS = {
    "_help": [
        "Reference World Builder layer vocabulary.",
        "Each category lists the words that identify it in a Photoshop layer or group name.",
        "Add your own words (lower case, singular) and re-analyze.  The LAST recognised word",
        "in a name wins, so 'Castle tower' is a tower and 'Tree shadow' is ignored.",
        "Force a category by putting it in brackets or before a colon: '[rock] big thing', 'tower: west'.",
        "Distance tags (depth): FG = right in front of the player, MG = middle, BG = far away.",
        "'Trees BG' is still trees, just placed far off. A group named 'BG' passes it to every layer inside.",
        "kind: object = 3D model, card = painted standee, scatter = many small plants,",
        "ground/hill/water/path = shapes the terrain, sky/fog/light = atmosphere, ignore = paint only.",
    ],
    "categories": {
        "tree": {"kind": "object", "words": [
            "tree", "oak", "maple", "birch", "elm", "willow", "beech", "apple", "forest", "wood", "woods",
            "grove", "orchard", "canopy", "sapling", "palm", "foliage", "leaf", "treeline", "jungle"]},
        "pine": {"kind": "object", "words": [
            "pine", "fir", "spruce", "evergreen", "conifer", "cedar", "cypress", "redwood", "sequoia"]},
        "bush": {"kind": "object", "words": [
            "bush", "shrub", "hedge", "hedgerow", "topiary", "thicket", "rosebush", "rose bush", "berry"]},
        "thorns": {"kind": "object", "words": [
            "thorn", "briar", "brier", "bramble", "vine", "ivy", "creeper", "overgrowth", "thornbush",
            "thorn wall", "thorns wall", "briar wall", "bramble wall", "wall of thorns", "thorn hedge"]},
        "flower": {"kind": "scatter", "words": [
            "flower", "rose", "tulip", "daisy", "lily", "poppy", "blossom", "bloom", "petal", "sunflower",
            "lavender", "wildflower", "bouquet", "dandelion"]},
        "grass": {"kind": "scatter", "words": [
            "grass", "weed", "reed", "fern", "plant", "clover", "tallgrass", "tall grass", "undergrowth", "cattail"]},
        "rock": {"kind": "object", "words": [
            "rock", "stone", "boulder", "pebble", "rubble", "crystal", "gem"]},
        "cliff": {"kind": "object", "words": [
            "cliff", "crag", "ledge", "bluff", "outcrop", "rockface", "plateau", "mesa"]},
        "log": {"kind": "object", "words": [
            "log", "stump", "trunk", "branch", "driftwood", "timber", "fallen tree", "tree trunk"]},
        "castle": {"kind": "object", "words": [
            "castle", "palace", "fortress", "fort", "keep", "citadel", "stronghold", "chateau", "kingdom"]},
        "tower": {"kind": "object", "words": ["gazebo", "pavilion", "rotunda", 
            "tower", "turret", "spire", "steeple", "belfry", "lighthouse", "minaret", "watchtower"]},
        "house": {"kind": "object", "words": [
            "house", "cottage", "hut", "cabin", "home", "shack", "barn", "building", "church", "chapel",
            "temple", "shop", "inn", "tavern", "mill", "windmill", "village", "town", "manor", "hall", "castle wing"]},
        "wall": {"kind": "object", "words": [
            "wall", "rampart", "battlement", "parapet", "ruin", "ruins"]},
        "fence": {"kind": "object", "words": [
            "fence", "railing", "rail", "picket", "balustrade"]},
        "bridge": {"kind": "object", "words": [
            "bridge", "boardwalk", "dock", "pier", "drawbridge"]},
        "gate": {"kind": "object", "words": [
            "gate", "arch", "archway", "door", "doorway", "portal", "entrance", "trellis"]},
        "pillar": {"kind": "object", "words": [
            "pillar", "column", "post", "obelisk", "pole", "totem", "monument"]},
        "lantern": {"kind": "object", "words": [
            "lantern", "lamp", "torch", "candle", "streetlight", "lamppost", "brazier", "campfire", "bonfire", "fire"]},
        "furniture": {"kind": "object", "words": [
            "bed", "table", "chair", "throne", "bench", "chest", "crate", "barrel", "box", "cart", "wagon",
            "well", "fountain", "altar", "desk", "shelf", "cabinet"]},
        "character": {"kind": "card", "words": [
            "character", "person", "people", "princess", "prince", "king", "queen", "girl", "boy", "man",
            "woman", "child", "kid", "baby", "fairy", "witch", "sorceress", "villain", "hero", "knight",
            "guard", "soldier", "figure", "crowd", "aurora", "phillip", "philip", "maleficent", "merryweather",
            "flora", "fauna", "lady", "maiden", "human", "servant", "goon"]},
        "creature": {"kind": "card", "words": [
            "animal", "bird", "owl", "raven", "crow", "deer", "horse", "rabbit", "bunny", "squirrel", "dragon",
            "cat", "dog", "wolf", "bear", "fox", "fish", "butterfly", "creature", "monster", "diablo"]},
        "prop_card": {"kind": "card", "words": ["signpost", "sign post", 
            "spindle", "spinning wheel", "mushroom", "sign", "banner", "flag", "statue", "sculpture",
            "stained glass", "window", "painting", "tapestry", "crown", "sword", "shield"]},
        "mushroom": {"kind": "object", "words": ["mushroom", "toadstool", "fungus", "fungi", "shroom"]},
        "boat": {"kind": "object", "words": ["boat", "canoe", "rowboat", "raft", "ship", "gondola", "kayak"]},
        "lily": {"kind": "object", "words": ["lily pad", "lilypad", "water lily", "lily large", "lily small", "lotus"]},
        "mountain": {"kind": "hill", "words": [
            "mountain", "hill", "peak", "ridge", "mound", "dune", "slope", "highland", "volcano", "hillside"]},
        "water": {"kind": "water", "words": ["waterfall", "pool", "fountain pool", 
            "water", "lake", "river", "pond", "sea", "ocean", "stream", "creek", "brook", "moat",
            "waterfall", "puddle", "lagoon", "bay", "wave"]},
        "path": {"kind": "path", "words": [
            "path", "road", "trail", "street", "lane", "pathway", "walkway", "step", "stair", "stairs",
            "track", "cobblestone", "cobble"]},
        "ground": {"kind": "ground", "words": [
            "ground", "field", "meadow", "floor", "land", "terrain", "earth", "dirt", "soil", "lawn",
            "grassland", "plain", "valley", "moss", "snow", "sand", "beach", "shore", "clearing"]},
        "sky": {"kind": "sky", "words": [
            "sky", "cloud", "sun", "moon", "star", "sunset", "sunrise", "horizon", "heaven", "northern lights"]},
        "fog": {"kind": "fog", "words": [
            "fog", "mist", "haze", "smoke", "atmosphere", "atmos", "steam"]},
        "light": {"kind": "light", "words": [
            "light", "lighting", "glow", "ray", "sunbeam", "beam", "godray", "god rays", "shine",
            "sparkle", "magic", "rim light"]},
    },
    "depth": {
        "fg": ["fg", "foreground", "front", "near", "close", "closeup", "forefront"],
        "mg": ["mg", "midground", "middleground", "mid", "middle"],
        "bg": ["bg", "background", "back", "far", "distant", "distance", "faraway"],
    },
    "library_ignore": ["paddle", "ground", "path", "platform", "tile", "floor", "edge", "corner", "side", "slab", "block"],
    "weak": {
        "sky": ["background", "bg", "backdrop", "distance", "distant"],
        "ground": ["foreground", "fg", "midground", "mg"],
    },
    "ignore": [
        "sketch", "line", "lineart", "line art", "linework", "outline", "ink", "ref", "reference", "guide",
        "note", "text", "title", "signature", "watermark", "shadow", "shade", "shading", "highlight",
        "texture", "overlay", "multiply", "screen", "adjustment", "adjust", "curve", "level", "gradient",
        "flat", "flats", "color", "colour", "value", "grain", "noise", "filter", "ao", "occlusion",
        "vignette", "border", "frame", "mask", "paper", "canvas", "blur", "render", "reflection",
    ],
}

# categories whose separate painted pieces become separate objects
SPLIT_CATEGORIES = {"tree", "pine", "bush", "thorns", "rock", "cliff", "log", "lantern", "pillar"}
# categories whose wide pieces are split into a row of objects (a tree line)
ROW_CATEGORIES = {"tree", "pine", "bush", "thorns", "rock"}
GENERIC_NAME = re.compile(r"^\s*(layer|group|copy|untitled|new layer)?\s*\d*(\s*copy)?\s*\d*\s*$", re.I)


def load_keywords(path):
    """Load (and, the first time, create) the editable vocabulary file."""
    path = Path(path)
    if not path.is_file():
        path.write_text(json.dumps(DEFAULT_KEYWORDS, indent=2), encoding="utf-8")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        data = DEFAULT_KEYWORDS
    words, phrases = {}, []
    for category, spec in (data.get("categories") or {}).items():
        kind = spec.get("kind", "object")
        for word in spec.get("words", []):
            word = str(word).strip().lower()
            if not word:
                continue
            if " " in word:
                phrases.append((tuple(word.split()), category, kind))
            else:
                words[word] = (category, kind)
    for word in data.get("ignore", []):
        word = str(word).strip().lower()
        if " " in word:
            phrases.append((tuple(word.split()), "ignore", "ignore"))
        elif word:
            words.setdefault(word, ("ignore", "ignore"))
    weak = {}
    for category, items in (data.get("weak") or {}).items():
        kind = ((data.get("categories") or {}).get(category) or {}).get("kind", category)
        for word in items:
            weak[str(word).strip().lower()] = (category, kind)
    depth = {}
    for band, items in (data.get("depth") or DEFAULT_KEYWORDS["depth"]).items():
        for word in items:
            depth[str(word).strip().lower()] = band
    return {"words": words, "phrases": phrases, "weak": weak, "depth": depth}


def _tokens(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text).lower()
    text = re.sub(r"\bmiddle[\s_-]*ground\b", " mg ", text)     # so 'ground' is not read as terrain
    text = re.sub(r"\bfore[\s_-]*ground\b", " fg ", text)
    text = re.sub(r"\bback[\s_-]*ground\b", " bg ", text)
    return re.findall(r"[a-z]+", text)


def depth_tag(name, vocab):
    """'fg' / 'mg' / 'bg' from a layer or group name (last distance word wins), else None."""
    found = None
    for token in _tokens(name):
        if token in vocab["depth"]:
            found = vocab["depth"][token]
    return found


def _variants(token):
    yield token
    if token.endswith("ies") and len(token) > 4:
        yield token[:-3] + "y"
    if token.endswith("ves") and len(token) > 4:
        yield token[:-3] + "f"
    if token.endswith("es") and len(token) > 3:
        yield token[:-2]
    if token.endswith("s") and len(token) > 3:
        yield token[:-1]


def _lookup(token, table):
    for variant in _variants(token):
        if variant in table:
            return table[variant], variant != token or token.endswith("s")
    return None, False


def classify(name, vocab):
    """Return {category, kind, word, plural, reason} for a layer name."""
    raw = str(name or "")
    forced = re.search(r"\[([^\]]+)\]", raw) or re.match(r"^\s*([A-Za-z][A-Za-z ]*?)\s*[:=]", raw)
    if forced:
        result = classify(forced.group(1), vocab)
        if result["category"] not in ("unknown",):
            result["reason"] = "forced by '%s'" % forced.group(0).strip()
            return result
    if GENERIC_NAME.match(raw):
        # "Layer 3" says nothing about the art.  Not "ignore": a group named "Group 1" would then
        # hide every layer inside it.  The inventory looks at the pixels of these instead.
        return {"category": "unknown", "kind": "card", "word": "", "plural": False, "generic": True,
                "reason": "generic name - recognised from its pixels"}
    tokens = _tokens(raw)
    matches = []                                  # (position, size, category, kind, word, plural)
    for size in (4, 3, 2):
        for start in range(0, max(0, len(tokens) - size + 1)):
            chunk = tuple(tokens[start:start + size])
            for phrase, category, kind in vocab["phrases"]:
                if len(phrase) == size and all(p in set(_variants(t)) for p, t in zip(phrase, chunk)):
                    matches.append((start + size - 1, size, category, kind, " ".join(phrase), False))
    for index, token in enumerate(tokens):
        if token in vocab["depth"]:
            continue
        hit, plural = _lookup(token, vocab["words"])
        if hit:
            matches.append((index, 1, hit[0], hit[1], token, plural))
    if matches:
        matches.sort(key=lambda item: (item[0], item[1]))     # last word wins; phrases beat single words
        _position, _size, category, kind, word, plural = matches[-1]
        return {"category": category, "kind": kind, "word": word, "plural": plural,
                "reason": "name word '%s'" % word}
    for token in reversed(tokens):
        hit, _plural = _lookup(token, vocab["weak"])
        if hit:
            return {"category": hit[0], "kind": hit[1], "word": token, "plural": False,
                    "reason": "location word '%s'" % token, "weak": True}
    return {"category": "unknown", "kind": "card", "word": "", "plural": False,
            "reason": "name not recognised - shown as a painted card"}


# --------------------------------------------------------------------------- #
# Photoshop export
# --------------------------------------------------------------------------- #
EXPORT_JSX = r"""
app.displayDialogs = DialogModes.NO;
var inFile = new File(__SRC__);
var outDir = new Folder(__OUT__);
if (!outDir.exists) outDir.create();
var oldUnits = app.preferences.rulerUnits;
app.preferences.rulerUnits = Units.PIXELS;
var doc = app.open(inFile);
var manifest = new File(outDir.fsName + "/layers.tsv");
manifest.encoding = "UTF-8";
manifest.open("w");
var count = 0;
var limit = __LIMIT__;
var pngOpts = new PNGSaveOptions();
pngOpts.interlaced = false;
function cleanName(s) { return String(s).replace(/[\t\r\n]+/g, " "); }
function safeName(s) {
    var t = String(s).replace(/[^A-Za-z0-9_-]+/g, "_").substr(0, 28);
    return t === "" ? "layer" : t;
}
function exportItem(item, id) {
    var W = doc.width.value, H = doc.height.value, res = doc.resolution.value;
    var fname = id + "_" + safeName(item.name) + ".png";
    var tmp = app.documents.add(W, H, res, "rwb_layer_tmp", NewDocumentMode.RGB, DocumentFill.TRANSPARENT);
    app.activeDocument = doc;
    var dup = item.duplicate(tmp, ElementPlacement.PLACEATBEGINNING);
    app.activeDocument = tmp;
    try { dup.visible = true; } catch (ev) {}
    tmp.saveAs(new File(outDir.fsName + "/" + fname), pngOpts, true, Extension.LOWERCASE);
    tmp.close(SaveOptions.DONOTSAVECHANGES);
    app.activeDocument = doc;
    return fname;
}
function pixelKind(item) {
    try {
        var k = item.kind;
        return k == LayerKind.NORMAL || k == LayerKind.SMARTOBJECT || k == LayerKind.SOLIDFILL ||
               k == LayerKind.GRADIENTFILL || k == LayerKind.PATTERNFILL;
    } catch (ek) { return true; }
}
function walk(layers, parentPath, depth) {
    for (var i = 0; i < layers.length && count < limit; i++) {
        var item = layers[i];
        var isSet = item.typename === "LayerSet";
        try { if (!item.visible) continue; } catch (e1) {}
        try { if (item.opacity <= 0) continue; } catch (e2) {}
        if (!isSet && !pixelKind(item)) continue;
        count++;
        var id = ("000" + count).slice(-3);
        var fname = "";
        try { fname = exportItem(item, id); } catch (ex) { fname = ""; }
        var opacity = 100;
        try { opacity = Math.round(item.opacity); } catch (e3) {}
        manifest.writeln([id, fname, isSet ? "group" : "layer", depth, cleanName(parentPath),
                          cleanName(item.name), opacity].join("\t"));
        if (isSet && depth < 5) walk(item.layers, parentPath + "/" + cleanName(item.name), depth + 1);
    }
}
try {
    try { if (doc.mode !== DocumentMode.RGB) doc.changeMode(ChangeMode.RGB); } catch (e0) {}
    try {
        if (doc.bitsPerChannel !== BitsPerChannelType.EIGHT) doc.bitsPerChannel = BitsPerChannelType.EIGHT;
    } catch (e4) {}
    doc.saveAs(new File(outDir.fsName + "/composite.png"), pngOpts, true, Extension.LOWERCASE);
    walk(doc.layers, "", 0);
    __result__ = "exported " + count + " layer(s)";
} catch (e) {
    __result__ = "ERROR " + e;
} finally {
    try { manifest.close(); } catch (em) {}
    try { doc.close(SaveOptions.DONOTSAVECHANGES); } catch (ec) {}
    app.preferences.rulerUnits = oldUnits;
}
"""


def export_layers(psd_path, out_dir):
    """Export composite + every visible layer/group through Photoshop (cached on disk)."""
    psd_path = Path(psd_path).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    st = psd_path.stat()
    stamp = "%s|%d|%d" % (psd_path, st.st_size, int(st.st_mtime))
    stamp_file = out_dir / "export_stamp.txt"
    manifest = out_dir / "layers.tsv"
    composite = out_dir / "composite.png"
    fresh = (stamp_file.is_file() and stamp_file.read_text(encoding="utf-8") == stamp
             and manifest.is_file() and composite.is_file())
    if not fresh:
        for old in list(out_dir.glob("*.png")) + [manifest, stamp_file]:
            try:
                old.unlink()
            except FileNotFoundError:
                pass
        import photoshop_bridge
        script = (EXPORT_JSX.replace("__SRC__", json.dumps(str(psd_path).replace("\\", "/")))
                  .replace("__OUT__", json.dumps(str(out_dir).replace("\\", "/")))
                  .replace("__LIMIT__", str(MAX_LAYERS)))
        reply = photoshop_bridge.run(script)
        if not manifest.is_file() or not composite.is_file():
            line = next((ln for ln in reply.splitlines() if "__result__" in ln or "ERROR" in ln), reply)
            raise RuntimeError("Photoshop could not export the layers: %s" % line.strip()[:200])
        stamp_file.write_text(stamp, encoding="utf-8")
    rows = []
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        lid, fname, typ, depth, parent, name, opacity = parts[:7]
        rows.append({
            "id": lid, "file": str(out_dir / fname) if fname else "", "group": typ == "group",
            "depth": int(depth or 0), "parent": parent, "name": name,
            "opacity": float(opacity or 100) / 100.0,
        })
    return rows, composite


# --------------------------------------------------------------------------- #
# pixel analysis
# --------------------------------------------------------------------------- #
def _hex(color):
    return "#%02X%02X%02X" % tuple(max(0, min(255, int(round(c)))) for c in color[:3])


def _lum(color):
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


def _sat(color):
    high, low = max(color[:3]), min(color[:3])
    return 0.0 if high <= 0 else (high - low) / float(high)


def _mean(colors, fallback=(128, 128, 128)):
    if not colors:
        return fallback
    n = float(len(colors))
    return (sum(c[0] for c in colors) / n, sum(c[1] for c in colors) / n, sum(c[2] for c in colors) / n)


def _kmeans(colors, k=4, rounds=8):
    if not colors:
        return []
    step = max(1, len(colors) // 700)
    pts = colors[::step]
    k = max(1, min(k, len(pts)))
    pts_sorted = sorted(pts, key=_lum)
    centres = [pts_sorted[int((i + 0.5) * len(pts_sorted) / k)] for i in range(k)]
    groups = [[] for _ in range(k)]
    for _ in range(rounds):
        groups = [[] for _ in range(k)]
        for p in pts:
            best = min(range(k), key=lambda i: (p[0] - centres[i][0]) ** 2 + (p[1] - centres[i][1]) ** 2
                       + (p[2] - centres[i][2]) ** 2)
            groups[best].append(p)
        centres = [_mean(g, centres[i]) for i, g in enumerate(groups)]
    out = [(len(g), centres[i]) for i, g in enumerate(groups) if g]
    out.sort(key=lambda item: -item[0])
    return out


def _colour_profile(pixels, box):
    """pixels: list of (x, y, (r,g,b)) in work coordinates inside box (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = box
    height = max(1, y1 - y0)
    every = [c for _x, _y, c in pixels]
    top = [c for _x, y, c in pixels if y < y0 + height * 0.55]
    bottom = [c for _x, y, c in pixels if y >= y0 + height * 0.75]
    clusters = _kmeans(every, 4)
    dominant = clusters[0][1] if clusters else _mean(every)
    accent = max((c for _n, c in clusters), key=lambda c: _sat(c) * (0.35 + _lum(c) / 255.0), default=dominant)
    dark = min((c for _n, c in clusters), key=_lum, default=dominant)
    return {
        "avg": _hex(_mean(every)),
        "dominant": _hex(dominant),
        "accent": _hex(accent),
        "dark": _hex(dark),
        "top": _hex(_mean(top, dominant)),
        "bottom": _hex(_mean(bottom, dark)),
        "palette": [_hex(c) for _n, c in clusters],
    }


def _dilate(mask, w, h, r=2):
    horiz = bytearray(w * h)
    for y in range(h):
        row = y * w
        for x in range(w):
            if any(mask[row + max(0, x - r):row + min(w, x + r + 1)]):
                horiz[row + x] = 1
    out = bytearray(w * h)
    for y in range(h):
        for x in range(w):
            for yy in range(max(0, y - r), min(h, y + r + 1)):
                if horiz[yy * w + x]:
                    out[y * w + x] = 1
                    break
    return out


def _components(mask, w, h):
    grown = _dilate(mask, w, h, 1)
    labels = [0] * (w * h)
    comps = []
    current = 0
    for start in range(w * h):
        if not grown[start] or labels[start]:
            continue
        current += 1
        stack = [start]
        labels[start] = current
        members = []
        while stack:
            i = stack.pop()
            if mask[i]:
                members.append(i)
            x, y = i % w, i // w
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
                           (x - 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1)):
                if 0 <= nx < w and 0 <= ny < h:
                    j = ny * w + nx
                    if grown[j] and not labels[j]:
                        labels[j] = current
                        stack.append(j)
        if members:
            comps.append(members)
    return comps


def _box(indices, w):
    xs = [i % w for i in indices]
    ys = [i // w for i in indices]
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def _save_crop(surface, box_full, out_path, pad_ratio=0.01):
    fw, fh = surface.get_size()
    x0, y0, x1, y1 = box_full
    pad = int(max(x1 - x0, y1 - y0) * pad_ratio)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(fw, x1 + pad), min(fh, y1 + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None, box_full
    crop = surface.subsurface(pygame.Rect(x0, y0, x1 - x0, y1 - y0)).copy()
    scale = min(1.0, TEXTURE_SIZE / float(max(crop.get_size())))
    if scale < 1.0:
        crop = pygame.transform.smoothscale(
            crop, (max(2, int(crop.get_width() * scale)), max(2, int(crop.get_height() * scale))))
    pygame.image.save(crop, str(out_path))
    return str(out_path), (x0, y0, x1, y1)


def _grids(mask, data, w, h, size):
    coverage = [[0.0] * size for _ in range(size)]
    colors = [[[0, 0, 0] for _ in range(size)] for _ in range(size)]
    for gy in range(size):
        y0, y1 = int(gy * h / size), max(int(gy * h / size) + 1, int((gy + 1) * h / size))
        for gx in range(size):
            x0, x1 = int(gx * w / size), max(int(gx * w / size) + 1, int((gx + 1) * w / size))
            hits, acc = 0, [0, 0, 0]
            total = 0
            for y in range(y0, min(y1, h)):
                for x in range(x0, min(x1, w)):
                    total += 1
                    i = y * w + x
                    if mask[i]:
                        hits += 1
                        acc[0] += data[i * 4]
                        acc[1] += data[i * 4 + 1]
                        acc[2] += data[i * 4 + 2]
            coverage[gy][gx] = round(hits / float(max(1, total)), 3)
            if hits:
                colors[gy][gx] = [acc[0] // hits, acc[1] // hits, acc[2] // hits]
    return coverage, colors


def analyze_layer(row, info, out_dir, grid, image_w, image_h):
    """Measure one exported layer.  Returns None when the layer is empty."""
    path = row.get("file")
    if not path or not Path(path).is_file():
        return None
    surface = pygame.image.load(path)
    fw, fh = surface.get_size()
    scale = min(1.0, WORK_SIZE / float(max(fw, fh)))
    ww, wh = max(1, int(round(fw * scale))), max(1, int(round(fh * scale)))
    small = pygame.transform.smoothscale(surface, (ww, wh)) if scale < 1.0 else surface
    data = pygame.image.tobytes(small, "RGBA")
    mask = bytearray(ww * wh)
    for i in range(ww * wh):
        if data[i * 4 + 3] >= ALPHA_ON:
            mask[i] = 1
    painted = sum(mask)
    if painted < 3:
        return None
    to_full = 1.0 / scale

    def rgb(i):
        return (data[i * 4], data[i * 4 + 1], data[i * 4 + 2])

    def full_box(box):
        return [int(box[0] * to_full), int(box[1] * to_full),
                min(fw, int(math.ceil(box[2] * to_full))), min(fh, int(math.ceil(box[3] * to_full)))]

    all_idx = [i for i in range(ww * wh) if mask[i]]
    whole_box = _box(all_idx, ww)
    entry = {
        "id": row["id"], "name": row["name"], "parent": row["parent"], "group": row["group"],
        "category": info["category"], "kind": info["kind"], "word": info["word"], "reason": info["reason"],
        "depth": info.get("depth"),
        "generic": bool(info.get("generic")),
        "opacity": row.get("opacity", 1.0),
        "bbox": full_box(whole_box),
        "area": round(painted / float(ww * wh), 4),
        "canvas_area": fw * fh,
        "colors": _colour_profile([(i % ww, i // ww, rgb(i)) for i in all_idx], whole_box),
        "instances": [],
    }
    kind, category = info["kind"], info["category"]

    if kind in ("ground", "hill", "water", "path", "scatter", "fog", "light"):
        coverage, cell_colors = _grids(mask, data, ww, wh, grid)
        entry["coverage"] = coverage
        entry["cell_colors"] = cell_colors
    if kind == "sky":
        tex, used = _save_crop(surface, full_box(whole_box), out_dir / ("tex_%s_sky.png" % row["id"]), 0.0)
        entry["texture"] = tex
        entry["texture_box"] = list(used)
    if kind not in ("object", "card"):
        return entry

    # ---- pieces: one object per painted piece, or one for the whole layer
    pieces = []
    split = category in SPLIT_CATEGORIES or (info.get("plural") and kind == "object")
    if split:
        comps = _components(mask, ww, wh)
        if comps:
            largest = max(len(c) for c in comps)
            keep = [c for c in comps if len(c) >= max(6, largest * 0.04, ww * wh * 0.0004)]
            keep.sort(key=len, reverse=True)
            if len(keep) > 40:
                entry["kind"] = "scatter"
                coverage, cell_colors = _grids(mask, data, ww, wh, grid)
                entry["coverage"] = coverage
                entry["cell_colors"] = cell_colors
                return entry
            pieces = keep[:24]
    if not pieces:
        pieces = [all_idx]

    for members in pieces:
        box = _box(members, ww)
        bw, bh = box[2] - box[0], box[3] - box[1]
        slices = 1
        if category in ROW_CATEGORIES and bh > 0 and bw / float(bh) > 1.7:
            slices = max(2, min(8, int(round((bw / float(bh)) / 0.95))))
        member_set = set(members)
        for s in range(slices):
            sx0 = box[0] + int(bw * s / slices)
            sx1 = box[0] + int(bw * (s + 1) / slices)
            if slices > 1:
                part = [i for i in members if sx0 <= i % ww < sx1]
                if len(part) < 4:
                    continue
                sub = _box(part, ww)
            else:
                part = members
                sub = box
            k = len(entry["instances"])
            tex, used = _save_crop(surface, full_box(sub), out_dir / ("tex_%s_%02d.png" % (row["id"], k)))
            if not tex:
                continue
            entry["instances"].append({
                "bbox": list(used),
                "area": round(len(part) / float(ww * wh), 5),
                "colors": _colour_profile([(i % ww, i // ww, rgb(i)) for i in part if i in member_set], sub),
                "texture": tex,
            })
    return entry


# --------------------------------------------------------------------------- #
# whole file
# --------------------------------------------------------------------------- #
def analyze_psd_layers(psd_path, out_dir, keywords_path, grid, base_analyze, colors=6):
    """Full layered analysis.  ``base_analyze(png, grid, colors)`` analyses the composite."""
    psd_path = Path(psd_path).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keywords_path = Path(keywords_path)
    vocab = load_keywords(keywords_path)
    st = psd_path.stat()
    cache_key = "%s|%s|%d|%d|%d|%d|%d" % (
        CACHE_VERSION, psd_path, st.st_size, int(st.st_mtime), grid, colors,
        int(keywords_path.stat().st_mtime) if keywords_path.is_file() else 0)
    cache_file = out_dir / "analysis.json"
    if cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("_cache_key") == cache_key:
                return cached
        except Exception:
            pass

    rows, composite = export_layers(psd_path, out_dir)
    for old in out_dir.glob("tex_*.png"):
        old.unlink()
    result = base_analyze(str(composite), grid=grid, colors=colors)
    result["image"] = str(psd_path)
    result["name"] = psd_path.stem
    result["source"] = "psd-layers"
    image_w, image_h = result["width"], result["height"]

    used_groups = []            # groups recognised as one object: skip their children
    skipped_groups = []         # ignored groups: skip children too
    layers = []
    for row in rows:
        parent = row["parent"]
        if any(parent == g or parent.startswith(g + "/") for g in used_groups + skipped_groups):
            continue
        info = classify(row["name"], vocab)
        path_key = (parent + "/" + row["name"]) if parent else "/" + row["name"]
        depth = depth_tag(row["name"], vocab)
        if depth is None:
            for segment in reversed([seg for seg in parent.split("/") if seg]):
                depth = depth_tag(segment, vocab)
                if depth:
                    break
        info["depth"] = depth
        if row["group"]:
            if info["category"] == "unknown" or info.get("weak"):
                continue                      # look inside the group instead (e.g. a group called "BG")
            if info["kind"] == "ignore":
                skipped_groups.append(path_key)
                layers.append({"id": row["id"], "name": row["name"], "parent": parent, "group": True,
                               "category": "ignore", "kind": "ignore", "word": info["word"],
                               "reason": info["reason"], "instances": [], "depth": depth})
                continue
            used_groups.append(path_key)
        if info["kind"] == "ignore":
            layers.append({"id": row["id"], "name": row["name"], "parent": parent, "group": row["group"],
                           "category": "ignore", "kind": "ignore", "word": info["word"],
                           "reason": info["reason"], "instances": [], "depth": depth})
            continue
        entry = analyze_layer(row, info, out_dir, grid, image_w, image_h)
        if entry:
            layers.append(entry)

    # where the walkable ground starts (rows above are sky / backdrop)
    terrain_top = result.get("terrain_start_row", int(image_h * 0.24))
    ground_tops = [l["bbox"][1] for l in layers if l.get("kind") in ("ground", "hill", "water", "path")
                   and l.get("bbox")]
    if ground_tops:
        terrain_top = int(max(image_h * 0.08, min(image_h * 0.6, min(ground_tops))))
    sky_bottoms = [l["bbox"][3] for l in layers if l.get("kind") == "sky" and l.get("bbox")]
    if sky_bottoms and max(sky_bottoms) < image_h * 0.85:
        terrain_top = int(max(terrain_top, min(image_h * 0.6, max(sky_bottoms) * 0.85)))
    result["terrain_start_row"] = terrain_top
    result["layers"] = layers
    result["layer_dir"] = str(out_dir)
    result["_cache_key"] = cache_key
    cache_file.write_text(json.dumps(result), encoding="utf-8")
    return result


def summary_lines(layers, limit=40):
    lines = []
    for layer in layers[:limit]:
        n = len(layer.get("instances") or [])
        kind = layer.get("kind")
        what = {
            "object": "%s x%d" % (layer["category"], max(1, n)),
            "card": "painted card" + (" x%d" % n if n > 1 else ""),
            "scatter": "scattered %s" % layer["category"],
            "ignore": "skipped (%s)" % layer.get("reason", ""),
        }.get(kind, kind)
        tag = {"fg": "  [foreground]", "mg": "  [middle ground]", "bg": "  [background]"}.get(layer.get("depth"), "")
        lines.append("%s -> %s%s" % (layer["name"], what, tag))
    return lines
