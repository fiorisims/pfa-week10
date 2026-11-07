"""Find a 3D model that matches a description (Reference World Builder).

You type what an object is ("birdcage", "spotted toadstool"); this searches three sources and ranks
what it finds by how well each model's picture matches your words AND the shape you painted:

* **Poly Haven** - ~500 high-quality models, all CC0 (free for anything, no credit needed).
* **Objaverse** - a large research collection of Sketchfab models, sorted into 1,156 everyday
  categories (mushrooms, birdcages, lanterns, canoes, butterflies...).  Only models whose authors
  chose CC0 or CC-BY are used; their credits are written to ``ModelLibrary/_online/CREDITS.txt``.
* **Your ModelLibrary folder** - any .glb / .gltf / .fbx / .obj you drop there, matched by file name.

Ranking uses CLIP (already used for recognition): your words against each model's preview image, plus
the painted cut-out against the same preview, so a "birdcage" that is tall and round like the one you
painted beats a flat one.  Downloads are cached; nothing is downloaded twice.

CLI (used by Unreal):
    python reference_catalog.py search --query "birdcage" [--cutout CUT.png] --out results.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
LIBRARY = HERE / "ModelLibrary"
ONLINE = LIBRARY / "_online"
CREDITS = ONLINE / "CREDITS.txt"
USER_AGENT = "ReferenceWorldBuilder/1.0 (student art tool)"
POLYHAVEN_LIST = "https://api.polyhaven.com/assets?t=models"
POLYHAVEN_FILES = "https://api.polyhaven.com/files/%s"
POLYHAVEN_THUMB = "https://cdn.polyhaven.com/asset_img/thumbs/%s.png?width=256&height=256"
OBJAVERSE = "allenai/objaverse"
ALLOWED_LICENSES = {"cc0": "CC0", "by": "CC-BY 4.0", "by-sa": "CC-BY-SA 4.0"}   # no NC / ND
CATALOG_DAYS = 7
MAX_FACES = 400000            # bigger downloads are scans, far too heavy to adapt quickly
PER_CATEGORY = 18             # Objaverse models looked at for each matching category
CATEGORIES = 3                # matching Objaverse categories looked at
POLYHAVEN_TOP = 10            # Poly Haven models looked at
BLANK_PREVIEW = 12.0         # a preview this flat (pixel spread) shows no object
SHAPE_WEIGHT = 0.25           # how much "looks like what I painted" counts next to "matches my words"


def _ssl():
    try:
        import truststore
        truststore.inject_into_ssl()                 # the system's certificates (school networks)
    except Exception:
        pass


def fetch_bytes(url, timeout=30):
    _ssl()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _fresh(path, days):
    return path.is_file() and time.time() - path.stat().st_mtime < days * 86400


# --------------------------------------------------------------------------- #
# words and pictures
# --------------------------------------------------------------------------- #
class Clip:
    """CLIP text + image embeddings (the models recognition already downloaded)."""

    def __init__(self):
        import reference_objects as ro
        from tokenizers import Tokenizer
        self.ro = ro
        self.tok = Tokenizer.from_file(str(ro._file(ro.CLIP_REPO, ro.CLIP_TOKENIZER, "clip")))
        self.tok.no_padding()
        self.tok.no_truncation()
        self.text_model = ro._session(ro._file(ro.CLIP_REPO, ro.CLIP_TEXT, "clip"))
        self.vision = ro._session(ro._file(ro.CLIP_REPO, ro.CLIP_VISION, "clip"))

    def texts(self, phrases, batch=64):
        out = []
        for start in range(0, len(phrases), batch):
            chunk = phrases[start:start + batch]
            ids = np.zeros((len(chunk), 77), np.int64)
            for i, phrase in enumerate(chunk):
                encoded = list(self.tok.encode(phrase).ids)[:77]
                ids[i, :len(encoded)] = encoded
            emb = self.text_model.run(None, {"input_ids": ids})[0]
            out.append(emb / np.linalg.norm(emb, axis=-1, keepdims=True))
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    def image(self, rgb):
        """rgb: HxWx3 0-255 (any size)."""
        import pygame
        h, w = rgb.shape[:2]
        side = max(h, w)
        canvas = np.full((side, side, 3), 128, np.uint8)
        canvas[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = rgb[..., :3]
        surf = pygame.transform.smoothscale(pygame.image.frombuffer(canvas.tobytes(), (side, side), "RGB"), (224, 224))
        x = np.frombuffer(pygame.image.tobytes(surf, "RGB"), np.uint8).reshape(224, 224, 3).astype(np.float32) / 255.0
        x = (x - self.ro.CLIP_MEAN) / self.ro.CLIP_STD
        emb = self.vision.run(None, {"pixel_values": x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        return emb / np.linalg.norm(emb)


def _picture(data):
    """Bytes of a PNG/JPEG -> HxWx3 array (on grey where transparent)."""
    import io
    import pygame
    surf = pygame.image.load(io.BytesIO(data))
    w, h = surf.get_size()
    base = pygame.Surface((w, h))
    base.fill((128, 128, 128))
    base.blit(surf, (0, 0))
    return np.frombuffer(pygame.image.tobytes(base, "RGB"), np.uint8).reshape(h, w, 3)


def cutout_picture(path):
    import pygame
    surf = pygame.image.load(str(path))
    w, h = surf.get_size()
    base = pygame.Surface((w, h))
    base.fill((128, 128, 128))
    base.blit(surf, (0, 0))
    return np.frombuffer(pygame.image.tobytes(base, "RGB"), np.uint8).reshape(h, w, 3)


def phrase_for(text):
    """'statue_(sculpture)' / 'ArmChair_01' -> 'statue sculpture' / 'arm chair'."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    text = re.sub(r"[_()\-]+", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    return " ".join(text.lower().split())


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def polyhaven_catalog():
    """{id: info} of Poly Haven's models, cached for a week."""
    path = ONLINE / "polyhaven_models.json"
    if not _fresh(path, CATALOG_DAYS):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_bytes(fetch_bytes(POLYHAVEN_LIST))
        except Exception:
            if not path.is_file():
                raise
    return json.loads(path.read_text(encoding="utf-8"))


def objaverse_index():
    """(category -> uids, uid -> path inside the dataset), both small and cached by Hugging Face."""
    from huggingface_hub import hf_hub_download
    lvis = json.load(gzip.open(hf_hub_download(OBJAVERSE, "lvis-annotations.json.gz", repo_type="dataset")))
    paths = json.load(gzip.open(hf_hub_download(OBJAVERSE, "object-paths.json.gz", repo_type="dataset")))
    return lvis, paths


def objaverse_metadata(batch):
    """Licence, author and preview of every model in one Objaverse batch (~3.5 MB, cached)."""
    from huggingface_hub import hf_hub_download
    return json.load(gzip.open(hf_hub_download(OBJAVERSE, "metadata/%s.json.gz" % batch, repo_type="dataset")))


def library_files():
    """Models you dropped in ModelLibrary (not the online cache)."""
    found = []
    for path in sorted(LIBRARY.rglob("*")) if LIBRARY.is_dir() else []:
        if path.suffix.lower() in (".glb", ".gltf", ".fbx", ".obj") and ONLINE not in path.parents:
            found.append(path)
    return found


def _embeddings(name, phrases, clip):
    """Text embeddings of a list of phrases, cached next to the downloads."""
    path = ONLINE / ("%s_text.npz" % name)
    key = str(hash("\n".join(phrases)) & 0xFFFFFFFF)
    if path.is_file():
        data = np.load(path)
        if str(data["key"]) == key and len(data["emb"]) == len(phrases):
            return data["emb"]
    emb = clip.texts(phrases)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, emb=emb, key=key)
    return emb


def _thumb(url, cache_name):
    """A preview image, downloaded once, saved with the extension its bytes say (Unreal needs it)."""
    folder = ONLINE / "thumbs"
    for ext in (".png", ".jpg"):
        if (folder / (cache_name + ext)).is_file():
            return folder / (cache_name + ext)
    data = fetch_bytes(url, timeout=20)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (cache_name + (".png" if data[:4] == b"\x89PNG" else ".jpg"))
    path.write_bytes(data)
    return path


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def gather(query, clip, query_emb):
    """Candidates from every source, each with a preview image path where there is one."""
    candidates = []
    notes = []
    # Poly Haven: words in name, tags and categories
    try:
        catalog = polyhaven_catalog()
        ids = sorted(catalog)
        texts = ["a 3D model of %s, %s" % (catalog[i]["name"], ", ".join(catalog[i].get("tags", [])[:8])) for i in ids]
        sims = _embeddings("polyhaven", texts, clip) @ query_emb
        for i in np.argsort(-sims)[:POLYHAVEN_TOP]:
            asset = ids[int(i)]
            authors = ", ".join(catalog[asset].get("authors", {}))
            candidates.append({"source": "polyhaven", "id": asset, "name": catalog[asset]["name"],
                               "license": "CC0", "author": authors, "url": "https://polyhaven.com/a/%s" % asset,
                               "thumb_url": POLYHAVEN_THUMB % asset, "text_score": float(sims[int(i)])})
    except Exception as error:
        notes.append("Poly Haven unavailable (%s)" % str(error).splitlines()[0][:80])
    # Objaverse: the categories closest to the words, then their freely licensed models
    try:
        lvis, paths = objaverse_index()
        names = sorted(lvis)
        sims = _embeddings("objaverse_categories", ["a 3D model of a %s" % phrase_for(n) for n in names], clip) @ query_emb
        order = np.argsort(-sims)
        chosen = [names[int(i)] for i in order[:CATEGORIES] if sims[int(i)] >= sims[int(order[0])] - 0.06]
        for category in chosen:
            uids = sorted(lvis[category])
            # look in the batches holding most of this category: each batch's licences are one download
            batches = {}
            for uid in uids:
                if uid in paths:
                    batches.setdefault(paths[uid].split("/")[1], []).append(uid)
            taken = 0
            for batch, members in sorted(batches.items(), key=lambda kv: -len(kv[1]))[:3]:
                meta = objaverse_metadata(batch)
                for uid in members:
                    info = meta.get(uid) or {}
                    licence = info.get("license")
                    if licence not in ALLOWED_LICENSES or (info.get("faceCount") or 0) > MAX_FACES:
                        continue
                    images = sorted((info.get("thumbnails") or {}).get("images", []), key=lambda im: im["width"])
                    preview = next((im["url"] for im in images if im["width"] >= 200), None)
                    if not preview:
                        continue
                    candidates.append({
                        "source": "objaverse", "id": uid, "name": info.get("name") or phrase_for(category),
                        "category": category, "license": ALLOWED_LICENSES[licence],
                        "author": (info.get("user") or {}).get("displayName", ""),
                        "url": info.get("viewerUrl") or "https://sketchfab.com/3d-models/%s" % uid,
                        "path": paths[uid], "thumb_url": preview, "faces": info.get("faceCount"),
                        "text_score": float(sims[names.index(category)])})
                    taken += 1
                    if taken >= PER_CATEGORY:
                        break
                if taken >= PER_CATEGORY:
                    break
        if not chosen:
            notes.append("no Objaverse category matched")
    except Exception as error:
        notes.append("Objaverse unavailable (%s)" % str(error).splitlines()[0][:80])
    # your own files, by name
    files = library_files()
    if files:
        sims = clip.texts(["a 3D model of a %s" % phrase_for(f.stem) for f in files]) @ query_emb
        for f, s in zip(files, sims):
            candidates.append({"source": "library", "id": f.name, "name": phrase_for(f.stem), "license": "your file",
                               "author": "", "url": str(f), "file": str(f), "thumb_url": None, "text_score": float(s)})
    return candidates, notes


def rank(candidates, query_emb, clip, cutout_emb=None):
    """Score = how the preview matches the words, nudged by how it matches the painting.

    Models with no preview can't be judged against the words the same way, so only your own
    library files (which never have one) are kept without it.
    """
    judged = []
    for c in candidates:
        if not c.get("thumb_url"):
            if c["source"] == "library":
                c["score"] = c["text_score"] - 0.02
                judged.append(c)
            continue
        try:
            cache = "%s_%s" % (c["source"], re.sub(r"[^A-Za-z0-9_]", "_", c["id"]))
            path = _thumb(c["thumb_url"], cache)
            picture = _picture(path.read_bytes())
        except Exception as error:
            c["preview_error"] = str(error)[:80]
            continue
        if picture.std() < BLANK_PREVIEW:
            continue                               # an empty backdrop: nothing to judge or to show you
        emb = clip.image(picture)
        c["thumb"] = str(path)
        c["image_score"] = float(emb @ query_emb)
        if cutout_emb is not None:
            c["shape_score"] = float(emb @ cutout_emb)
        judged.append(c)
    # every picture looks a bit like every other to CLIP: compare shapes with each other, not in absolute
    shapes = [c["shape_score"] for c in judged if "shape_score" in c]
    middle = float(np.mean(shapes)) if shapes else 0.0
    for c in judged:
        if "image_score" in c:
            c["score"] = c["image_score"] + 0.25 * (c["text_score"] - 0.8)
            if "shape_score" in c:
                c["score"] += SHAPE_WEIGHT * (c["shape_score"] - middle)
    return sorted(judged, key=lambda c: -c["score"])


def search(query, cutout=None, limit=6):
    """Ranked matches for a description.  Returns {matches, notes}."""
    query = " ".join(str(query).split())
    if not query:
        raise ValueError("type what the object is first")
    clip = Clip()
    query_emb = clip.texts(["a 3D model of a %s" % query])[0]
    candidates, notes = gather(query, clip, query_emb)
    cutout_emb = clip.image(cutout_picture(cutout)) if cutout and Path(cutout).is_file() else None
    ranked = rank(candidates, query_emb, clip, cutout_emb)
    # one per model; mixed sources near the top so there's a real choice
    seen, matches = set(), []
    for c in ranked:
        if (c["source"], c["id"]) in seen:
            continue
        seen.add((c["source"], c["id"]))
        matches.append(c)
        if len(matches) >= limit:
            break
    failed = sum(1 for c in candidates if c.get("preview_error"))
    if failed:
        notes.append("%d previews couldn't be loaded" % failed)
    return {"query": query, "matches": matches, "notes": notes, "looked_at": len(candidates)}


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def download(match):
    """Local file for a match (downloading it the first time).  Returns the path."""
    source = match["source"]
    if source == "library":
        return match["file"]
    if source == "objaverse":
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(OBJAVERSE, match["path"], repo_type="dataset")
        out = ONLINE / "objaverse" / (match["id"] + ".glb")
        if not out.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, out)
        return str(out)
    if source == "polyhaven":
        folder = ONLINE / "polyhaven" / match["id"]
        files = json.loads(fetch_bytes(POLYHAVEN_FILES % match["id"]))
        gltf = files["gltf"]["1k"]["gltf"]
        main = folder / Path(gltf["url"]).name
        if not main.is_file():
            folder.mkdir(parents=True, exist_ok=True)
            for rel, extra in (gltf.get("include") or {}).items():
                target = folder / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(fetch_bytes(extra["url"], timeout=120))
            main.write_bytes(fetch_bytes(gltf["url"], timeout=120))
        return str(main)
    raise ValueError("unknown source " + str(source))


def credit(match):
    """One line in CREDITS.txt for a model that ended up in a world (CC-BY asks for this)."""
    if match.get("source") == "library":
        return
    line = "%s - %s by %s (%s) %s" % (match.get("source"), match.get("name"), match.get("author") or "unknown",
                                       match.get("license"), match.get("url"))
    CREDITS.parent.mkdir(parents=True, exist_ok=True)
    existing = CREDITS.read_text(encoding="utf-8").splitlines() if CREDITS.is_file() else []
    if line not in existing:
        with open(CREDITS, "a", encoding="utf-8") as out:
            if not existing:
                out.write("Models used by Reference World Builder, with their licences.\n"
                          "CC-BY models must be credited if you share or publish work that uses them.\n\n")
            out.write(line + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Find a 3D model by description")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--query", required=True)
    s.add_argument("--cutout", default=None)
    s.add_argument("--limit", type=int, default=6)
    s.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        import pygame
        pygame.display.init()
        import rwb_progress
        rwb_progress.report(0.1, "Searching for '%s'" % args.query)
        result = search(args.query, args.cutout, args.limit)
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    except Exception as error:
        print("CATALOG_ERROR: %s" % error, file=sys.stderr)
        return 1
    print(json.dumps({"out": args.out, "matches": len(result["matches"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
