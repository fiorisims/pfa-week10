"""Reference World Builder - the 3D model library.

Imports model packs from ``Default Project/ModelLibrary`` (zips or loose
FBX / GLB / GLTF / OBJ files, plus any ``kenney_*.zip`` in Downloads) and
indexes every static mesh by what its FILE NAME says it is, using the same
vocabulary as the layer names (``rwb_layer_keywords.json``).  The 15 nature
models already in the project are always included.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from pathlib import Path

import unreal

LIB_DEST = "/Game/ReferenceWorldBuilder/Library"
BUILTIN = "/Game/WorldBuilder/Imported/Kenney"
MODEL_EXTS = (".fbx", ".glb", ".gltf", ".obj")
EXT_RANK = {".fbx": 0, ".glb": 1, ".gltf": 2, ".obj": 3}


def _wb():
    import world_builder
    return world_builder


def _safe(text, size=40):
    return (re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_") or "pack")[:size]


# --------------------------------------------------------------------------- #
# name -> category (same rules as reference_layers.classify, without pygame)
# --------------------------------------------------------------------------- #
def _tokens(text):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text).lower()
    return re.findall(r"[a-z]+", text)


def _variants(token):
    yield token
    if token.endswith("ies") and len(token) > 4:
        yield token[:-3] + "y"
    if token.endswith("es") and len(token) > 3:
        yield token[:-2]
    if token.endswith("s") and len(token) > 3:
        yield token[:-1]


def load_vocab(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    words, phrases, kinds = {}, [], {}
    for category, spec in (data.get("categories") or {}).items():
        kinds[category] = spec.get("kind", "object")
        for word in spec.get("words", []):
            word = str(word).lower().strip()
            if " " in word:
                phrases.append((tuple(word.split()), category))
            elif word:
                words[word] = category
    ignore = set(str(w).lower() for w in data.get("library_ignore", []))
    return {"words": words, "phrases": phrases, "kinds": kinds, "ignore": ignore}


def classify_name(stem, vocab):
    tokens = _tokens(stem)
    if any(t in vocab["ignore"] for t in tokens):
        return None, None
    matches = []
    for size in (3, 2):
        for start in range(0, max(0, len(tokens) - size + 1)):
            chunk = tokens[start:start + size]
            for phrase, category in vocab["phrases"]:
                if len(phrase) == size and all(p in set(_variants(t)) for p, t in zip(phrase, chunk)):
                    matches.append((start + size - 1, size, category, " ".join(phrase)))
    for i, token in enumerate(tokens):
        for v in _variants(token):
            if v in vocab["words"]:
                matches.append((i, 1, vocab["words"][v], v))
                break
    if not matches:
        return None, None
    matches.sort(key=lambda m: (m[0], m[1]))
    _pos, _size, category, word = matches[-1]
    return category, word


# --------------------------------------------------------------------------- #
# import
# --------------------------------------------------------------------------- #
def _sources(root_dir):
    lib = Path(root_dir) / "ModelLibrary"
    lib.mkdir(exist_ok=True)
    found = []
    for path in sorted(lib.rglob("*")):
        if path.suffix.lower() == ".zip" or path.suffix.lower() in MODEL_EXTS:
            found.append(path)
    downloads = Path(os.path.expanduser("~")) / "Downloads"
    if downloads.is_dir():
        found.extend(sorted(downloads.glob("kenney_*.zip")))
    return found


def _pick_members(names):
    """One file per model name, preferring FBX > GLB > GLTF > OBJ."""
    best = {}
    for name in names:
        ext = os.path.splitext(name)[1].lower()
        if ext not in MODEL_EXTS or "/__macosx/" in name.lower():
            continue
        stem = os.path.splitext(os.path.basename(name))[0]
        if stem not in best or EXT_RANK[ext] < EXT_RANK[os.path.splitext(best[stem])[1].lower()]:
            best[stem] = name
    return best


def _import_files(files, destination):
    tasks = []
    for path in files:
        task = unreal.AssetImportTask()
        task.set_editor_property("filename", str(path))
        task.set_editor_property("destination_path", destination)
        task.set_editor_property("automated", True)
        task.set_editor_property("replace_existing", True)
        task.set_editor_property("save", True)
        if str(path).lower().endswith(".fbx"):
            options = unreal.FbxImportUI()
            options.set_editor_property("import_mesh", True)
            options.set_editor_property("import_as_skeletal", False)
            options.set_editor_property("import_animations", False)
            options.set_editor_property("import_materials", True)
            options.set_editor_property("import_textures", True)
            try:
                options.static_mesh_import_data.set_editor_property("combine_meshes", True)
                options.static_mesh_import_data.set_editor_property("auto_generate_collision", True)
            except Exception:
                pass
            task.set_editor_property("options", options)
        tasks.append(task)
    if tasks:
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(tasks)


def ensure_imported(root_dir, project_dir, log=None):
    """Import any new packs.  Returns a list of messages."""
    messages = []
    record_path = Path(project_dir) / "Saved" / "RWBLibrary" / "imported.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception:
        record = {}
    work = Path(project_dir) / "Intermediate" / "RWBLibrary"
    for source in _sources(root_dir):
        stamp = "%d|%d" % (source.stat().st_size, int(source.stat().st_mtime))
        if record.get(str(source)) == stamp:
            continue
        pack = _safe(source.stem)
        destination = LIB_DEST + "/" + pack
        try:
            if source.suffix.lower() == ".zip":
                with zipfile.ZipFile(source) as archive:
                    members = _pick_members(archive.namelist())
                    target = work / pack
                    target.mkdir(parents=True, exist_ok=True)
                    files = []
                    for stem, member in members.items():
                        out = target / os.path.basename(member)
                        with archive.open(member) as src, open(out, "wb") as dst:
                            dst.write(src.read())
                        files.append(out)
                    # textures that GLTF/OBJ files reference sit next to them in the zip
                    for name in archive.namelist():
                        if name.lower().endswith((".png", ".jpg", ".bin", ".mtl")):
                            out = target / os.path.basename(name)
                            if not out.exists():
                                with archive.open(name) as src, open(out, "wb") as dst:
                                    dst.write(src.read())
            else:
                files = [source]
                destination = LIB_DEST + "/Loose"
            unreal.log("Reference World Builder: importing %d models from %s ..." % (len(files), source.name))
            _import_files(files, destination)
            record[str(source)] = stamp
            messages.append("imported %d models from %s" % (len(files), source.name))
        except Exception as error:
            messages.append("could not import %s: %s" % (source.name, error))
    record_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return messages


def index(vocab_path):
    """{category: [ {mesh, name, word, height, width} ]} for every usable static mesh."""
    vocab = load_vocab(vocab_path)
    out = {}
    paths = []
    for folder in (BUILTIN, LIB_DEST):
        if unreal.EditorAssetLibrary.does_directory_exist(folder):
            paths.extend(unreal.EditorAssetLibrary.list_assets(folder, recursive=True, include_folder=False))
    seen = set()
    for path in paths:
        name = str(path).split(".")[-1] if "." in str(path) else str(path).split("/")[-1]
        stem = re.sub(r"^(SM_|S_)", "", name)
        try:
            data = unreal.EditorAssetLibrary.find_asset_data(path)
            if str(data.asset_class_path.asset_name) != "StaticMesh":
                continue
        except Exception:
            pass
        if stem in seen:
            continue
        category, word = classify_name(stem, vocab)
        if not category or vocab["kinds"].get(category) not in ("object", "card", "scatter"):
            continue
        mesh = unreal.load_asset(path)
        if not isinstance(mesh, unreal.StaticMesh):
            continue
        seen.add(stem)
        bounds = mesh.get_bounds()
        out.setdefault(category, []).append({
            "mesh": mesh, "name": stem, "word": word,
            "height": max(1.0, bounds.box_extent.z * 2.0),
            "width": max(1.0, max(bounds.box_extent.x, bounds.box_extent.y) * 2.0),
            "origin_z": bounds.origin.z - bounds.box_extent.z,
        })
    return out
