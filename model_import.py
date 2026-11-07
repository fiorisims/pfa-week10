"""Load 3D models into the viewer.

``load_model(path)`` returns ``(faces, summary)`` where ``faces`` is a list of
``([(x, y, z), ...], (r, g, b))`` polygons - the exact shape the software
renderer draws.

OBJ files are parsed directly (``v`` / ``f`` / ``usemtl`` plus the ``Kd`` colours
in the referenced ``.mtl``). Maya scenes (``.ma`` / ``.mb``) are converted to a
temporary OBJ with Autodesk Maya's ``mayapy.exe`` first - Maya's own loader is
the only reliable way to evaluate construction history and the compressed
``polyFaces`` mesh format, and ``.ma`` files with history store no vertex data
at all. The converted OBJ is then parsed like any other.

Requires Maya installed for ``.ma`` / ``.mb``. ``find_mayapy()`` searches the
platform's usual install locations on Windows, macOS and Linux (or the
``MAYAPY`` environment variable, or ``mayapy`` on ``PATH``).
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_COLOR = (0.72, 0.73, 0.76)

FIT_SIZE = 1.7                      # imported models are scaled to this size
FIT_TARGET_Y = FIT_SIZE * 0.5       # camera looks at the model's middle
FIT_DISTANCE = FIT_SIZE * 2.5       # and sits this far back


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def load_model(path):
    """Return ``(faces, summary)`` for an OBJ / Maya file. Raises on failure."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return _load_obj(path)
    if suffix in (".ma", ".mb"):
        tmp_dir, obj = _maya_to_obj(path)
        try:
            return _load_obj(obj)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    raise ValueError("unsupported file type: %s" % (path.suffix or path.name))


def fit_to_view(faces, size=FIT_SIZE):
    """Center the model on the floor at the origin and scale it to ``size``."""
    xs, ys, zs = [], [], []
    for pts, _ in faces:
        for p in pts:
            xs.append(p[0])
            ys.append(p[1])
            zs.append(p[2])
    if not xs:
        return faces
    cx = (min(xs) + max(xs)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    min_y = min(ys)
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) or 1.0
    k = size / span
    return [([((p[0] - cx) * k, (p[1] - min_y) * k, (p[2] - cz) * k)
              for p in pts], color) for pts, color in faces]


# --------------------------------------------------------------------------- #
# OBJ
# --------------------------------------------------------------------------- #
def _parse_mtl(path):
    colors, name = {}, None
    try:
        fh = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return colors
    with fh:
        for line in fh:
            p = line.split()
            if not p or p[0].startswith("#"):
                continue
            if p[0] == "newmtl":
                name = " ".join(p[1:])
                colors.setdefault(name, DEFAULT_COLOR)
            elif p[0] == "Kd" and name is not None and len(p) >= 4:
                try:
                    r, g, b = float(p[1]), float(p[2]), float(p[3])
                except ValueError:
                    continue
                colors[name] = (min(1.0, max(0.0, r)), min(1.0, max(0.0, g)),
                                min(1.0, max(0.0, b)))
    return colors


def _load_obj(path):
    path = Path(path)
    verts, colors, faces = [], {}, []
    current = DEFAULT_COLOR
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line or line[0] in "#\n\r":
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v" and len(parts) >= 4:
                try:
                    verts.append((float(parts[1]), float(parts[2]),
                                  float(parts[3])))
                except ValueError:
                    pass
            elif tag == "mtllib":
                for name in parts[1:]:
                    mtl = path.parent / name
                    if mtl.exists():
                        colors.update(_parse_mtl(mtl))
                        break
            elif tag == "usemtl":
                current = colors.get(" ".join(parts[1:]), DEFAULT_COLOR)
            elif tag == "f" and len(parts) >= 4:
                idx = []
                for tok in parts[1:]:
                    v = tok.split("/")[0]
                    try:
                        i = int(v)
                    except ValueError:
                        idx = []
                        break
                    i = len(verts) + i if i < 0 else i - 1
                    if not 0 <= i < len(verts):
                        idx = []
                        break
                    idx.append(i)
                if len(idx) >= 3:
                    faces.append(([verts[i] for i in idx], current))
    if not faces:
        raise ValueError("no polygon faces found in %s" % path.name)
    return faces, "%d faces, %d verts" % (len(faces), len(verts))


# --------------------------------------------------------------------------- #
# Maya .ma / .mb -> OBJ via mayapy.exe
# --------------------------------------------------------------------------- #
_CONVERTER = r'''
import sys
import maya.standalone
maya.standalone.initialize(name="python")
import maya.cmds as cmds
src, dst = sys.argv[1], sys.argv[2]
cmds.file(src, open=True, force=True, prompt=False)
if not cmds.pluginInfo("objExport", q=True, loaded=True):
    cmds.loadPlugin("objExport")
opts = "groups=0;ptgroups=0;materials=1;smoothing=1;normals=1"
cmds.file(dst, force=True, type="OBJexport", options=opts, exportAll=True)
print("CONVERTED")
'''


def find_mayapy():
    """Locate ``mayapy`` on Windows, macOS or Linux (or ``%MAYAPY%``)."""
    env = os.environ.get("MAYAPY")
    if env and Path(env).exists():
        return env
    on_path = shutil.which("mayapy")
    if on_path:
        return on_path
    if sys.platform.startswith("win"):
        patterns = [r"C:\Program Files\Autodesk\Maya*\bin\mayapy.exe"]
    elif sys.platform == "darwin":
        patterns = [
            "/Applications/Autodesk/maya*/Maya.app/Contents/bin/mayapy",
            "/Applications/Autodesk/Maya*/Maya.app/Contents/bin/mayapy",
            "/Applications/maya*/Maya.app/Contents/bin/mayapy",
        ]
    else:                                            # Linux
        patterns = [
            "/usr/autodesk/maya*/bin/mayapy",
            "/opt/autodesk/maya*/bin/mayapy",
            "/usr/local/autodesk/maya*/bin/mayapy",
        ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern))
    return sorted(found)[-1] if found else None


def _maya_to_obj(path):
    mayapy = find_mayapy()
    if not mayapy:
        raise RuntimeError(
            "mayapy.exe not found - Maya is required to read .ma/.mb "
            "(set the MAYAPY environment variable to its path)")
    tmp_dir = tempfile.mkdtemp(prefix="hv_import_")
    conv = Path(tmp_dir) / "_convert.py"
    conv.write_text(_CONVERTER, encoding="utf-8")
    obj = Path(tmp_dir) / (path.stem + ".obj")
    try:
        proc = subprocess.run([mayapy, str(conv), str(path), str(obj)],
                              capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Maya conversion timed out (300s)")
    if not obj.exists():
        tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Maya could not convert the scene. %s" % tail)
    return tmp_dir, obj
