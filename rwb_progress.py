"""Tell the Unreal panel how far along a helper script is.

Unreal runs the helpers as separate processes and watches a small file named by RWB_PROGRESS.
Without that environment variable (tests, command line) this does nothing.
"""

import os


def report(fraction, text=""):
    """fraction 0..1 of this helper's work, plus what it is doing now."""
    path = os.environ.get("RWB_PROGRESS")
    if not path:
        return
    fraction = max(0.0, min(1.0, float(fraction)))
    temp = path + ".tmp"
    try:
        with open(temp, "w", encoding="utf-8") as out:
            out.write("%.4f|%s" % (fraction, text))
        os.replace(temp, path)                  # Unreal never reads half a line
    except OSError:
        pass                                    # progress must never break the work


def read(path):
    """(fraction, text) from a progress file, or None if there's nothing yet."""
    try:
        with open(path, encoding="utf-8") as src:
            fraction, _, text = src.read().partition("|")
        return float(fraction), text
    except (OSError, ValueError):
        return None
