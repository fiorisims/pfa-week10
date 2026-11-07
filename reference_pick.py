"""Tiny native file picker used by the Unreal panel.

Kept in the normal system Python so it can use tkinter without depending on
Unreal's bundled interpreter.
"""

import tkinter as tk
from tkinter import filedialog


def main():
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass
    path = filedialog.askopenfilename(
        title="Choose a reference image or Photoshop file",
        filetypes=[
            ("Images and Photoshop", "*.png *.jpg *.jpeg *.bmp *.tga *.gif *.webp *.psd *.psb"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if path:
        print(path)


if __name__ == "__main__":
    main()
