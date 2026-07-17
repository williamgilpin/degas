"""Regenerate assets/colormaps.png, the colormap reference image in the README.

Run from the repository root:
    python assets/generate_colormap_figure.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# make the script work from a checkout even if degas is not installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import degas.colormaps as cm

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colormaps.png")

names = sorted(n for n in cm.__all__ if not n.endswith("_r"))
names.insert(0, names.pop(names.index("vernalis")))

gradient = np.linspace(0, 1, 256)[None, :]

fig, axes = plt.subplots(len(names), 1, figsize=(6, 0.32 * len(names)))
fig.subplots_adjust(top=1, bottom=0, left=0.2, right=0.98, hspace=0.35)
for ax, name in zip(axes, names):
    ax.imshow(gradient, aspect="auto", cmap=getattr(cm, name))
    ax.set_axis_off()
    ax.text(-0.02, 0.5, name, va="center", ha="right", fontsize=10,
            transform=ax.transAxes)

fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {len(names)} colormaps to {out_path}")
