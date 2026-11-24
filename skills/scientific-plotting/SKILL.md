---
name: scientific-plotting
description: >-
  Create clean, publication-grade scientific plots with matplotlib and the degas
  library. Use this skill whenever generating paper figures, presentation slides,
  charts, heatmaps, regression fits, error-band plots, or multi-panel graphics
  with degas house styles (base, default, degas_figure, pastel_rainbow, dark),
  perceptually uniform colormaps (vernalis, batlow, roma, vik, etc.), and
  degas plotting/export utilities.
---

# Scientific Plotting with `degas`

Follow this workflow to produce clean paper figures and presentation plots using `degas` house styles, perceptually uniform colormaps, and helper utilities.

For exact palette hex/RGB tables, full style-sheet parameters, the complete 41-colormap catalog, and API gotchas, consult [references/style-guide.md](./references/style-guide.md).

## 1. Choose and Apply a House Style

All house styles share a common foundation via `base.mplstyle`: **Helvetica** sans-serif typography (clean fallback chain), upright Helvetica math (`mathtext.default: regular`), **`dg.vernalis`** default image colormap, and **transparent high-DPI export** (`savefig.transparent: True`, `savefig.dpi: 300`). `dg.set_style(name)` layers `base.mplstyle` automatically underneath the chosen style.

Light and dark styles serve **different visual roles**:

| Style | Intended Use | Ticks & Spines | Default `figsize` | Typography & Strokes | Color Cycle |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `degas_figure` | **Paper figures** (manuscripts, multi-panel) | Full high-contrast black (`#000000`) box (`0.75pt`) & outward ticks (`3.5pt`/`2pt`) | `(2, 2)` | `10pt` ticks, `12pt` Helvetica labels, `1.5pt` lines, frameless legends | `high_contrast` (7 colors) |
| `pastel_rainbow` | **Paper figures** with >6 series | Full high-contrast black (`#000000`) box (`0.75pt`) & outward ticks (`3.5pt`/`2pt`) | `(2, 2)` | `10pt` ticks, `12pt` Helvetica labels, `1.5pt` lines, frameless legends | `pastel_rainbow` (12 colors) |
| `dark` | **Presentations & talks** (transparent on dark slides) | Full lower-contrast muted box (`#8C8C8C`, `1.1pt`) matching outward `#8C8C8C` ticks | `(3.5, 3.5)` | `12pt` gray ticks, `15pt` white labels, bold `2.5pt` lines, frameless legends | Presentation cycle led by `cornflower_blue` (`#4DAAE8`) |
| `default` | Exploratory scripts | Matplotlib defaults + shared base | `(6.4, 4.8)` | Helvetica + `vernalis` | `high_contrast` (7 colors) |
| `base` | Minimal shared foundation for custom styling | Matplotlib defaults | `(6.4, 4.8)` | Helvetica, `vernalis`, transparent `300 dpi` | Matplotlib default |

Apply a style globally or scope it to a single figure:

```python
import os
import matplotlib.pyplot as plt
import degas as dg

# Global (layers base.mplstyle + chosen style)
dg.set_style("degas_figure")  # paper figures
dg.set_style("dark")          # presentation slides

# Scoped to a single figure (preferred when mixing paper + slide figures)
style_file = os.path.join(dg.data_path, "dark.mplstyle")
with plt.style.context(style_file):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ...
```

> **Size & proportion rule**: Keep figures **small** (`2–3.5 in` per single panel) and export at **`dpi=300` or `dpi=600`**. Working at small physical sizes makes `10pt` (paper) and `12–15pt` (presentation) Helvetica text naturally bold and legible without shrinking into unreadable micro-type.

## 2. Select a Perceptually Uniform Colormap

Importing `degas` registers all 41 colormaps (plus `_r` reversed variants) with matplotlib. Every colormap is available both as `dg.<name>` / `dg.<name>_r` and by string name (e.g. `cmap="vernalis"`).

| Data Type | Recommended Colormaps | Notes |
| :--- | :--- | :--- |
| **Sequential** (0 to max, density, magnitude) | `vernalis` (default), `batlow`, `lipari`, `navia`, `oslo`, `grayC` | Monotonic lightness; safe for colorblind viewers and grayscale printing. |
| **Diverging on light paper figures** | `vik`, `roma`, `broc`, `cork`, `bam` | Neutral light midpoint; center with `TwoSlopeNorm` or symmetric `vmin=-v, vmax=v`. |
| **Diverging on `dark` slides** | `berlin`, `vanimo`, `managua`, `lisbon`, `tofino` | Dark neutral midpoint recedes cleanly into dark slide backgrounds. |
| **Cyclic / periodic** (phase, angle, orientation) | `romaO`, `vikO`, `bamO`, `brocO`, `corkO` | Seamless wrap-around between `0` and `2*pi`. |
| **Multi-sequential** (elevation / threshold break) | `oleron`, `bukavu`, `fes` | Sharp visual boundary at zero/threshold. |

## 3. Use `degas` Plotting and Formatting Helpers

| Helper | Purpose | Key Usage Note |
| :--- | :--- | :--- |
| `dg.plot_bar(y, x=None, width=0.64, corner_radius=1.5, ax=ax)` | Bar chart with house proportions (`0.64` width, ~`0.36` gap, subtle `1.5pt` rounded top corners) | Borderless flat fill, aspect-preserving rounded top corners (flat on `y=0`), balanced `xmargin=0.04`. For histograms pass `rwidth=0.64`. |
| `dg.colorbar(mappable, ax=ax, label=...)` | Frameless colorbar matching the adjacent square axes height | No outline border (`outline.set_visible(False)`), outward ticks, label foot facing inward (`rotation=270` on right), exact box-height alignment. Always prefer over raw `fig.colorbar(...)`. |
| `dg.plot_err(y, errs, x=x, color=dg.blue)` | Line with shaded symmetric or asymmetric `(N, 2)` error band | `y` comes first; `color` must be an RGB 3-tuple in `[0, 1]` (not a hex string). |
| `dg.plot_linear_confidence(x, y, ax=ax, return_model=True)` | Linear regression with 95% CI & PI bands (supports log axes) | Set `ax.set_xscale("log")` / `set_yscale("log")` *before* calling, or pass `logx`/`logy`. |
| `dg.plot_splity(x, y1, y2, ax=ax, kwargs1=..., kwargs2=...)` | Dual y-axes with clean z-order layering | Pass distinct colors in `kwargs1`/`kwargs2` and match axis label colors. |
| `dg.colorline(x, y, z=vals, cmap=dg.vernalis, norm=...)` | Continuous gradient-colored 2D trajectory | Call `ax.autoscale()` or set `xlim`/`ylim` afterward (`LineCollection` skips autoscaling). |
| `dg.fixed_aspect_ratio(ratio, ax=ax)` | Exact visual aspect ratio regardless of axis units or log scales | Call *after* plotting data and setting limits/scales. |
| `dg.xyz_to_image(x, y, z, method="linear")` | Gridded 2D image + `extent` from scattered `(x, y, z)` samples | Returns `(img, extent, xi, yi)` ready for `ax.imshow(img, extent=extent)`. |
| `dg.dim_axes(ax=ax, alpha=0.5)` | De-emphasize spines, ticks, and labels for secondary panels | Useful for reference insets or background trajectories. |
| `dg.vanish_axes(ax)` | Strip all spines, ticks, and locators for borderless images | Pair with `better_savefig(..., remove_border=True)`. |

### Paper Figure Pattern (`degas_figure`, light)

```python
import numpy as np
import matplotlib.pyplot as plt
import degas as dg

dg.set_style("degas_figure")

x = np.linspace(0, 6, 120)
y = np.exp(-0.25 * x) * np.cos(2 * x)
err = 0.08 + 0.04 * np.abs(y)

fig, ax = plt.subplots(figsize=(3.2, 2.4))
dg.plot_err(y, err, x=x, color=dg.blue, ax=ax, linewidth=1.6, label="Measured")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Response (a.u.)")
ax.legend(frameon=False)
dg.fixed_aspect_ratio(0.75, ax=ax)

dg.better_savefig("response.pdf", dpi=600, transparent=True)
plt.close(fig)
```

### Presentation Slide Pattern (`dark`, transparent)

For talks, combine `dark` with translucent light-gray background scatter (`color="#CCCCCC", alpha=0.55, edgecolors="none"`), a bold `dg.cornflower_blue` (`#4DAAE8`) fit + translucent band, and optionally a two-tone white + `#8C8C8C` axis label for units/subtitles:

```python
import numpy as np
import matplotlib.pyplot as plt
import degas as dg

dg.set_style("dark")

rng = np.random.default_rng(7)
x = np.geomspace(3e-3, 25.0, 90)
y = np.clip(35.0 + 10.0 * np.log10(x) + rng.normal(0, 14, size=len(x)), 0.5, 80.0)

fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.set_xscale("log")
ax.scatter(x, y, s=36, color="#CCCCCC", alpha=0.55, edgecolors="none", zorder=2)
dg.plot_linear_confidence(
    x, y, ax=ax, show_pi=False,
    color=dg.cornflower_blue, linewidth=2.6,
    ci_kwargs={"color": dg.cornflower_blue, "alpha": 0.28, "edgecolor": "none"},
)
ax.set_xlabel("Chaoticity")
ax.set_ylabel("Forecast Error")
dg.better_savefig("slide_scatter.png", dpi=300, transparent=True)
plt.close(fig)
```

## 4. Always Export Transparently with `dg.better_savefig`

- **Always use transparent exports** (`transparent=True`, also enabled by default in `base.mplstyle`) and **`dg.better_savefig`** with **`dpi=300` or `dpi=600`** at compact figure sizes whenever possible:
  ```python
  dg.better_savefig("figure.png", dpi=300, transparent=True)  # slides / web
  dg.better_savefig("figure.png", dpi=600, transparent=True)  # high-res raster / print
  dg.better_savefig("figure.pdf", dpi=600, transparent=True)  # vector manuscript figures
  ```
- `dark.mplstyle` sets `figure.facecolor: none`, `axes.facecolor: none`, and `savefig.transparent: True` so figures drop directly onto any dark slide deck without a mismatched dark rectangle behind them.
- For borderless matrices or colormaps, use `dg.vanish_axes(ax)` with `dg.better_savefig(path, dpi=300, remove_border=True, transparent=True)` or `dg.save_matrix_pixels(A, path)`.

## 5. Verify Before Finishing

1. Run the script cleanly (use `matplotlib.use("Agg")` in headless/batch scripts) and confirm zero errors or unexpected warnings.
2. Inspect rendered figures—for `dark` transparent PNGs, composite onto a dark background (e.g. `#181818`) when previewing so white labels and translucent bands are clearly visible.
3. Confirm compact figure size, high DPI (`300` or `600`), transparent background (`alpha == 0` outside ink), readable Helvetica labels, and sensible axis limits (especially after `dg.colorline`).
4. Close figures with `plt.close(fig)` when generating batches to avoid memory buildup.
