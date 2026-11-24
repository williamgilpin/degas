# `degas` Style Guide & Reference

Detailed specification of `degas` matplotlib style sheets, color palettes, perceptually uniform colormaps, export conventions, and common idioms/gotchas.

---

## 1. House Style Sheets (`degas/styles/`)

Access the directory at runtime via `degas.data_path` (or `degas.style_path`). Activate globally with `dg.set_style("<name>")` (which automatically layers `base.mplstyle` underneath `<name>.mplstyle`) or per figure with `plt.style.context(os.path.join(dg.data_path, "<name>.mplstyle"))`.

### Shared Base (`base.mplstyle`)
All house styles share:
- **Font & Vector Text**: `font.family: sans-serif`, `font.sans-serif: Helvetica, Helvetica Neue, Arial, Open Sans, DejaVu Sans, sans-serif`, `mathtext.default: regular` (upright Helvetica for exponents like `$10^{-2}$`), plus editable vector text (`pdf.fonttype: 42`, `ps.fonttype: 42`, `svg.fonttype: none`).
- **Colormap**: `image.cmap: vernalis`.
- **Rendering, Bars & Legends**: `lines.antialiased: True`, `patch.antialiased: True`, `patch.edgecolor: none`, `patch.linewidth: 0.0`, `patch.force_edgecolor: False`, `axes.grid: False`, `legend.frameon: False`.
- **Export**: `savefig.transparent: True`, `savefig.dpi: 300`.

### Paper vs Presentation Comparison

| Parameter | `default` | `degas_figure` (Paper) | `pastel_rainbow` (Paper) | `dark` (Presentation) |
| :--- | :--- | :--- | :--- | :--- |
| **Role** | Exploratory | Journal manuscripts | Multi-series manuscripts (>6 curves) | Slides, talks, dark posters |
| **Figure / Axes bg** | mpl default | mpl default | mpl default | `none` (fully transparent) |
| **Savefig export** | `transparent: True`, `300 dpi` | `transparent: True`, `300 dpi` | `transparent: True`, `300 dpi` | `transparent: True`, `300 dpi` |
| **Text / axis labels** | black | `#000000` (`12pt` labels, `10pt` ticks) | `#000000` (`12pt` labels, `10pt` ticks) | `#FFFFFF` (`15pt` labels, `labelpad=8`) |
| **Spines (full box)** | black | `#000000` (`0.75pt`, full 4-sided box) | `#000000` (`0.75pt`, full 4-sided box) | `#8C8C8C` (`1.1pt`, matches tick labels) |
| **Ticks** | mpl default | **outward** (`3.5pt`/`2pt`), `#000000` ticks & `10pt` labels | **outward** (`3.5pt`/`2pt`), `#000000` ticks & `10pt` labels | **outward** (`5pt`/`2.5pt`), `#8C8C8C` ticks & `12pt` labels |
| **Default `figsize`** | `(6.4, 4.8)` | `(2, 2)` | `(2, 2)` | `(3.5, 3.5)` |
| **Line width / markers** | mpl default | `linewidth: 1.5` | `linewidth: 1.5` | `linewidth: 2.5`, `markersize: 6.5` |
| **Margins** (`x`/`y`) | `0.05` / `0.05` | `0.01` / `0.01` | `0.01` / `0.01` | `0.02` / `0.02` |
| **Legend** | `frameon: False` | `frameon: False` | `frameon: False` | `frameon: False`, `11pt`, transparent |
| **Colorbars** | `dg.colorbar(...)` | Frameless, outward ticks, label foot facing inward, exact box height | Frameless, outward ticks, label foot facing inward, exact box height | Frameless, outward ticks, label foot facing inward, exact box height |
| **Image defaults** | `cmap: vernalis` | `vernalis`, `origin: lower`, `nearest` | `vernalis`, `origin: lower`, `nearest` | `vernalis`, `origin: lower`, `nearest` |
| **Color cycle** | `high_contrast` (7) | `high_contrast` (7) | `pastel_rainbow` (12) | Dark presentation cycle led by `#4DAAE8` (7) |

---

## 2. Color Palettes & Color Utilities

### `degas.high_contrast` (Light paper styles: `default`, `degas_figure`)

Exposed as `dg.high_contrast` and unpacked as top-level RGB tuples `dg.blue`, `dg.red`, `dg.turquoise`, `dg.purple`, `dg.magenta`, `dg.orange`, `dg.gray`.

| Index | Attribute | Hex | RGB `[0, 1]` |
| :---: | :--- | :--- | :--- |
| 0 | `dg.blue` | `#5F98FF` | `(0.3725, 0.5961, 1.0000)` |
| 1 | `dg.red` | `#FF6347` | `(1.0000, 0.3882, 0.2784)` |
| 2 | `dg.turquoise` | `#35AD99` | `(0.2078, 0.6784, 0.6000)` |
| 3 | `dg.purple` | `#9842E3` | `(0.5961, 0.2588, 0.8902)` |
| 4 | `dg.magenta` | `#CD1076` | `(0.8039, 0.0627, 0.4627)` |
| 5 | `dg.orange` | `#EAAE1B` | `(0.9176, 0.6824, 0.1059)` |
| 6 | `dg.gray` | `#B2B2B2` | `(0.7000, 0.7000, 0.7000)` |

### Presentation Dark Cycle (`dark.mplstyle`)

Led by `dg.cornflower_blue` (`#4DAAE8`) followed by bright, projector-legible variants of `degas.high_contrast`:

| Index | Role | Hex | RGB `[0, 1]` |
| :---: | :--- | :--- | :--- |
| 0 | Cornflower Blue (`dg.cornflower_blue`) | `#4DAAE8` | `(0.3020, 0.6667, 0.9098)` |
| 1 | Coral Red | `#FF7259` | `(1.0000, 0.4471, 0.3490)` |
| 2 | Turquoise | `#3EC4AE` | `(0.2431, 0.7686, 0.6824)` |
| 3 | Purple | `#B66EFF` | `(0.7137, 0.4314, 1.0000)` |
| 4 | Magenta | `#F24B9D` | `(0.9490, 0.2941, 0.6157)` |
| 5 | Gold | `#F5C242` | `(0.9608, 0.7608, 0.2588)` |
| 6 | Light Gray (background clouds / ref) | `#CCCCCC` | `(0.8000, 0.8000, 0.8000)` |

### `degas.pastel_rainbow` (12 colors, `pastel_rainbow.mplstyle`)

`dg.pastel_rainbow` is a `(12, 3)` float NumPy array in `[0, 1]`. `dg.pastel_rainbow_alt` reorders the same 12 colors (`[0, 5, 3, 1, 7, 4, 2, 8, 6, 9, 10, 11]`) for higher hue separation between adjacent curves.

| Index | Hex | RGB `(0-255)` | Index | Hex | RGB `(0-255)` |
| :---: | :--- | :--- | :---: | :--- | :--- |
| 0 | `#DD3B35` | `(221, 59, 53)` | 6 | `#2B347C` | `(43, 52, 124)` |
| 1 | `#ED9D3F` | `(237, 157, 63)` | 7 | `#A4247C` | `(164, 36, 124)` |
| 2 | `#A5B485` | `(165, 180, 133)` | 8 | `#BAADD6` | `(186, 173, 214)` |
| 3 | `#3F946D` | `(63, 148, 109)` | 9 | `#A3DA85` | `(163, 218, 133)` |
| 4 | `#327A89` | `(50, 122, 137)` | 10 | `#889F7A` | `(136, 159, 122)` |
| 5 | `#2C73B2` | `(44, 115, 178)` | 11 | `#A8C0DD` | `(168, 192, 221)` |

### Additional Named Colors & Helpers

- Named scalars: `dg.cornflower_blue` (`(77, 170, 232)/255`, `#4DAAE8`), `dg.royal_purple` (`(120, 81, 169)/255`), `dg.mblue` (`"#00B4E7"`), `dg.mred` (`"#FF0000"`).
- Arcadia accents: `dg.indigo_arcadia` (`"#5088C5"`), `dg.blue_arcadia` (`"#282A49"`), `dg.green_arcadia` (`"#97CD78"`), `dg.yellow_arcadia` (`"#FFFDBD"`).
- Manipulation helpers:
  - `dg.lighter(rgb_tuple, f=1/3)` / `dg.darker(rgb_tuple, f=1/3)` — blend an RGB `[0, 1]` 3-tuple toward white or black.
  - `dg.cmap1D(rgb255_1, rgb255_2, N)` — list of `N` RGB `[0, 1]` tuples interpolated between two 0-255 colors.
  - `dg.color_sort(rgb_array, method="hue", reverse=False)` — sort RGB colors by `"hue"`, `"luminance"`, `"step"`, `"red"`, `"green"`, or `"blue"`.
  - `dg.make_cmap(hex_list, positions=None, name="custom", N=256)` — build a `LinearSegmentedColormap` from hex strings and optional stop positions.

---

## 3. Complete Colormap Catalog (41 Registered + Reversed `_r`)

All 41 colormaps in `degas/colormaps/` (Michael Waskom's `vernalis` + Fabio Crameri's Scientific Colour Maps 8) are registered with matplotlib on `import degas` and accessible as `dg.<name>` and `dg.<name>_r` (82 total).

| Category | Colormaps | When to Use |
| :--- | :--- | :--- |
| **Sequential (default)** | `vernalis` | Default image/heatmap colormap across all `degas` styles (deep purple -> teal -> bright lime-yellow). |
| **Sequential (multi-hue)** | `batlow`, `batlowK`, `batlowW`, `bamako`, `buda`, `glasgow`, `hawaii`, `imola`, `lajolla`, `lapaz`, `lipari`, `navia`, `naviaW`, `nuuk`, `tokyo` | Ordered positive quantities (intensity, density, probability, concentration). `batlow` is the flagship rainbow replacement; `*K` extends to near-black, `*W` extends to white. |
| **Sequential (near-monochrome)** | `acton`, `bilbao`, `davos`, `devon`, `grayC`, `oslo`, `turku` | Subtle backgrounds, microscopy, or grayscale-first figures (`grayC` is perceptually uniform grayscale; `oslo` works cleanly on dark slides). |
| **Diverging (light midpoint)** | `vik`, `roma`, `bam`, `broc`, `cork` | Signed deviations, correlations, residuals on **light paper** styles (`degas_figure`, `default`, `pastel_rainbow`). |
| **Diverging (dark midpoint)** | `berlin`, `lisbon`, `managua`, `tofino`, `vanimo` | Signed deviations on **`dark` presentation** slides where zero should recede into the dark background. |
| **Cyclic / Circular (`*O`)** | `bamO`, `brocO`, `corkO`, `romaO`, `vikO` | Periodic quantities (phase `[-pi, pi]`, angle `[0, 360)`, time-of-day, azimuth). |
| **Multi-sequential** | `bukavu`, `fes`, `oleron` | Elevation/bathymetry or two-regime fields with a sharp threshold transition. |

Additional custom `LinearSegmentedColormap` objects exposed directly on `degas` (pass by object, not string):
- `dg.blbkrd` — Blue-Black-Red diverging (`[dg.blue, 'k', dg.red]`), useful for signed fields on dark canvases.
- `dg.viridis_arcadia` — Arcadia indigo-navy-green-yellow sequential map.

---

## 4. Export Convention & Presentation Idioms

### Transparent High-DPI Exports at Compact Sizes
Always prefer **transparent exports** using `dg.better_savefig` at **`dpi=300` or `dpi=600`** with compact figure sizes (`2–3.5 in` per single panel):
```python
dg.better_savefig("figure.png", dpi=300, transparent=True)
dg.better_savefig("figure.png", dpi=600, transparent=True)
dg.better_savefig("figure.pdf", dpi=600, transparent=True)
```
Why small sizes + high DPI?
- Font sizes (`10pt` for papers, `12–15pt` for talks) stay proportionally clear without cranking fonts up to `28pt`.
- `dpi=300` at `3.5 × 3.5 in` produces a crisp `~1050 × 1050 px` transparent PNG; `dpi=600` produces `~2100 × 2100 px` for dense scatter or print figures.
- Transparent backgrounds (`savefig.transparent: True` in `base.mplstyle`) eliminate mismatched white/gray boxes on slides and manuscripts.

### Two-Tone Presentation Axis Labels (`dark` style)
To highlight the concept while keeping units or metric names visually secondary on a dark slide, use white (`#FFFFFF`) for the primary label and muted gray (`#8C8C8C`, matching tick labels) for the parenthetical subtitle:
```python
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker

# Multi-line two-tone x-label (e.g. "Chaoticity" / "(Largest Lyapunov Exponent)")
ax.set_xlabel("")
box = VPacker(
    children=[
        TextArea("Chaoticity", textprops=dict(color="#FFFFFF", size=15)),
        TextArea("(Largest Lyapunov Exponent)", textprops=dict(color="#8C8C8C", size=12.5)),
    ],
    align="center", pad=0, sep=3,
)
anchored = AnchoredOffsetbox(
    loc="upper center", child=box, frameon=False,
    bbox_to_anchor=(0.5, -0.15), bbox_transform=ax.transAxes, borderpad=0,
)
ax.add_artist(anchored)
```

### Bar Charts & Histograms (Relative Width & Spacing)
Matplotlib's default `ax.bar(..., width=0.8)` crowds bars too closely together and does not expose a `bar.width` rcParam. Across all `degas` styles (paper and presentation), prefer airy, borderless bars with a **`0.64` bar-to-pitch ratio** (~`0.36` inter-bar gap) and subtle outer horizontal padding (`xmargin=0.04`):
- **Preferred**: call `dg.plot_bar(heights, x=labels, ax=ax)` (uses `width=dg.BAR_WIDTH` = `0.64`, subtle aspect-preserving `corner_radius=1.5` pt rounded top corners with flat baseline, `edgecolor="none"`, `linewidth=0.0`, `xmargin=0.04`, and `bottom=0` for non-negative values). Pass `corner_radius=0` if sharp corners are needed.
- **Direct matplotlib**: `ax.bar(x, y, width=dg.BAR_WIDTH)` + `ax.margins(x=0.04)` (borderless fill is already set by `patch.edgecolor: none` and `patch.linewidth: 0.0` in all house styles).
- **Histograms**: pass `rwidth=dg.BAR_WIDTH` (`0.64`) to `ax.hist(...)` for matching inter-bar spacing.

### Background Scatter + Accent Fit on Dark Slides
When showing noisy observations behind a trend line on a dark slide:
- Plot points in soft translucent gray (`color="#CCCCCC", alpha=0.55, edgecolors="none", s=36`) so the cloud recedes cleanly.
- Plot the highlighted fit in `dg.cornflower_blue` (`#4DAAE8`) with `linewidth=2.5` and confidence interval `alpha=0.28`.

---

## 5. API Idioms & Gotchas

### Scoped Styling & Automatic Tick-Density Cap
`dg.set_style(name)` mutates global `plt.rcParams` (applying `base.mplstyle` followed by `<name>.mplstyle`) and activates `dg.set_max_ticks(5)` so linear axes, log axes, and colorbars show **3–5 clean major ticks** and always capture `0` whenever `0` is in range (pass `max_ticks=None` to disable, or `dg.limit_ticks(ax, max_ticks=4)` on a single axes).

When generating multiple figures with different themes in the same script or notebook, isolate each figure cleanly using `dg.style_context(name)`:
```python
with dg.style_context("dark"):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ...
    dg.better_savefig("dark_panel.png", dpi=300, transparent=True)
    plt.close(fig)
```

### `plot_err` Argument Order & RGB Tuples
1. Signature is `dg.plot_err(y, errs, x=[], color=(0, 0, 0), lightness=0.33/0.5, ax=None, fill_kwargs={}, **plot_kwargs)` — **`y` comes before `errs` and `x`**.
2. `color` must be an **RGB 3-tuple in `[0, 1]`** (e.g. `dg.blue`, `dg.cornflower_blue`, or `matplotlib.colors.to_rgb("#4DAAE8")`), because `plot_err` calls `dg.lighter(color, lightness)` numerically on the tuple. Passing a hex string raises a `TypeError`.
3. On `dark` style, `lighter()` blends toward white (which can look chalky) and the default `color=(0, 0, 0)` is invisible. Pass an explicit RGB color plus translucent `fill_kwargs`:
   ```python
   dg.plot_err(
       y, err, x=x, color=dg.cornflower_blue, ax=ax,
       fill_kwargs={"color": dg.cornflower_blue, "alpha": 0.28},
   )
   ```

### `colorline` Does Not Autoscale Axes
`dg.colorline(x, y, z=vals, cmap=dg.vernalis, norm=plt.Normalize(vmin, vmax))` adds a `LineCollection` to `plt.gca()`. Because matplotlib does not automatically expand axis limits for added collections, always follow with `ax.autoscale()` (or explicit `set_xlim`/`set_ylim`) and optionally attach a colorbar via `fig.colorbar(lc, ax=ax)`.

### `plot_linear_confidence` on Log Axes
`dg.plot_linear_confidence` checks `ax.get_xscale() == "log"` and `ax.get_yscale() == "log"` when fitting in log space. Set log scales on `ax` **before** calling `plot_linear_confidence` (or pass `logx=True`/`logy=True` explicitly). All values on a log axis must be strictly positive.

### Protecting the Plot Box Aspect Ratio (`dg.subplots` & `dg.set_box_aspect`)
Standard `plt.subplots(figsize=(3.5, 3.5))` sets the *outer canvas* size, so long y-labels, titles, or colorbars compress the actual plot box into a distorted rectangle. Use `dg.subplots(...)` to lock the **physical shape of the plot box itself**:
```python
# Exact 1:1 square plot box regardless of label widths or colorbars
fig, ax = dg.subplots(figsize=(3.5, 3.5))          # infers box_aspect=1.0
fig, axes = dg.subplots(2, 2, box_size=(2.6, 2.6)) # exact 2.6x2.6 in boxes with room for labels
dg.set_box_aspect(0.75, ax=ax)                     # lock existing axes to 4:3
```

### Inner-Tick Colorbar Layout (`dg.colorbar(..., inner_ticks=True)`)
`dg.colorbar(im, ax=ax, label="Residual", inner_ticks=True)` moves the ticks and tick labels to the figure-facing side of the colorbar strip, places the text label directly beside the outer side of the color strip (with foot facing inward), and dynamically measures rendered tick-label widths so even many-sig-fig ticks (`-0.00125`) never overlap the plot box.

### `fixed_aspect_ratio` Call Order
`dg.fixed_aspect_ratio(ratio, ax=ax, log=..., semilogx=..., semilogy=...)` computes the box aspect from the current `xlim`/`ylim`. Call it **after** all plotting and limit adjustments are complete.

### `draw_ellipse` Argument Order
The signature is `dg.draw_ellipse(points, ax=None, **kwargs)` (`points` first, `ax` second). Note that `dg.fit_ellipse(data)` returns `(width, height, angle_radians)`.
