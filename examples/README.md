# `degas` Examples

Self-contained, headless-safe Python scripts demonstrating `degas` house styles, scientific colormaps, and plotting helpers.

## Available Scripts

- **`styles_showcase.py`** — Renders one multi-series figure in each house style (`default`, `degas_figure`, `pastel_rainbow`, `dark`) using isolated `plt.style.context(...)` blocks, a compact transparent presentation scatter + confidence-band figure (`dark_presentation.png`), and a two-panel dark presentation showcase (`dark_showcase.png`) pairing `dark.mplstyle` with `plot_err` and the `berlin` dark-center diverging colormap. All figures export transparently at 300 DPI via `dg.better_savefig`.
- **`style_comparison.py`** — Persistent visual comparison script rendering 2x2 galleries (regression + CI, `plot_err`, `plot_bar`, 2D field) for current `degas_figure`, `+ High-Impact No-Regret Upgrades`, `+ Paper-Figure Aesthetic Upgrades`, and `dark`. Pass `--composite` to preview transparent exports over white/dark backdrops.
- **`scientific_helpers.py`** — Builds a 2x3 publication gallery (`scientific_helpers.png`) covering `plot_err` (symmetric & asymmetric bands), `plot_linear_confidence` on log-log axes, `plot_splity`, `colorline` with `vernalis` and `fixed_aspect_ratio`, diverging `vik` heatmaps, `xyz_to_image` with `batlow`, `dim_axes`, and `better_savefig`.

## Running the Examples

From the repository root using `uv`:

```bash
uv run python examples/styles_showcase.py
uv run python examples/scientific_helpers.py
```

By default, figures are saved to `examples/output/`. Pass `--output-dir` to write elsewhere:

```bash
uv run python examples/styles_showcase.py --output-dir /tmp/degas_examples
uv run python examples/scientific_helpers.py --output-dir /tmp/degas_examples
```
