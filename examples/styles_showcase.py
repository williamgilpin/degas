"""Showcase all degas house styles: default, degas_figure, pastel_rainbow, and dark.

Each style is applied in an isolated ``plt.style.context`` block using
``degas.data_path`` so style settings do not leak between figures. All figures
are exported transparently at 300 DPI with ``degas.better_savefig``.

Run from the repository root:
    uv run python examples/styles_showcase.py
    uv run python examples/styles_showcase.py --output-dir /tmp/degas_examples
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

import degas as dg

STYLES = ("default", "degas_figure", "pastel_rainbow", "dark")


def _damped_harmonics(n_curves: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic damped wave curves for style comparisons."""
    t = np.linspace(0.0, 10.0, 250)
    curves = np.array(
        [
            np.exp(-0.14 * t) * np.sin(1.4 * t + 0.45 * k) + 0.18 * k
            for k in range(n_curves)
        ]
    )
    return t, curves


def render_style_card(style_name: str, out_dir: str) -> str:
    """Render a standalone multi-series line plot in a single degas house style."""
    style_file = os.path.join(dg.data_path, f"{style_name}.mplstyle")
    n_curves = 10 if style_name == "pastel_rainbow" else 6
    t, curves = _damped_harmonics(n_curves)
    figsize = (3.5, 3.2) if style_name == "dark" else (3.4, 2.6)

    with plt.style.context(style_file):
        fig, ax = plt.subplots(figsize=figsize)
        lw = 2.4 if style_name == "dark" else 1.5
        for idx, y in enumerate(curves):
            ax.plot(t, y, linewidth=lw, label=f"Mode {idx + 1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.set_title(f"degas style: {style_name}")
        ax.legend(
            ncol=2 if n_curves > 6 else 1,
            fontsize=8 if style_name != "dark" else 9,
            loc="upper right",
            frameon=False,
        )
        out_path = os.path.join(out_dir, f"style_{style_name}.png")
        dg.better_savefig(out_path, dpi=300, transparent=True)
        plt.close(fig)

    return out_path


def render_dark_presentation_plot(out_dir: str) -> str:
    """Render a compact transparent dark presentation figure (scatter + log-x fit)."""
    style_file = os.path.join(dg.data_path, "dark.mplstyle")
    rng = np.random.default_rng(11)

    with plt.style.context(style_file):
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.set_xscale("log")

        x = np.geomspace(3e-3, 25.0, 95)
        y = np.clip(
            36.0 + 10.5 * np.log10(x) + rng.normal(0.0, 14.0, size=len(x)),
            0.5,
            80.0,
        )
        ax.scatter(
            x,
            y,
            s=36,
            color="#CCCCCC",
            alpha=0.55,
            edgecolors="none",
            zorder=2,
        )
        dg.plot_linear_confidence(
            x,
            y,
            ax=ax,
            show_pi=False,
            extend_fraction=0.15,
            color=dg.cornflower_blue,
            linewidth=2.6,
            ci_kwargs={"color": dg.cornflower_blue, "alpha": 0.28, "edgecolor": "none"},
        )
        ax.set_yticks([0, 25, 50, 75])
        ax.set_ylim(-3, 84)
        ax.set_ylabel("Forecast Error")

        # Two-tone presentation x-label: white primary title + muted gray subtitle
        ax.set_xlabel("")
        label_box = VPacker(
            children=[
                TextArea("Chaoticity", textprops=dict(color="#FFFFFF", size=15)),
                TextArea(
                    "(Largest Lyapunov Exponent)",
                    textprops=dict(color="#8C8C8C", size=12),
                ),
            ],
            align="center",
            pad=0,
            sep=3,
        )
        anchored = AnchoredOffsetbox(
            loc="upper center",
            child=label_box,
            frameon=False,
            bbox_to_anchor=(0.5, -0.17),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        ax.add_artist(anchored)

        out_path = os.path.join(out_dir, "dark_presentation.png")
        dg.better_savefig(out_path, dpi=300, transparent=True)
        plt.close(fig)

    return out_path


def render_dark_showcase(out_dir: str) -> str:
    """Render a two-panel transparent presentation figure using the dark style."""
    style_file = os.path.join(dg.data_path, "dark.mplstyle")
    rng = np.random.default_rng(7)

    with plt.style.context(style_file):
        fig, (ax_line, ax_map) = plt.subplots(
            1, 2, figsize=(7.2, 3.3), constrained_layout=True
        )

        # Left panel: error bands using presentation cornflower_blue and coral
        x = np.linspace(0.0, 8.0, 140)
        y1 = np.exp(-0.22 * x) * np.cos(1.8 * x)
        y2 = 0.7 * np.exp(-0.18 * x) * np.sin(1.5 * x - 0.5)
        err1 = 0.07 + 0.04 * rng.random(len(x))
        err2 = 0.06 + 0.03 * rng.random(len(x))

        c_blue = tuple(dg.cornflower_blue)
        c_coral = (1.0, 0.4471, 0.3490)
        dg.plot_err(
            y1,
            err1,
            x=x,
            color=c_blue,
            ax=ax_line,
            linewidth=2.4,
            fill_kwargs={"color": c_blue, "alpha": 0.28, "edgecolor": "none"},
            label="Channel A",
        )
        dg.plot_err(
            y2,
            err2,
            x=x,
            color=c_coral,
            ax=ax_line,
            linewidth=2.4,
            fill_kwargs={"color": c_coral, "alpha": 0.28, "edgecolor": "none"},
            label="Channel B",
        )
        ax_line.set_xlabel("Time (ms)")
        ax_line.set_ylabel("Signal (mV)")
        ax_line.set_title("Error Bands")
        ax_line.legend(loc="upper right", frameon=False, fontsize=10)

        # Right panel: signed 2D field with dark-center diverging colormap (berlin)
        grid = np.linspace(-3.0, 3.0, 120)
        X, Y = np.meshgrid(grid, grid)
        Z = np.exp(-0.25 * (X**2 + Y**2)) * (np.sin(2.0 * X) * np.cos(2.0 * Y))
        vmax = float(np.max(np.abs(Z)))
        im = ax_map.imshow(
            Z,
            cmap="berlin",
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            extent=(-3.0, 3.0, -3.0, 3.0),
        )
        ax_map.set_xlabel("x (mm)")
        ax_map.set_ylabel("y (mm)")
        ax_map.set_title("Diverging Field (berlin)")
        dg.colorbar(im, ax=ax_map, label="Residual")

        out_path = os.path.join(out_dir, "dark_showcase.png")
        dg.better_savefig(out_path, dpi=300, transparent=True)
        plt.close(fig)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Showcase degas matplotlib styles.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
        help="Directory where generated PNGs are saved (default: examples/output).",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    saved = [render_style_card(name, args.output_dir) for name in STYLES]
    saved.append(render_dark_presentation_plot(args.output_dir))
    saved.append(render_dark_showcase(args.output_dir))

    for path in saved:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
