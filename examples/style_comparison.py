"""Generate persistent visual comparisons of the degas paper and presentation styles.

Renders two standardized 2x2 galleries (regression + 95% CI, time series with
``plot_err``, bar chart with ``plot_bar``, and 2D field with ``dg.colorbar``):

1. ``compare_light_paper.png``        — ``degas_figure`` (high-contrast black box & outward
                                        ticks, larger 12pt axis labels, frameless legend &
                                        height-matched frameless colorbar)
2. ``compare_dark_presentation.png``  — ``dark`` (muted ``#8C8C8C`` box matching tick labels,
                                        white 15pt axis labels, bold cornflower_blue lead,
                                        frameless legend & height-matched frameless colorbar)

Pass ``--composite`` to composite transparent PNGs onto white (``#FFFFFF``) and dark
slide (``#161618``) backdrops for immediate side-by-side viewing.

Run from the repository root:
    uv run python examples/style_comparison.py
    uv run python examples/style_comparison.py --output-dir /tmp/degas_compare --composite
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

import degas as dg


def _make_sample_data() -> dict:
    """Deterministic datasets shared across both comparison figures."""
    rng = np.random.default_rng(11)
    x_log = np.geomspace(3e-3, 25.0, 85)
    y_log = np.clip(
        36.0 + 10.5 * np.log10(x_log) + rng.normal(0.0, 14.0, size=len(x_log)),
        1.0,
        80.0,
    )

    t = np.linspace(0.0, 8.0, 140)
    y1 = np.exp(-0.22 * t) * np.cos(1.8 * t)
    y2 = 0.7 * np.exp(-0.18 * t) * np.sin(1.5 * t - 0.5)
    err1 = 0.07 + 0.04 * rng.random(len(t))
    err2 = 0.06 + 0.03 * rng.random(len(t))

    bar_vals = [0.64, 1.0, 0.48, 0.90, 0.31, 0.12, 0.55, 0.96, 0.88, 0.98, 0.24, 0.85]
    bar_labels = [f"M{i + 1}" for i in range(len(bar_vals))]

    g = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(g, g)
    Z_seq = np.exp(-0.25 * ((X - 0.6) ** 2 + (Y + 0.4) ** 2)) + 0.65 * np.exp(
        -0.35 * ((X + 1.1) ** 2 + (Y - 0.9) ** 2)
    )
    Z_div = np.exp(-0.25 * (X**2 + Y**2)) * (np.sin(2.0 * X) * np.cos(2.0 * Y))

    return {
        "x_log": x_log,
        "y_log": y_log,
        "t": t,
        "y1": y1,
        "y2": y2,
        "err1": err1,
        "err2": err2,
        "bar_vals": bar_vals,
        "bar_labels": bar_labels,
        "Z_seq": Z_seq,
        "Z_div": Z_div,
    }


def _save_figure(out_path: str, bg_rgb: tuple[int, int, int] | None, dpi: int) -> None:
    """Export transparently via degas.better_savefig and optionally composite."""
    if bg_rgb is None:
        dg.better_savefig(out_path, dpi=dpi, transparent=True)
        return

    tmp_path = out_path + ".tmp.png"
    dg.better_savefig(tmp_path, dpi=dpi, transparent=True)
    arr = np.array(Image.open(tmp_path))
    bg = np.full((arr.shape[0], arr.shape[1], 3), bg_rgb, dtype=np.float32)
    alpha = arr[..., 3:4].astype(np.float32) / 255.0
    comp = (arr[..., :3].astype(np.float32) * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    Image.fromarray(comp).save(out_path)
    os.remove(tmp_path)


def render_light_gallery(
    data: dict,
    out_path: str,
    composite: bool = False,
    dpi: int = 300,
) -> str:
    """Render the 2x2 light paper gallery under degas_figure."""
    with dg.style_context("degas_figure"):
        fig, axes = dg.subplots(2, 2, box_size=(2.6, 2.6))

        # 1. Regression + 95% CI
        ax = axes[0, 0]
        ax.set_xscale("log")
        ax.scatter(
            data["x_log"],
            data["y_log"],
            s=18,
            color=dg.gray,
            alpha=0.65,
            edgecolors="none",
            zorder=2,
        )
        dg.plot_linear_confidence(
            data["x_log"],
            data["y_log"],
            ax=ax,
            show_pi=False,
            extend_fraction=0.1,
            color=dg.blue,
            ci_kwargs={"color": dg.blue, "alpha": 0.22},
        )
        ax.set_xlabel("Largest Lyapunov Exponent")
        ax.set_ylabel("Forecast Error (sMAPE)")
        ax.set_title("Regression + 95% CI")

        # 2. Time series (plot_err)
        ax = axes[0, 1]
        dg.plot_err(data["y1"], data["err1"], x=data["t"], color=dg.blue, ax=ax, label="Channel A")
        dg.plot_err(data["y2"], data["err2"], x=data["t"], color=dg.red, ax=ax, label="Channel B")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Signal (mV)")
        ax.set_title("Time Series (plot_err)")
        ax.legend(fontsize=8.5)

        # 3. Bar chart (dg.plot_bar)
        ax = axes[1, 0]
        dg.plot_bar(data["bar_vals"], ax=ax, color=dg.blue)
        ax.set_xticks(range(0, len(data["bar_vals"]), 2))
        ax.set_xticklabels(
            [data["bar_labels"][i] for i in range(0, len(data["bar_vals"]), 2)]
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("Normalized Score")
        ax.set_title("Bar Chart (width=0.64)")

        # 4. 2D Field (vernalis default) with frameless height-matched colorbar
        ax = axes[1, 1]
        im = ax.imshow(data["Z_seq"], extent=(-3, 3, -3, 3))
        dg.colorbar(im, ax=ax, label="Intensity")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("2D Field (vernalis default)")

        _save_figure(out_path, (255, 255, 255) if composite else None, dpi=dpi)
        plt.close(fig)

    return out_path


def render_dark_gallery(
    data: dict,
    out_path: str,
    composite: bool = False,
    dpi: int = 300,
) -> str:
    """Render the 2x2 dark presentation gallery under dark.mplstyle."""
    with dg.style_context("dark"):
        fig, axes = dg.subplots(2, 2, box_size=(2.9, 2.9))

        # 1. Regression + 95% CI
        ax = axes[0, 0]
        ax.set_xscale("log")
        ax.scatter(
            data["x_log"],
            data["y_log"],
            s=30,
            color="#CCCCCC",
            alpha=0.55,
            edgecolors="none",
            zorder=2,
        )
        dg.plot_linear_confidence(
            data["x_log"],
            data["y_log"],
            ax=ax,
            show_pi=False,
            extend_fraction=0.1,
            color=dg.cornflower_blue,
            linewidth=2.5,
            ci_kwargs={"color": dg.cornflower_blue, "alpha": 0.28},
        )
        ax.set_xlabel("Chaoticity")
        ax.set_ylabel("Forecast Error")
        ax.set_title("Regression + 95% CI")

        # 2. Time series (plot_err)
        ax = axes[0, 1]
        c_blue = tuple(dg.cornflower_blue)
        c_coral = (1.0, 0.4471, 0.3490)
        dg.plot_err(
            data["y1"],
            data["err1"],
            x=data["t"],
            color=c_blue,
            ax=ax,
            linewidth=2.4,
            fill_kwargs={"color": c_blue, "alpha": 0.28},
            label="Channel A",
        )
        dg.plot_err(
            data["y2"],
            data["err2"],
            x=data["t"],
            color=c_coral,
            ax=ax,
            linewidth=2.4,
            fill_kwargs={"color": c_coral, "alpha": 0.28},
            label="Channel B",
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Signal (mV)")
        ax.set_title("Time Series (plot_err)")
        ax.legend(fontsize=10)

        # 3. Bar chart (dg.plot_bar)
        ax = axes[1, 0]
        dg.plot_bar(data["bar_vals"], ax=ax, color=dg.cornflower_blue)
        ax.set_xticks(range(0, len(data["bar_vals"]), 2))
        ax.set_xticklabels(
            [data["bar_labels"][i] for i in range(0, len(data["bar_vals"]), 2)]
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Bar Chart (width=0.64)")

        # 4. Diverging field (berlin) with frameless height-matched colorbar
        ax = axes[1, 1]
        vmax = float(np.max(np.abs(data["Z_div"])))
        im = ax.imshow(
            data["Z_div"],
            cmap="berlin",
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            extent=(-3, 3, -3, 3),
        )
        dg.colorbar(im, ax=ax, label="Residual")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Diverging Field (berlin)")

        _save_figure(out_path, (22, 22, 24) if composite else None, dpi=dpi)
        plt.close(fig)

    return out_path


def render_colorbar_comparison(
    data: dict,
    out_dir: str,
    composite: bool = False,
    dpi: int = 300,
) -> list[str]:
    """Compare outer-tick vs inner-tick dynamic-pad colorbars on light and dark styles."""
    saved = []
    for style_name, bg_rgb, cmap, field_scale, label in (
        ("degas_figure", (255, 255, 255), "vernalis", 1.0, "Intensity"),
        ("dark", (22, 22, 24), "berlin", 0.00125, "Residual"),
    ):
        with dg.style_context(style_name):
            fig, axes = dg.subplots(1, 2, box_size=(2.7, 2.7))
            for ax, inner, title in (
                (axes[0], False, "Outer ticks (default)"),
                (axes[1], True, "Inner ticks (dynamic pad)"),
            ):
                if style_name == "dark":
                    Z = field_scale * data["Z_div"]
                    vmax = float(np.max(np.abs(Z)))
                    im = ax.imshow(
                        Z,
                        cmap=cmap,
                        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
                        extent=(-3, 3, -3, 3),
                    )
                else:
                    im = ax.imshow(
                        field_scale * data["Z_seq"],
                        cmap=cmap,
                        extent=(-3, 3, -3, 3),
                    )
                dg.colorbar(im, ax=ax, label=label, inner_ticks=inner)
                ax.set_xlabel("x (mm)")
                ax.set_ylabel("y (mm)")
                ax.set_title(title)

            fname = f"compare_colorbar_{'light' if style_name == 'degas_figure' else 'dark'}.png"
            out_path = os.path.join(out_dir, fname)
            _save_figure(out_path, bg_rgb if composite else None, dpi=dpi)
            plt.close(fig)
            saved.append(out_path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render degas light paper and dark presentation comparison galleries."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
        help="Directory where generated PNGs are saved (default: examples/output).",
    )
    parser.add_argument(
        "--composite",
        action="store_true",
        help="Composite transparent PNGs onto white/dark backdrops for quick viewing.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Export DPI (default: 300).",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = _make_sample_data()

    outputs = [
        render_light_gallery(
            data,
            os.path.join(args.output_dir, "compare_light_paper.png"),
            composite=args.composite,
            dpi=args.dpi,
        ),
        render_dark_gallery(
            data,
            os.path.join(args.output_dir, "compare_dark_presentation.png"),
            composite=args.composite,
            dpi=args.dpi,
        ),
    ]
    outputs.extend(
        render_colorbar_comparison(
            data,
            args.output_dir,
            composite=args.composite,
            dpi=args.dpi,
        )
    )

    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
