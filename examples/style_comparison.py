"""Generate persistent visual style comparisons for degas paper and presentation styles.

Renders four standardized 2x2 galleries (regression + 95% CI, time series with
``plot_err``, bar chart with ``plot_bar``, and 2D field) so style changes can be
inspected side by side:

1. ``compare_light_baseline.png``      — Current ``degas_figure`` (high-contrast black axes/ticks)
2. ``compare_light_no_regret.png``     — + High-Impact No-Regret Upgrades
                                         (editable PDF/SVG fonts, frameless legends,
                                         0.75pt spines / 1.5pt lines, borderless CI/error bands)
3. ``compare_light_tufte_paper.png``   — + Paper-Figure Aesthetic Upgrades
                                         (despined top/right on 1D plots, 3.5pt outward ticks,
                                         preserving high-contrast black spines & ticks)
4. ``compare_dark_presentation.png``   — Current ``dark`` presentation style
                                         (lower-contrast gray box/ticks, white axis labels)

By default, transparent PNGs are also composited onto white (``#FFFFFF``) or dark
slide (``#161618``) backgrounds when ``--composite`` is passed so differences are
immediately legible in any image viewer.

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

NO_REGRET_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "legend.frameon": False,
    "axes.linewidth": 0.75,
    "lines.linewidth": 1.5,
}

TUFTE_PAPER_RC = {
    **NO_REGRET_RC,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
}


def _make_sample_data() -> dict:
    """Deterministic datasets shared across all comparison figures."""
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
    suptitle: str,
    extra_rc: dict | None = None,
    borderless_bands: bool = False,
    keep_heatmap_box: bool = False,
    composite: bool = False,
    dpi: int = 300,
) -> str:
    """Render the 2x2 light paper gallery under degas_figure + optional rc overrides."""
    style_chain = [
        os.path.join(dg.data_path, "base.mplstyle"),
        os.path.join(dg.data_path, "degas_figure.mplstyle"),
    ]
    if extra_rc:
        style_chain.append(extra_rc)

    band_edge = {"edgecolor": "none"} if borderless_bands else {}

    with plt.style.context(style_chain):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
        fig.suptitle(suptitle, fontsize=11.5, fontweight="bold")

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
            ci_kwargs={"color": dg.blue, "alpha": 0.22, **band_edge},
        )
        ax.set_yticks([0, 25, 50, 75])
        ax.set_xlabel("Largest Lyapunov Exponent")
        ax.set_ylabel("Forecast Error (sMAPE)")
        ax.set_title("Regression + 95% CI")

        # 2. Time series (plot_err)
        ax = axes[0, 1]
        dg.plot_err(
            data["y1"],
            data["err1"],
            x=data["t"],
            color=dg.blue,
            ax=ax,
            fill_kwargs=dict(band_edge),
            label="Channel A",
        )
        dg.plot_err(
            data["y2"],
            data["err2"],
            x=data["t"],
            color=dg.red,
            ax=ax,
            fill_kwargs=dict(band_edge),
            label="Channel B",
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Signal (mV)")
        ax.set_title("Time Series (plot_err)")
        # Show default legend styling from the active rcParams (framed vs frameless)
        if not plt.rcParams["legend.frameon"]:
            ax.legend(fontsize=8)
        else:
            leg = ax.legend(fontsize=8)
            leg.get_frame().set_edgecolor("#CCCCCC")
            leg.get_frame().set_linewidth(0.8)

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

        # 4. 2D Field (vernalis default)
        ax = axes[1, 1]
        im = ax.imshow(data["Z_seq"], extent=(-3, 3, -3, 3))
        if keep_heatmap_box:
            for side in ("top", "right"):
                ax.spines[side].set_visible(True)
        cbar = fig.colorbar(im, ax=ax, shrink=0.88)
        cbar.set_label("Intensity")
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
    style_chain = [
        os.path.join(dg.data_path, "base.mplstyle"),
        os.path.join(dg.data_path, "dark.mplstyle"),
    ]

    with plt.style.context(style_chain):
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.4), constrained_layout=True)
        fig.suptitle(
            "Dark Presentation Style (dark.mplstyle — muted gray box & ticks)",
            fontsize=13,
            color="#FFFFFF",
        )

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
            ci_kwargs={
                "color": dg.cornflower_blue,
                "alpha": 0.28,
                "edgecolor": "none",
            },
        )
        ax.set_yticks([0, 25, 50, 75])
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
            fill_kwargs={"color": c_blue, "alpha": 0.28, "edgecolor": "none"},
            label="Channel A",
        )
        dg.plot_err(
            data["y2"],
            data["err2"],
            x=data["t"],
            color=c_coral,
            ax=ax,
            linewidth=2.4,
            fill_kwargs={"color": c_coral, "alpha": 0.28, "edgecolor": "none"},
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

        # 4. Diverging field (berlin)
        ax = axes[1, 1]
        vmax = float(np.max(np.abs(data["Z_div"])))
        im = ax.imshow(
            data["Z_div"],
            cmap="berlin",
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            extent=(-3, 3, -3, 3),
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.88)
        cbar.set_label("Residual", color="#FFFFFF")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Diverging Field (berlin)")

        _save_figure(out_path, (22, 22, 24) if composite else None, dpi=dpi)
        plt.close(fig)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render degas light/dark style comparison galleries."
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
            os.path.join(args.output_dir, "compare_light_baseline.png"),
            suptitle="1. Latest degas_figure (black axes & ticks, 7pt inward ticks)",
            extra_rc=None,
            borderless_bands=False,
            composite=args.composite,
            dpi=args.dpi,
        ),
        render_light_gallery(
            data,
            os.path.join(args.output_dir, "compare_light_no_regret.png"),
            suptitle="2. + High-Impact No-Regret Upgrades (0.75pt box, 1.5pt lines, frameless legend, borderless bands)",
            extra_rc=NO_REGRET_RC,
            borderless_bands=True,
            composite=args.composite,
            dpi=args.dpi,
        ),
        render_light_gallery(
            data,
            os.path.join(args.output_dir, "compare_light_tufte_paper.png"),
            suptitle="3. + Paper-Figure Aesthetic Upgrades (despined 1D plots, 3.5pt outward black ticks)",
            extra_rc=TUFTE_PAPER_RC,
            borderless_bands=True,
            keep_heatmap_box=True,
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

    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
