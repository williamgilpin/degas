"""Demonstrate core degas scientific plotting, colormap, and export utilities.

Covers:
  - ``plot_err`` (symmetric and asymmetric error bands)
  - ``plot_linear_confidence`` (log-log linear regression with 95% CI & PI)
  - ``plot_splity`` (dual y-axis plotting with clean z-order)
  - ``colorline`` (gradient-colored trajectory with ``vernalis``)
  - ``fixed_aspect_ratio``, ``dim_axes``, ``xyz_to_image``
  - Scientific colormaps (``vernalis``, ``batlow``, ``vik``)
  - ``better_savefig`` (tight high-DPI export)

Run from the repository root:
    uv run python examples/scientific_helpers.py
    uv run python examples/scientific_helpers.py --output-dir /tmp/degas_examples
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import degas as dg


def build_helpers_gallery(out_dir: str) -> str:
    """Create a 2x3 multi-panel gallery showcasing degas plotting helpers."""
    rng = np.random.default_rng(42)
    style_file = os.path.join(dg.data_path, "degas_figure.mplstyle")

    with plt.style.context(style_file):
        fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.8), constrained_layout=True)

        # 1. plot_err: symmetric and asymmetric error bands
        ax = axes[0, 0]
        x = np.linspace(0.0, 6.0, 120)
        y_sym = np.exp(-0.25 * x) * np.cos(2.0 * x)
        err_sym = 0.08 + 0.04 * np.abs(y_sym)
        y_asym = 0.55 * np.exp(-0.2 * x) * np.sin(1.7 * x) - 0.5
        err_asym = np.column_stack(
            [
                0.05 + 0.06 * np.exp(-0.3 * x),
                0.10 + 0.08 * np.exp(-0.2 * x),
            ]
        )
        dg.plot_err(y_sym, err_sym, x=x, color=dg.blue, ax=ax, linewidth=1.6, label="Symmetric")
        dg.plot_err(y_asym, err_asym, x=x, color=dg.red, ax=ax, linewidth=1.6, label="Asymmetric")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Response")
        ax.set_title("plot_err")
        ax.legend(frameon=False, fontsize=8)

        # 2. plot_linear_confidence on log-log axes
        ax = axes[0, 1]
        x_log = np.geomspace(1.0, 100.0, 35)
        y_log = 2.4 * (x_log**0.72) * np.exp(rng.normal(0.0, 0.22, size=len(x_log)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.scatter(x_log, y_log, s=18, color=dg.turquoise, alpha=0.85, zorder=3)
        _, slope, _, r2, _ = dg.plot_linear_confidence(
            x_log,
            y_log,
            ax=ax,
            return_model=True,
            extend_fraction=0.1,
            color=dg.blue,
            linewidth=1.6,
            ci_kwargs={"color": dg.blue, "alpha": 0.22},
            pi_kwargs={"color": dg.blue, "alpha": 0.09},
        )
        ax.set_xlabel("System size N")
        ax.set_ylabel("Scaling metric")
        ax.set_title(f"plot_linear_confidence (slope={slope:.2f}, $R^2$={r2:.2f})")

        # 3. plot_splity: dual y-axes
        ax = axes[0, 2]
        t = np.linspace(0.0, 24.0, 120)
        temp_c = 18.0 + 7.5 * np.sin((t - 8.0) * np.pi / 12.0)
        rate = 120.0 * np.exp(0.09 * (temp_c - 18.0))
        ax, ax2 = dg.plot_splity(
            t,
            temp_c,
            rate,
            ax=ax,
            kwargs1={"color": dg.blue, "linewidth": 1.8},
            kwargs2={"color": dg.magenta, "linewidth": 1.8, "linestyle": "--"},
        )
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Temperature (°C)", color=dg.blue)
        ax.tick_params(axis="y", colors=dg.blue)
        ax2.set_ylabel("Reaction rate (s$^{-1}$)", color=dg.magenta)
        ax2.tick_params(axis="y", colors=dg.magenta)
        ax.set_title("plot_splity")

        # 4. colorline + fixed_aspect_ratio + vernalis colormap
        ax = axes[1, 0]
        theta = np.linspace(0.0, 6.0 * np.pi, 400)
        radius = np.linspace(0.2, 2.2, len(theta))
        xs = radius * np.cos(theta)
        ys = radius * np.sin(theta)
        plt.sca(ax)
        lc = dg.colorline(
            xs,
            ys,
            z=theta,
            cmap=dg.vernalis,
            norm=plt.Normalize(theta.min(), theta.max()),
            linewidth=2.2,
        )
        ax.autoscale()
        dg.fixed_aspect_ratio(1.0, ax=ax)
        dg.colorbar(lc, ax=ax, label="Phase (rad)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("colorline (vernalis)")

        # 5. Diverging scientific colormap (vik) with symmetric zero center
        ax = axes[1, 1]
        u = np.linspace(-2.5, 2.5, 90)
        U, V = np.meshgrid(u, u)
        field = np.exp(-0.4 * (U**2 + V**2)) * (U**2 - V**2)
        vmax = float(np.max(np.abs(field)))
        im = ax.imshow(
            field,
            cmap=dg.vik,
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            extent=(-2.5, 2.5, -2.5, 2.5),
        )
        dg.colorbar(im, ax=ax, label="Anomaly")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Diverging cmap (vik)")

        # 6. xyz_to_image + batlow + dim_axes
        ax = axes[1, 2]
        gx, gy = np.meshgrid(np.linspace(-2.0, 2.0, 24), np.linspace(-2.0, 2.0, 24))
        keep = rng.random(gx.size) > 0.25  # simulate missing samples
        sx, sy = gx.ravel()[keep], gy.ravel()[keep]
        sz = np.exp(-0.5 * (sx**2 + sy**2)) + 0.35 * np.sin(2.0 * sx) * np.cos(1.5 * sy)
        img, extent, _, _ = dg.xyz_to_image(sx, sy, sz, method="linear")
        im = ax.imshow(img, cmap=dg.batlow, extent=extent)
        dg.colorbar(im, ax=ax, label="Interpolated z")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("xyz_to_image (batlow) + dim_axes")
        dg.dim_axes(ax=ax, alpha=0.65)

        out_path = os.path.join(out_dir, "scientific_helpers.png")
        dg.better_savefig(out_path, dpi=300, transparent=True)
        plt.close(fig)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run degas scientific plotting helper examples."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
        help="Directory where generated PNGs are saved (default: examples/output).",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = build_helpers_gallery(args.output_dir)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
