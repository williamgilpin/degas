"""Unit tests for degas plotting and utility functions."""

import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import degas as dg


class TestPlottingAndUtilities(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_plot_bar_default_width_and_spacing(self):
        heights = np.array([0.5, 0.9, 0.35, 0.8, 0.2])
        fig, ax = plt.subplots()
        out_ax = dg.plot_bar(heights, ax=ax)
        self.assertIs(out_ax, ax)
        self.assertEqual(len(ax.patches), len(heights))
        fig.canvas.draw()
        for patch in ax.patches:
            self.assertAlmostEqual(patch.get_width(), dg.BAR_WIDTH, places=6)
            self.assertIn(matplotlib.path.Path.CURVE4, patch.get_path().codes)
        self.assertEqual(ax.get_ylim()[0], 0.0)

        # corner_radius=0 falls back to sharp corners
        fig2, ax2 = plt.subplots()
        dg.plot_bar(heights, ax=ax2, corner_radius=0)
        fig2.canvas.draw()
        self.assertNotIn(matplotlib.path.Path.CURVE4, ax2.patches[0].get_path().codes)

    def test_colorbar_frameless_and_aligned(self):
        fig, ax = plt.subplots()
        im = ax.imshow(np.linspace(-0.85, 0.85, 16).reshape(4, 4))
        cbar = dg.colorbar(im, ax=ax, label="Value")
        self.assertFalse(cbar.outline.get_visible())
        self.assertEqual(cbar.ax.get_ylabel(), "Value")
        self.assertEqual(cbar.ax.yaxis.label.get_rotation(), 270.0)
        fig.canvas.draw()
        cticks = [t for t in cbar.get_ticks() if -0.85 <= t <= 0.85]
        self.assertTrue(3 <= len(cticks) <= 5, f"Unexpected colorbar ticks: {cticks}")
        self.assertTrue(any(abs(t) < 1e-9 for t in cticks), f"0 missing from colorbar ticks: {cticks}")

    def test_colorbar_inner_ticks_and_dynamic_spacing(self):
        fig, (ax_small, ax_wide) = dg.subplots(1, 2, box_size=(2.5, 2.5))
        im_small = ax_small.imshow(np.linspace(0.0, 1.0, 16).reshape(4, 4))
        im_wide = ax_wide.imshow(np.linspace(-0.00125, 0.00125, 16).reshape(4, 4))

        cbar_small = dg.colorbar(im_small, ax=ax_small, label="A", inner_ticks=True)
        cbar_wide = dg.colorbar(im_wide, ax=ax_wide, label="B", inner_ticks=True)
        self.assertEqual(cbar_small.ax.yaxis.get_ticks_position(), "left")
        self.assertEqual(cbar_small.ax.yaxis.get_label_position(), "right")
        self.assertEqual(cbar_wide.ax.yaxis.get_ticks_position(), "left")
        self.assertEqual(cbar_wide.ax.yaxis.get_label_position(), "right")

        # Wide sig-fig tick labels should produce a wider dynamic gap than [0, 1]
        fig.canvas.draw()
        gap_small = cbar_small.ax.get_position().x0 - ax_small.get_position().x1
        gap_wide = cbar_wide.ax.get_position().x0 - ax_wide.get_position().x1
        self.assertGreater(gap_wide, gap_small)

    def test_subplots_and_set_box_aspect(self):
        fig, ax = dg.subplots(figsize=(4.0, 3.0))
        self.assertAlmostEqual(ax.get_box_aspect(), 0.75, places=6)

        fig2, axes = dg.subplots(2, 2, box_size=(2.5, 2.5))
        for sub_ax in axes.ravel():
            self.assertAlmostEqual(sub_ax.get_box_aspect(), 1.0, places=6)
        dg.set_box_aspect(0.5, ax=axes[0, 0])
        self.assertAlmostEqual(axes[0, 0].get_box_aspect(), 0.5, places=6)

    def test_limit_ticks_preserves_zero_and_caps_count(self):
        fig, ax = plt.subplots()
        ax.plot([-3.0, 3.0], [-0.75, 1.05])
        dg.limit_ticks(ax, max_ticks=5)
        fig.canvas.draw()
        for ticks, (lo, hi) in ((ax.get_xticks(), ax.get_xlim()), (ax.get_yticks(), ax.get_ylim())):
            visible = [t for t in ticks if lo <= t <= hi]
            self.assertTrue(3 <= len(visible) <= 5, f"Unexpected ticks: {visible}")
            self.assertTrue(any(abs(t) < 1e-9 for t in visible), f"0 missing: {visible}")
        with self.assertRaises(ValueError):
            dg.limit_ticks(ax, axis="z")

    def test_plot_err_symmetric_and_asymmetric(self):
        x = np.linspace(0, 1, 20)
        y = x**2
        err = np.full_like(x, 0.1)

        fig, ax = plt.subplots()
        out_ax = dg.plot_err(y, err, x=x, color=(0.2, 0.4, 0.8), ax=ax)
        self.assertIs(out_ax, ax)
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(len(ax.collections), 1)

        # Asymmetric errors with default x
        fig2, ax2 = plt.subplots()
        asym_err = np.column_stack([err, 2 * err])
        dg.plot_err(y, asym_err, ax=ax2)
        self.assertEqual(len(ax2.lines), 1)

    def test_plot_splity(self):
        x = np.linspace(0, 2 * np.pi, 30)
        fig, ax = plt.subplots()
        ax1, ax2 = dg.plot_splity(
            x,
            np.sin(x),
            100 * np.cos(x),
            ax=ax,
            kwargs1={"color": "blue"},
            kwargs2={"color": "red"},
        )
        self.assertIs(ax1, ax)
        self.assertEqual(len(ax1.lines), 1)
        self.assertEqual(len(ax2.lines), 1)

    def test_fixed_aspect_ratio(self):
        fig, ax = plt.subplots()
        ax.plot([1, 10], [1, 100])
        ax.set_xlim(1, 10)
        ax.set_ylim(1, 100)
        dg.fixed_aspect_ratio(1.0, ax=ax)
        self.assertGreater(ax.get_aspect(), 0)

        dg.fixed_aspect_ratio(1.0, ax=ax, log=True)
        dg.fixed_aspect_ratio(1.0, ax=ax, semilogx=True)
        dg.fixed_aspect_ratio(1.0, ax=ax, semilogy=True)

    def test_compute_pca_and_fit_ellipse(self):
        rng = np.random.default_rng(42)
        pts = rng.normal(size=(200, 2)) @ np.array([[3.0, 0.0], [0.0, 1.0]])
        eigvals, eigvecs, explained = dg.compute_pca(pts)
        self.assertEqual(eigvals.shape, (2,))
        self.assertEqual(eigvecs.shape, (2, 2))
        self.assertAlmostEqual(float(np.sum(explained)), 1.0, places=6)
        self.assertGreater(eigvals[0], eigvals[1])

        width, height, angle = dg.fit_ellipse(pts)
        self.assertGreater(width, height)
        self.assertTrue(np.isfinite(angle))

    def test_plot_cross(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=50)
        pts = np.column_stack([x, 2 * x + 0.2 * rng.normal(size=50)])

        for slope in ("pca", "spearman", "pearson"):
            for scaling in ("absolute", "relative", "equal"):
                fig, ax = plt.subplots()
                out = dg.plot_cross(
                    pts, ax=ax, center="median", slope=slope, scaling=scaling
                )
                self.assertIs(out, ax)
                self.assertEqual(len(ax.lines), 2)
                plt.close(fig)

    def test_scatter_lines(self):
        x = np.linspace(0, 1, 10)
        y = np.sin(x)
        c = np.linspace(0, 1, 10)
        fig, ax = plt.subplots()
        scatter = dg.scatter_lines(x, y, s=20, c=c)
        self.assertIsNotNone(scatter)
        self.assertEqual(len(ax.lines), len(x) - 1)

    def test_plot_linear_confidence_linear_and_log(self):
        x = np.linspace(1.0, 10.0, 25)
        y = 2.0 * x + 3.0

        fig, ax = plt.subplots()
        sc = ax.scatter(x, y)
        out_ax, slope, intercept, r2, mse = dg.plot_linear_confidence(
            x, y, ax=ax, return_model=True, extend_fraction=0.1
        )
        self.assertIs(out_ax, ax)
        self.assertAlmostEqual(slope, 2.0, places=5)
        self.assertAlmostEqual(intercept, 3.0, places=5)
        self.assertAlmostEqual(r2, 1.0, places=5)
        self.assertAlmostEqual(mse, 0.0, places=5)
        # xlim matches exact extended interval and boundary scatter points are unclipped
        self.assertAlmostEqual(ax.get_xlim()[0], 0.9, places=6)
        self.assertAlmostEqual(ax.get_xlim()[1], 11.0, places=6)
        self.assertFalse(sc.get_clip_on())
        # CI/PI ribbons remain clipped inside the axes box
        for coll in ax.collections:
            if coll is not sc:
                self.assertTrue(coll.get_clip_on())

        # Default extend_fraction=0.0 matches [x.min(), x.max()] exactly
        fig_exact, ax_exact = plt.subplots()
        ax_exact.scatter(x, y)
        dg.plot_linear_confidence(x, y, ax=ax_exact)
        self.assertAlmostEqual(ax_exact.get_xlim()[0], 1.0, places=6)
        self.assertAlmostEqual(ax_exact.get_xlim()[1], 10.0, places=6)

        # Log-log power law y = x^2
        fig2, ax2 = plt.subplots()
        _, log_slope, _, log_r2, _ = dg.plot_linear_confidence(
            x, x**2, ax=ax2, logx=True, logy=True, return_model=True
        )
        self.assertAlmostEqual(log_slope, 2.0, places=5)
        self.assertAlmostEqual(log_r2, 1.0, places=5)

        with self.assertRaises(ValueError):
            dg.plot_linear_confidence(np.array([-1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]), ax=ax2, logx=True)

    def test_make_segments_and_colorline(self):
        x = np.linspace(0, 1, 10)
        y = x**2
        segs = dg.make_segments(x, y)
        self.assertEqual(segs.shape, (9, 2, 2))

        fig, _ = plt.subplots()
        lc = dg.colorline(x, y)
        self.assertIsInstance(lc, LineCollection)

        # Multi-line input
        x2 = np.vstack([x, x])
        y2 = np.vstack([y, 2 * y])
        lc2 = dg.colorline(x2, y2)
        self.assertIsInstance(lc2, LineCollection)

    def test_plot3dproj_and_plt_sphere(self):
        t = np.linspace(0, 2 * np.pi, 30)
        ax = dg.plot3dproj(np.cos(t), np.sin(t), t, aspect_ratio=None)
        self.assertEqual(ax.name, "3d")
        self.assertEqual(len(ax.lines), 4)

        ax_s = dg.plt_sphere(np.array([[0.0, 0.0, 0.0]]), radius=1.0)
        self.assertEqual(ax_s.name, "3d")

    def test_image_grid_and_interpolation_utilities(self):
        gx, gy = np.meshgrid(np.arange(4.0), np.arange(3.0))
        x = gx.ravel()
        y = gy.ravel()
        z = x + 2 * y

        img = dg.scatter_image((x, y, z))
        self.assertEqual(img.ndim, 2)

        interp = dg.coords_to_image(x, y, z)
        self.assertEqual(interp.shape, (3, 4))

        for method in ("nan", "nearest", "zero", "linear", "cubic"):
            mat, extent, xi, yi = dg.xyz_to_image(x, y, z, method=method)
            self.assertEqual(mat.shape, (len(yi), len(xi)))
            self.assertEqual(len(extent), 4)
        with self.assertRaises(ValueError):
            dg.xyz_to_image(x, y, z, method="invalid")

        snapshots = [np.full((4, 4), i) for i in range(4)]
        grid = dg.make_snapshot_grid(snapshots, nx=2, ny=2)
        self.assertEqual(grid.shape, (8, 8))

        img_pad = np.ones((5, 6))
        img_pad[:, 2:4] = 0.0
        cropped = dg.crop_by_value(img_pad, target=1.0)
        self.assertEqual(cropped.shape, (5, 2))

    def test_axes_and_math_helpers(self):
        vals = np.array([-9.0, 0.0, 9.0])
        expected = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_allclose(dg.split_log(vals), expected)

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set_title("Test")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        dg.font_size(14, ax=ax)
        self.assertEqual(ax.title.get_fontsize(), 14)

        dg.dim_axes(ax=ax, alpha=0.4)
        self.assertAlmostEqual(ax.spines["bottom"].get_alpha(), 0.4)

        dg.scientific_ticks(ax=ax, axis="y")
        dg.scientific_ticks(ax=ax, axis="x")
        with self.assertRaises(ValueError):
            dg.scientific_ticks(ax=ax, axis="z")

        dg.vanish_axes(gca=ax)
        self.assertFalse(ax.axison)

    def test_file_saving_utilities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.linspace(0, 1, 64).reshape(8, 8)

            exact_path = os.path.join(tmpdir, "exact.png")
            dg.savefig_exact(arr, exact_path, dpi=100)
            self.assertTrue(os.path.isfile(exact_path))

            resampled_path = os.path.join(tmpdir, "exact_resampled.png")
            dg.savefig_exact(arr, resampled_path, target_shape=(16, 16), dpi=100)
            self.assertTrue(os.path.isfile(resampled_path))

            cmap_path = os.path.join(tmpdir, "cmap.png")
            dg.save_colormap(dg.vernalis, cmap_path)
            self.assertTrue(os.path.isfile(cmap_path))

            mat_path = os.path.join(tmpdir, "matrix.png")
            dg.save_matrix_pixels(arr, mat_path)
            self.assertTrue(os.path.isfile(mat_path))

            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            dry_path = os.path.join(tmpdir, "dry.png")
            dg.better_savefig(dry_path, dryrun=True)
            self.assertFalse(os.path.exists(dry_path))

            saved_path = os.path.join(tmpdir, "saved.png")
            dg.better_savefig(saved_path, remove_border=True)
            self.assertTrue(os.path.isfile(saved_path))


if __name__ == "__main__":
    unittest.main()
