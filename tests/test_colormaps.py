"""Unit tests for degas colormaps and color utilities."""

import unittest

import matplotlib
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap

import degas as dg
import degas.colormaps as dg_colormaps


class TestColormaps(unittest.TestCase):
    def test_vernalis_exposed_and_registered(self):
        self.assertIsInstance(dg.vernalis, Colormap)
        self.assertIsInstance(dg.vernalis_r, Colormap)
        self.assertIs(dg.cmap_vn, dg.vernalis)
        self.assertIs(dg.cmap_vn_r, dg.vernalis_r)
        self.assertIn("vernalis", matplotlib.colormaps)
        self.assertIn("vernalis_r", matplotlib.colormaps)

    def test_all_packaged_colormaps_accessible_and_valid(self):
        self.assertGreater(len(dg_colormaps.__all__), 30)
        for name in dg_colormaps.__all__:
            cmap = getattr(dg_colormaps, name)
            self.assertIsInstance(cmap, Colormap)
            self.assertTrue(hasattr(dg, name), f"degas missing colormap {name}")
            self.assertIn(name, matplotlib.colormaps)
            rgba = cmap(0.5)
            self.assertEqual(len(rgba), 4)
            self.assertTrue(all(0.0 <= c <= 1.0 for c in rgba))

    def test_reversed_colormaps_match_forward(self):
        for name in ("vernalis", "batlow", "roma", "vik"):
            fwd = getattr(dg, name)
            rev = getattr(dg, f"{name}_r")
            np.testing.assert_allclose(fwd(0.0), rev(1.0), atol=1e-6)
            np.testing.assert_allclose(fwd(1.0), rev(0.0), atol=1e-6)

    def test_custom_colormaps_and_utilities(self):
        self.assertIsInstance(dg.blbkrd, LinearSegmentedColormap)
        self.assertIsInstance(dg.viridis_arcadia, LinearSegmentedColormap)

        cmap_even = dg.make_cmap(["#000000", "#ffffff"], name="bw_even")
        np.testing.assert_allclose(cmap_even(0.0)[:3], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(cmap_even(1.0)[:3], [1.0, 1.0, 1.0], atol=1e-6)

        cmap_pos = dg.make_cmap(["#000000", "#ff0000", "#ffffff"], positions=[0.0, 0.25, 1.0])
        self.assertIsInstance(cmap_pos, LinearSegmentedColormap)

        lin = dg.make_linear_cmap([(0, 0, 0), (1, 1, 1)], name="lin_bw")
        self.assertIsInstance(lin, LinearSegmentedColormap)

        interp = dg.cmap1D((0, 0, 0), (255, 255, 255), 5)
        self.assertEqual(len(interp), 5)
        np.testing.assert_allclose(interp[0], (0.0, 0.0, 0.0))
        np.testing.assert_allclose(interp[-1], (1.0, 1.0, 1.0))

    def test_lighter_darker_and_color_sort(self):
        base = (0.4, 0.5, 0.6)
        light = dg.lighter(base, 0.5)
        dark = dg.darker(base, 0.5)
        self.assertTrue(all(l > b for l, b in zip(light, base)))
        self.assertTrue(all(d < b for d, b in zip(dark, base)))

        colors = np.array(
            [
                [0.1, 0.2, 0.9],
                [0.9, 0.1, 0.2],
                [0.2, 0.8, 0.1],
            ]
        )
        for method in ("hue", "red", "green", "blue", "luminance", "step", "unknown"):
            sorted_c = dg.color_sort(colors, method=method)
            self.assertEqual(sorted_c.shape, colors.shape)
        rev_red = dg.color_sort(colors, method="red", reverse=True)
        self.assertGreaterEqual(rev_red[0, 0], rev_red[-1, 0])


if __name__ == "__main__":
    unittest.main()
