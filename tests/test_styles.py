"""Unit tests for degas matplotlib style sheets."""

import os
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import degas as dg


class TestStyles(unittest.TestCase):
    def tearDown(self):
        plt.close("all")
        matplotlib.rcdefaults()

    def test_data_path_contains_expected_styles(self):
        self.assertTrue(os.path.isdir(dg.data_path))
        style_files = sorted(
            f for f in os.listdir(dg.data_path) if f.endswith(".mplstyle")
        )
        for expected in (
            "base.mplstyle",
            "dark.mplstyle",
            "default.mplstyle",
            "degas_figure.mplstyle",
            "pastel_rainbow.mplstyle",
        ):
            self.assertIn(expected, style_files)

    def test_shared_base_and_dark_presentation_style(self):
        for style_name in ("base", "default", "degas_figure", "pastel_rainbow", "dark"):
            with self.subTest(style=style_name):
                matplotlib.rcdefaults()
                dg.set_style(style_name)
                self.assertEqual(plt.rcParams["image.cmap"], "vernalis")
                self.assertEqual(plt.rcParams["font.sans-serif"][0], "Helvetica")
                self.assertTrue(plt.rcParams["savefig.transparent"])
                self.assertEqual(plt.rcParams["patch.edgecolor"], "none")
                self.assertEqual(plt.rcParams["patch.linewidth"], 0.0)

        for paper_style in ("degas_figure", "pastel_rainbow"):
            with self.subTest(paper_style=paper_style):
                matplotlib.rcdefaults()
                dg.set_style(paper_style)
                self.assertEqual(plt.rcParams["axes.edgecolor"].upper(), "#000000")
                self.assertEqual(plt.rcParams["xtick.color"].upper(), "#000000")
                self.assertEqual(plt.rcParams["xtick.labelcolor"].upper(), "#000000")
                self.assertEqual(plt.rcParams["ytick.color"].upper(), "#000000")
                self.assertEqual(plt.rcParams["ytick.labelcolor"].upper(), "#000000")

        matplotlib.rcdefaults()
        dg.set_style("dark")
        self.assertEqual(plt.rcParams["figure.facecolor"], "none")
        self.assertEqual(plt.rcParams["axes.facecolor"], "none")
        self.assertEqual(plt.rcParams["axes.edgecolor"].upper(), "#D9D9D9")
        self.assertEqual(plt.rcParams["xtick.direction"], "out")
        self.assertEqual(plt.rcParams["ytick.direction"], "out")
        self.assertEqual(plt.rcParams["xtick.labelcolor"].upper(), "#8C8C8C")
        self.assertEqual(plt.rcParams["axes.labelcolor"].upper(), "#FFFFFF")
        self.assertFalse(plt.rcParams["legend.frameon"])

    def test_all_styles_load_cleanly(self):
        style_names = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(dg.data_path)
            if f.endswith(".mplstyle")
        )
        self.assertGreaterEqual(len(style_names), 3)
        for name in style_names:
            with self.subTest(style=name):
                matplotlib.rcdefaults()
                dg.set_style(name)
                self.assertEqual(plt.rcParams["image.cmap"], "vernalis")
                fig, ax = plt.subplots()
                ax.plot([0, 1], [0, 1])
                plt.close(fig)

    def test_style_context_does_not_leak(self):
        original_cmap = plt.rcParams["image.cmap"]
        style_file = os.path.join(dg.data_path, "pastel_rainbow.mplstyle")
        with plt.style.context(style_file):
            self.assertEqual(plt.rcParams["image.cmap"], "vernalis")
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            self.assertEqual(len(colors), 12)
        self.assertEqual(plt.rcParams["image.cmap"], original_cmap)

    def test_invalid_style_raises(self):
        with self.assertRaises(OSError):
            dg.set_style("nonexistent_style_name")


if __name__ == "__main__":
    unittest.main()
