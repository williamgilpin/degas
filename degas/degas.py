import contextlib
import os, warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.path as mpath
import matplotlib.scale as mscale
import matplotlib.ticker as mticker


try:
    import scipy
    from scipy.interpolate import griddata
    has_scipy = True
except ImportError:
    has_scipy = False

#############################################################
#
#
#   Color schemes and themes
#
#
############################################################

# degas high contrast color scheme
high_contrast = [
    [0.372549, 0.596078, 1], 
    [1.0, .3882, .2784], 
    [0.20784314, 0.67843137, 0.6], 
    [0.59607843, 0.25882353, 0.89019608],
    [0.803922, 0.0627451, 0.462745], 
    [0.917647, 0.682353, 0.105882],
    [0.7, 0.7, 0.7]
]

blue, red, turquoise, purple, magenta, orange, gray  = high_contrast

pastel_rainbow = np.array([
    [221, 59,  53],
    #[211, 132, 71],
    [237, 157, 63],
    [165, 180, 133],
    [63,  148, 109], 
    [50,  122, 137], 
    [44,  115, 178], 
    [43,  52,  124],
    [164, 36, 124],
    [186, 173, 214],
    # [191, 163, 215],
    # [139,  211, 126],
    [163, 218, 133],
    [136, 159, 122],
    [168, 192, 221]
])/255.

pastel_rainbow_alt = pastel_rainbow[[0, 5, 3, 1, 7, 4, 2, 8, 6, 9, 10, 11]]

# degas line plot colors
royal_purple = np.array((120, 81, 169)) / 255.
cornflower_blue = np.array([77, 170, 232]) / 255.
mblue = "#00B4E7"
mred = "#FF0000"

# from matplotlib.colors import LinearSegmentedColormap
# aegean = HexCode("aegean", "#5088C5")
# space = HexCode("space", "#282A49")
# lime = HexCode("lime", "#97CD78")
# butter = HexCode("butter", "#FFFDBD")
# viridis = LinearSegmentedColormap(
#     "viridis",
#     [(0, "space"), (0.468, "aegean"), (0.746, "lime"), (1, "butter")],
# )


# from matplotlib.colors import to_rgb
# def hex_to_rgb01(hex_list):
#     """Return Nx3 float RGB array in [0,1] from hex strings."""
#     return np.array([to_rgb(h) for h in hex_list], dtype=float)


## Arcadia Sciences color scheme
from matplotlib.colors import LinearSegmentedColormap
def make_cmap(hex_list, positions=None, name="custom", N=256):
    """
    Args:
        hex_list (sequence[str]): Hex colors like ["#002", "#c33", ...].
        positions (sequence[float] | None): Monotonic positions in [0,1] where
            each hex should be exact. If None, colors are spaced evenly.
        name (str): Colormap name.
        N (int): Colormap resolution.

    Returns:
        (LinearSegmentedColormap): Matplotlib colormap.
    """
    if positions is None:
        return LinearSegmentedColormap.from_list(name, list(hex_list), N=N)
    return LinearSegmentedColormap.from_list(name, list(zip(positions, hex_list)), N=N)
indigo_arcadia = "#5088C5"
blue_arcadia = "#282A49"
green_arcadia = "#97CD78"
yellow_arcadia = "#FFFDBD"
hexes = [indigo_arcadia, blue_arcadia, green_arcadia, yellow_arcadia]
stops = [0.00, 0.468, 0.746, 1.00]
viridis_arcadia = make_cmap(hexes, stops)


style_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "styles")

# Default relative bar width (bar width / center-to-center pitch), leaving ~36% inter-bar gap
BAR_WIDTH = 0.64
# Default subtle top-corner radius in points for degas.plot_bar / plot_hist rectangles
BAR_CORNER_RADIUS = 1.5


def _is_dark_theme():
    """Return True when the active matplotlib style uses light text (dark theme)."""
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(plt.rcParams.get("text.color", "black"))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luminance > 0.5


def _next_cycle_color(ax):
    """Fetch the next RGB color from ``ax``'s active property cycle."""
    from matplotlib.colors import to_rgb

    try:
        return to_rgb(ax._get_lines.get_next_color())
    except Exception:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#5F98FF"])
        return to_rgb(cycle[0])


class _RoundedBar(patches.Rectangle):
    """Rectangle patch with subtle aspect-preserving rounded top corners."""

    def __init__(self, xy, width, height, corner_radius=BAR_CORNER_RADIUS, **kwargs):
        super().__init__(xy, width, height, **kwargs)
        self._corner_radius = float(corner_radius)
        self._rounded_path = super().get_path()

    def get_path(self):
        return self._rounded_path

    def _update_rounded_path(self, renderer):
        if self._corner_radius <= 0:
            self._rounded_path = super().get_path()
            return
        t = self.get_transform()
        p0 = t.transform((0.0, 0.0))
        p1 = t.transform((1.0, 1.0))
        w_px = abs(p1[0] - p0[0])
        h_px = abs(p1[1] - p0[1])
        if w_px <= 1e-6 or h_px <= 1e-6:
            self._rounded_path = super().get_path()
            return
        r_px = min(
            self._corner_radius * renderer.points_to_pixels(1.0),
            w_px * 0.45,
            h_px * 0.45,
        )
        rx = r_px / w_px
        ry = r_px / h_px
        # Cubic Bezier approximation of a quarter-circle arc
        k = 0.5522847498
        verts = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0 - ry),
            (1.0, 1.0 - ry * (1.0 - k)),
            (1.0 - rx * (1.0 - k), 1.0),
            (1.0 - rx, 1.0),
            (rx, 1.0),
            (rx * (1.0 - k), 1.0),
            (0.0, 1.0 - ry * (1.0 - k)),
            (0.0, 1.0 - ry),
            (0.0, 0.0),
        ]
        codes = [
            mpath.Path.MOVETO,
            mpath.Path.LINETO,
            mpath.Path.LINETO,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.LINETO,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CURVE4,
            mpath.Path.CLOSEPOLY,
        ]
        self._rounded_path = mpath.Path(verts, codes)

    def draw(self, renderer):
        self._update_rounded_path(renderer)
        return super().draw(renderer)


def plot_bar(
    y,
    x=None,
    width=BAR_WIDTH,
    ax=None,
    xmargin=0.04,
    corner_radius=BAR_CORNER_RADIUS,
    **kwargs,
):
    """
    Create a bar chart using the degas house bar proportions (relative bar width
    ``0.64``, ~36% spacing between bars, subtle rounded top corners, borderless fill,
    and balanced outer margins).

    Args:
        y (array-like): Bar heights.
        x (array-like or None): Bar x positions or category labels. Defaults to
            ``np.arange(len(y))``.
        width (float): Bar width relative to unit spacing (default ``0.64``).
        ax (matplotlib.axes.Axes or None): Axes to draw on. Defaults to ``plt.gca()``.
        xmargin (float): Horizontal padding fraction outside the outer bars
            (default ``0.04``).
        corner_radius (float): Subtle rounding radius in points for the outer bar
            corners (default ``1.5``). Set to ``0`` for sharp square corners.
        **kwargs: Additional keyword arguments forwarded to ``ax.bar``.

    Returns:
        ax (matplotlib.axes.Axes): The axes containing the bar chart.
    """
    if ax is None:
        ax = plt.gca()
    y_arr = np.asarray(y)
    if x is None:
        x = np.arange(len(y_arr))
    kwargs.setdefault("edgecolor", "none")
    kwargs.setdefault("linewidth", 0.0)
    container = ax.bar(x, y_arr, width=width, **kwargs)
    if corner_radius and corner_radius > 0:
        for rect in list(container.patches):
            rounded = _RoundedBar(
                rect.get_xy(),
                rect.get_width(),
                rect.get_height(),
                corner_radius=corner_radius,
                facecolor=rect.get_facecolor(),
                edgecolor=rect.get_edgecolor(),
                linewidth=rect.get_linewidth(),
                alpha=rect.get_alpha(),
                zorder=rect.get_zorder(),
                label=rect.get_label(),
            )
            rect.remove()
            ax.add_patch(rounded)
    if xmargin is not None:
        ax.margins(x=xmargin)
    if y_arr.size > 0 and np.all(np.nan_to_num(y_arr, nan=0.0) >= 0):
        ax.set_ylim(bottom=0)
    return ax


def plot_hist(
    x,
    bins="auto",
    rwidth=BAR_WIDTH,
    ax=None,
    xmargin=0.04,
    corner_radius=BAR_CORNER_RADIUS,
    **kwargs,
):
    """
    Create a histogram matching the degas bar chart aesthetic (relative bar width
    ``0.64``, subtle rounded top corners, borderless fill, and flush ``y=0`` baseline).

    Args:
        x (array-like): Input data values.
        bins (int, sequence, or str): Bin specification passed to ``ax.hist``
            (default ``"auto"``).
        rwidth (float): Relative width of bars as a fraction of bin width
            (default ``0.64``).
        ax (matplotlib.axes.Axes or None): Axes to draw on. Defaults to ``plt.gca()``.
        xmargin (float): Horizontal padding fraction outside outer bars (default ``0.04``).
        corner_radius (float): Subtle top-corner rounding radius in points
            (default ``1.5``). Set to ``0`` for sharp square corners.
        **kwargs: Additional keyword arguments forwarded to ``ax.hist``.

    Returns:
        ax (matplotlib.axes.Axes): The axes containing the histogram.
    """
    if ax is None:
        ax = plt.gca()
    kwargs.setdefault("edgecolor", "none")
    kwargs.setdefault("linewidth", 0.0)
    _, _, patches_out = ax.hist(x, bins=bins, rwidth=rwidth, **kwargs)
    if corner_radius and corner_radius > 0:
        flat_patches = (
            [p for sub in patches_out for p in sub]
            if isinstance(patches_out, list) and patches_out and hasattr(patches_out[0], "__iter__")
            else list(patches_out)
        )
        for rect in flat_patches:
            rounded = _RoundedBar(
                rect.get_xy(),
                rect.get_width(),
                rect.get_height(),
                corner_radius=corner_radius,
                facecolor=rect.get_facecolor(),
                edgecolor=rect.get_edgecolor(),
                linewidth=rect.get_linewidth(),
                alpha=rect.get_alpha(),
                zorder=rect.get_zorder(),
                label=rect.get_label(),
            )
            rect.remove()
            ax.add_patch(rounded)
    if xmargin is not None:
        ax.margins(x=xmargin)
    ax.set_ylim(bottom=0)
    return ax


def colorbar(
    mappable,
    ax=None,
    location="right",
    size="5%",
    pad=None,
    label=None,
    labelpad=None,
    max_ticks=5,
    inner_ticks=False,
    **kwargs,
):
    """
    Attach a minimal, frameless colorbar whose length matches the adjacent axes box,
    with outward ticks and label baseline (foot of the text) facing inward.

    When ``fig`` uses ``constrained_layout=True`` (the default in ``degas.subplots``),
    uses matplotlib's layout-aware colorbar so multi-panel grids never collide and
    ``inner_ticks=True`` dynamically expands spacing for wide/many-sig-fig tick labels.
    Otherwise falls back to ``make_axes_locatable`` with dynamic tick-extent measurement.

    Args:
        mappable: Matplotlib ScalarMappable (e.g. return value of ``imshow``,
            ``scatter``, ``pcolormesh``, or ``colorline``).
        ax (matplotlib.axes.Axes or None): Parent axes to align with. Defaults to
            ``mappable.axes`` or ``plt.gca()``.
        location (str): Side to place the colorbar on (``"right"``, ``"left"``,
            ``"top"``, or ``"bottom"``).
        size (str or float): Colorbar thickness (e.g. ``"5%"`` or fraction ``0.05``).
        pad (float or None): Spacing between parent axes and colorbar. Automatically
            chosen (and dynamically sized for ``inner_ticks=True``) when ``None``.
        label (str or None): Optional colorbar label.
        labelpad (float or None): Distance in points between tick labels (or the
            colorbar strip when ``inner_ticks=True``) and the colorbar label.
        max_ticks (int or None): Maximum number of colorbar ticks (default ``5``).
        inner_ticks (bool): If True, place ticks and tick labels on the side of the
            colorbar closer to the figure, put the text label directly against the
            outer side of the colorbar strip, and dynamically size the gap between
            the figure and colorbar so tick labels never overlap the figure.
        **kwargs: Additional keyword arguments forwarded to ``fig.colorbar``.

    Returns:
        cbar (matplotlib.colorbar.Colorbar): The created colorbar.
    """
    from mpl_toolkits.axes_grid1 import Size, make_axes_locatable

    if ax is None:
        ax = getattr(mappable, "axes", None) or plt.gca()
    fig = ax.figure
    orientation = "horizontal" if location in ("top", "bottom") else "vertical"
    use_constrained = bool(fig.get_constrained_layout())
    divider = None

    if use_constrained:
        if isinstance(size, str) and size.endswith("%"):
            fraction = float(size[:-1]) / 100.0
        else:
            fraction = float(size)
        cb_pad = 0.04 if pad is None else pad
        kwargs.setdefault("fraction", fraction)
        kwargs.setdefault("pad", cb_pad)
        cbar = fig.colorbar(mappable, ax=ax, location=location, **kwargs)
    else:
        divider = make_axes_locatable(ax)
        initial_pad = 0.08 if pad is None else pad
        cax = divider.append_axes(location, size=size, pad=initial_pad)
        kwargs.setdefault("orientation", orientation)
        cbar = fig.colorbar(mappable, cax=cax, **kwargs)

    cbar.outline.set_visible(False)
    cbar.ax.tick_params(direction="out")
    if max_ticks is not None:
        cb_axis = "x" if orientation == "horizontal" else "y"
        limit_ticks(cbar.ax, max_ticks=max_ticks, axis=cb_axis)

    if inner_ticks:
        if location == "right":
            cbar.ax.yaxis.set_ticks_position("left")
            cbar.ax.yaxis.set_label_position("right")
        elif location == "left":
            cbar.ax.yaxis.set_ticks_position("right")
            cbar.ax.yaxis.set_label_position("left")
        elif location == "top":
            cbar.ax.xaxis.set_ticks_position("bottom")
            cbar.ax.xaxis.set_label_position("top")
        elif location == "bottom":
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("bottom")

    default_labelpad = 4 if inner_ticks else 6
    lpad = default_labelpad if labelpad is None else labelpad
    if label is not None:
        if location == "right":
            cbar.set_label(label, rotation=270, va="bottom", labelpad=lpad)
        elif location == "left":
            cbar.set_label(label, rotation=90, va="bottom", labelpad=lpad)
        else:
            cbar.set_label(label, labelpad=lpad)

    if inner_ticks and pad is None and not use_constrained and divider is not None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        if orientation == "vertical":
            labels = [t for t in cbar.ax.get_yticklabels() if t.get_text()]
            max_px = max((t.get_window_extent(renderer).width for t in labels), default=0.0)
            extra_pt = plt.rcParams["ytick.major.size"] + plt.rcParams["ytick.major.pad"] + 8.0
            dynamic_pad = max(0.08, max_px / fig.dpi + extra_pt / 72.0)
            horiz = list(divider.get_horizontal())
            if len(horiz) >= 2:
                horiz[1] = Size.Fixed(dynamic_pad)
                divider.set_horizontal(horiz)
        else:
            labels = [t for t in cbar.ax.get_xticklabels() if t.get_text()]
            max_px = max((t.get_window_extent(renderer).height for t in labels), default=0.0)
            extra_pt = plt.rcParams["xtick.major.size"] + plt.rcParams["xtick.major.pad"] + 8.0
            dynamic_pad = max(0.08, max_px / fig.dpi + extra_pt / 72.0)
            vert = list(divider.get_vertical())
            if len(vert) >= 2:
                vert[1] = Size.Fixed(dynamic_pad)
                divider.set_vertical(vert)

    return cbar


def set_box_aspect(aspect=1.0, ax=None):
    """
    Lock the physical height/width aspect ratio of the plot box itself, independent
    of axis limits, log scaling, tick label widths, or attached colorbars.

    Args:
        aspect (float): Desired physical ``height / width`` ratio of the plot box
            (e.g. ``1.0`` for a square box, ``0.75`` for a 4:3 box).
        ax (matplotlib.axes.Axes or None): Target axes. Defaults to ``plt.gca()``.

    Returns:
        ax (matplotlib.axes.Axes): The updated axes.
    """
    if ax is None:
        ax = plt.gca()
    if ax.name == "3d":
        ax.set_box_aspect((1.0, 1.0, float(aspect)))
    else:
        ax.set_box_aspect(float(aspect))
    return ax


def subplots(
    nrows=1,
    ncols=1,
    figsize=None,
    box_aspect=None,
    box_size=None,
    constrained_layout=True,
    **kwargs,
):
    """
    Create a figure and subplot grid while protecting the physical aspect ratio
    of each plot box (so labels, titles, and colorbars do not distort the box shape).

    Args:
        nrows (int): Number of subplot rows (default ``1``).
        ncols (int): Number of subplot columns (default ``1``).
        figsize (tuple[float, float] or None): Overall figure size ``(width, height)``
            in inches. When ``box_aspect`` and ``box_size`` are omitted, the intended
            per-panel box aspect is inferred as ``(H / nrows) / (W / ncols)``.
        box_aspect (float or None): Explicit physical ``height / width`` ratio for
            every plot box (e.g. ``1.0`` for square boxes).
        box_size (tuple[float, float] or None): Desired physical ``(width, height)``
            in inches of each individual plot box. Automatically sets ``box_aspect``
            and computes an enclosing ``figsize`` with room for labels and ticks.
        constrained_layout (bool): Enable matplotlib constrained layout (default True).
        **kwargs: Additional keyword arguments forwarded to ``plt.subplots``.

    Returns:
        tuple: ``(fig, axes)`` matching ``plt.subplots``.
    """
    if box_size is not None:
        bw, bh = box_size
        if box_aspect is None:
            box_aspect = float(bh) / float(bw)
        if figsize is None:
            # Add compact breathing room around each plot box for labels/ticks/colorbars
            pad_w = 0.85 * ncols + 0.25
            pad_h = 0.75 * nrows + 0.25
            figsize = (bw * ncols + pad_w, bh * nrows + pad_h)

    if figsize is None:
        base_w, base_h = plt.rcParams["figure.figsize"]
        figsize = (base_w * ncols, base_h * nrows)

    if box_aspect is None:
        box_aspect = (figsize[1] / nrows) / (figsize[0] / ncols)

    if "layout" not in kwargs:
        kwargs["constrained_layout"] = constrained_layout

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    for ax in np.atleast_1d(axes).ravel():
        set_box_aspect(box_aspect, ax=ax)
    return fig, axes


# Default maximum major tick count (yields 3-5 clean ticks and always preserves 0)
DEFAULT_MAX_TICKS = 5
_DEFAULT_TICK_STEPS = (1, 2, 2.5, 5, 10)
_ORIG_LINEAR_LOCATORS = mscale.LinearScale.set_default_locators_and_formatters
_ORIG_LOG_LOCATORS = mscale.LogScale.set_default_locators_and_formatters
_ACTIVE_TICK_CONFIG = None


def _make_clean_locator(scale_name, max_ticks, min_ticks=3, steps=_DEFAULT_TICK_STEPS, base=10.0):
    if scale_name == "log":
        return mticker.LogLocator(base=base, numticks=max(2, int(max_ticks)))
    nbins = max(1, int(max_ticks) - 1)
    eff_min = min(int(min_ticks), int(max_ticks))
    return mticker.MaxNLocator(nbins=nbins, steps=list(steps), min_n_ticks=eff_min)


def set_max_ticks(max_ticks=DEFAULT_MAX_TICKS, min_ticks=3, steps=_DEFAULT_TICK_STEPS):
    """
    Set a global cap on default major tick counts for newly created matplotlib axes
    and colorbars (3-5 clean ticks by default, always preserving ``0`` when in range).

    Explicit ``ax.set_xticks`` / ``ax.set_yticks`` calls and categorical axes are
    unaffected. Pass ``max_ticks=None`` to restore matplotlib's default behavior.

    Args:
        max_ticks (int or None): Maximum number of major ticks per axis (default ``5``,
            giving 3-5 ticks via ``MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])``).
            Set to ``None`` to restore matplotlib's default ``AutoLocator``.
        min_ticks (int): Minimum number of major ticks to target (default ``3``).
        steps (sequence[float]): Preferred nice step multiples (default
            ``(1, 2, 2.5, 5, 10)``).
    """
    global _ACTIVE_TICK_CONFIG
    if max_ticks is None:
        mscale.LinearScale.set_default_locators_and_formatters = _ORIG_LINEAR_LOCATORS
        mscale.LogScale.set_default_locators_and_formatters = _ORIG_LOG_LOCATORS
        _ACTIVE_TICK_CONFIG = None
        return

    config = (int(max_ticks), int(min_ticks), tuple(steps))
    _ACTIVE_TICK_CONFIG = config

    def _degas_linear_locators(self, axis):
        _ORIG_LINEAR_LOCATORS(self, axis)
        mt, mn, st = _ACTIVE_TICK_CONFIG
        axis.set_major_locator(_make_clean_locator("linear", mt, min_ticks=mn, steps=st))

    def _degas_log_locators(self, axis):
        _ORIG_LOG_LOCATORS(self, axis)
        mt, mn, st = _ACTIVE_TICK_CONFIG
        axis.set_major_locator(
            _make_clean_locator("log", mt, min_ticks=mn, steps=st, base=self.base)
        )

    mscale.LinearScale.set_default_locators_and_formatters = _degas_linear_locators
    mscale.LogScale.set_default_locators_and_formatters = _degas_log_locators


def limit_ticks(
    ax=None,
    max_ticks=DEFAULT_MAX_TICKS,
    axis="both",
    min_ticks=3,
    steps=_DEFAULT_TICK_STEPS,
):
    """
    Cap the number of major ticks on an existing axes (typically 3-5 clean ticks,
    preserving ``0`` whenever it lies within the axis limits).

    Args:
        ax (matplotlib.axes.Axes or None): Target axes. Defaults to ``plt.gca()``.
        max_ticks (int): Maximum number of major ticks (default ``5``).
        axis (str): ``"x"``, ``"y"``, or ``"both"`` (default ``"both"``).
        min_ticks (int): Minimum number of major ticks to target (default ``3``).
        steps (sequence[float]): Preferred nice step multiples.

    Returns:
        ax (matplotlib.axes.Axes): The updated axes.
    """
    if ax is None:
        ax = plt.gca()
    if axis not in ("x", "y", "both"):
        raise ValueError("axis must be 'x', 'y', or 'both'")
    axes_to_update = []
    if axis in ("x", "both"):
        axes_to_update.append((ax.xaxis, ax.get_xscale()))
    if axis in ("y", "both"):
        axes_to_update.append((ax.yaxis, ax.get_yscale()))
    for ax_obj, scale_name in axes_to_update:
        if isinstance(ax_obj.get_major_locator(), mticker.FixedLocator):
            continue
        base = getattr(ax_obj.get_scale_obj(), "base", 10.0) if hasattr(ax_obj, "get_scale_obj") else 10.0
        ax_obj.set_major_locator(
            _make_clean_locator(scale_name, max_ticks, min_ticks=min_ticks, steps=steps, base=base)
        )
    return ax


def _style_files(style_name="default"):
    base_file = os.path.join(style_path, "base.mplstyle")
    target_file = os.path.join(style_path, style_name + ".mplstyle")
    if style_name == "base" or not os.path.exists(base_file):
        return [target_file]
    return [base_file, target_file]


def set_style(style_name="default", max_ticks=DEFAULT_MAX_TICKS):
    """
    Set the style of the plot to one of the styles in the styles folder. This wraps
    matplotlib's style.use function, layering ``base.mplstyle`` underneath the
    selected style and capping default tick density to 3-5 clean ticks (preserving ``0``).

    Args:
        style_name (str): Name of a style in ``degas/styles`` (e.g. ``"degas_figure"``,
            ``"dark"``, ``"pastel_rainbow"``, ``"default"``, ``"base"``).
        max_ticks (int or None): Maximum default major ticks per axis (default ``5``).
            Pass ``None`` to keep matplotlib's default tick density.
    """
    plt.style.use(_style_files(style_name))
    set_max_ticks(max_ticks)


@contextlib.contextmanager
def style_context(style_name="default", max_ticks=DEFAULT_MAX_TICKS):
    """
    Context manager applying a ``degas`` house style and tick-density cap for a
    single figure without leaking global state.

    Args:
        style_name (str): Name of a style in ``degas/styles``.
        max_ticks (int or None): Maximum default major ticks per axis (default ``5``).
    """
    prev_config = _ACTIVE_TICK_CONFIG
    with plt.style.context(_style_files(style_name)):
        set_max_ticks(max_ticks)
        try:
            yield
        finally:
            if prev_config is None:
                set_max_ticks(None)
            else:
                set_max_ticks(*prev_config)


def plot_splity(x, y1, y2, ax=None, kwargs1=None, kwargs2=None):
    """
    Plot two y-axes on the same plot with a split y axis. This is useful for
    plotting two quantities with very different scales on the same plot. Separate ticks
    and labels are used for each y axis

    Args:
        x (array-like): x values
        y1 (array-like): y values for the first axis
        y2 (array-like): y values for the second axis
        ax (matplotlib.axes.Axes): axes to plot on. If None, use plt.gca()
        kwargs1 (dict): keyword arguments to pass to the plot

    Returns:
        ax (matplotlib.axes.Axes): axes with the plot
        ax2 (matplotlib.axes.Axes): axes with the second y axis
    """
    if not ax:
        ax = plt.gca()
    kw1 = dict(kwargs1) if kwargs1 else {}
    kw2 = dict(kwargs2) if kwargs2 else {}
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#5F98FF", "#FF6347"])
    if "color" not in kw1 and "c" not in kw1 and len(cycle) >= 1:
        kw1["color"] = cycle[0]
    if "color" not in kw2 and "c" not in kw2 and len(cycle) >= 2:
        kw2["color"] = cycle[1]
    ax.plot(x, y1, **kw1)
    ax2 = ax.twinx()
    ax2.plot(x, y2, **kw2)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax2.patch.set_visible(True)
    if _ACTIVE_TICK_CONFIG is not None:
        mt, mn, st = _ACTIVE_TICK_CONFIG
        limit_ticks(ax2, max_ticks=mt, axis="y", min_ticks=mn, steps=st)
    return ax, ax2

def fixed_aspect_ratio(ratio, ax=None, log=False, semilogy=False, semilogx=False):
    """
    Set a fixed aspect ratio on matplotlib plots, regardless of axis units

    Args:
        ratio (float): The desired aspect ratio
        ax (matplotlib.axes.Axes): The axes to set the aspect ratio on. If None, use the
            current axes.
        log (bool): Whether to use a log scale
        semilogy (bool): Whether to use a semilog scale on the y axis
        semilogx (bool): Whether to use a semilog scale on the x axis

    Returns:
        ax (matplotlib.axes.Axes): The axes with the aspect ratio set
    """
    if not ax:
        ax = plt.gca()
    xvals, yvals = ax.axes.get_xlim(), ax.axes.get_ylim()
    xrange = np.abs(xvals[1] - xvals[0])
    yrange = np.abs(yvals[1] - yvals[0])
    if log:
        xrange = np.log10(xvals[1]) - np.log10(xvals[0])
        yrange = np.log10(yvals[1]) - np.log10(yvals[0])
    if semilogy:
        yrange = np.log10(yvals[1]) - np.log10(yvals[0])
    if semilogx:
        xrange = np.log10(xvals[1]) - np.log10(xvals[0])
    try:
        ax.set_aspect(ratio * (xrange / yrange), adjustable='box')
    except (NotImplementedError, ValueError):
        warnings.warn("Setting aspect ratio is experimental for 3D plots.")
        plt.gca().set_box_aspect((1, 1, ratio*(xrange/yrange)))
        #ax.set_box_aspect((ratio*(xrange/yrange), 1, 1))

#############################################################
#
#
#   Plotting functions
#
#
############################################################

# def fit_ellipse(cont, method=0):
#     """
#     Fit an ellipse to a set of points.

#     Args:
#         cont (ndarray): The points to fit an ellipse to, containing n ndarray elements
#                         representing each point, each with d elements representing the
#                         coordinates for the point.
#         method (int):   The method to use to fit the ellipse. 1 uses the algebraic method
#                         and 2 uses the geometric method.

#     Returns:
#         a (float): The x-coordinate of the center of the ellipse.
#         b (float): The y-coordinate of the center of the ellipse.
#         c (float): The x-radius of the ellipse.
#         d (float): The y-radius of the ellipse.
#         e (float): The angle of the ellipse in radians.

#     References:
#         [1] Fitzgibbon, A.W., Pilu, M., and Fischer R.B., Direct least squares fitting 
#             of ellipses, 1996:

#     """
#     x, y = cont[:, 0][:, None], cont[:, 1][:, None]

#     D = np.hstack([x * x, x * y, y * y, x, y, np.ones(x.shape)])
#     S = np.dot(D.T, D)
#     C = np.zeros([6, 6])
#     C[0, 2] = C[2, 0] = 2
#     C[1, 1] = -1
#     E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))

#     if method == 1:
#         n = np.argmax(np.abs(E))
#     else:
#         n = np.argmax(E)
#     a = V[:, n]

#     # Fit ellipse
#     b, c, d, f, g, a = a[1] / 2., a[2], a[3] / 2., a[4] / 2., a[5], a[0]
#     num = b * b - a * c
#     cx = (c * d - b * f) / num
#     cy = (a * f - b * d) / num

#     angle = 0.5 * np.arctan(2 * b / (a - c)) * 180 / np.pi
#     up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
#     down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
#     down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
#     a = np.sqrt(abs(up / down1))
#     b = np.sqrt(abs(up / down2))

#     return a*2, b*2, angle

# def imadjust(im0, threshold=.01):
#     """
#     Adjust the contrast of an image by setting the lowest and highest values to 0 and 1,
#     respectively, and linearly scaling the values in between.

#     Args:
#         im0 (ndarray): the image to adjust
#         threshold (float): the fraction of pixels to set to 0 and 1, or, if a tuple,
#                 the fraction of pixels to set to 0 and 1 in the lower and upper tails,
#                 respectively.

#     Returns:
#         im (ndarray): the adjusted image

#     """
#     if isinstance(threshold, (list, tuple, np.ndarray)):
#         low_threshold, high_threshold = threshold
#     else:
#         low_threshold, high_threshold = threshold,threshold
    
#     im = np.copy(im0)
#     if max(im.ravel) > 1:
#         im = im.astype(float) / 255.
    
#     if len(im.shape)==2:
#         im = im[..., :]
    
#     for chan_ind in range(im.shape[2]):
#         if chan_ind < 3:
#             chan = np.copy(im[..., chan_ind])

#             lo_edge = np.percentile(np.ravel(chan), 100.*low_threshold)
#             hi_edge = np.percentile(np.ravel(chan), 100.*(1.-high_threshold))
#             new_chan = (chan - lo_edge)/(hi_edge - lo_edge)
#             new_chan[new_chan < 0.0]=0.0
#             new_chan[new_chan > 1.0]=1.0
#             im[...,chan_ind] = new_chan
        
#     return(im)

def scatter_image(data, default=np.nan):
    """
    Create an image from scattered data by mapping x, y, z values to pixel indices.

    Args:
        data (array-like): 3xN array of x, y, z values.
        default (float): Default value for empty pixels.

    Returns:
        img (ndarray): Image with mapped data.
    """
    x, y, z = data
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Determine grid resolutions for x and y
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    dx = np.min(np.diff(np.sort(unique_x))) if unique_x.size > 1 else 1
    dy = np.min(np.diff(np.sort(unique_y))) if unique_y.size > 1 else 1
    N_x = int(np.ceil((x_max - x_min) / dx))# + 1
    N_y = int(np.ceil((y_max - y_min) / dy))# + 1

    img = np.full((N_y, N_x), default, dtype=z.dtype)
    ix = np.round((x - x_min) / dx).astype(int)
    iy = np.round((y - y_min) / dy).astype(int)
    valid = (ix >= 0) & (ix < N_x) & (iy >= 0) & (iy < N_y)
    img[iy[valid], ix[valid]] = z[valid]

    return img

def make_snapshot_grid(data_list, nx=10, ny=10):
    """
    Make a grid of snapshots from a list of data points, and return a concatenated
    image in the form of a numpy array.

    Args:
        data_list (list): a list of data points, each of which is a 2D array
        nx, ny (int): the number of snapshots along each axis

    Returns:
        grid (ndarray): a concatenated image of the snapshots
    """
    n = len(data_list)
    n_gap = nx * ny
    im_list = list(data_list[::(n // n_gap)])[:n_gap]
    # split into nx sublists
    im_list = [im_list[i:i + nx] for i in range(0, len(im_list), nx)]
    # concatenate each sublist horizontally
    im_list = [np.hstack(im) for im in im_list]
    # concatenate each horizontal sublist vertically
    grid = np.vstack(im_list)
    return grid

def compute_pca(data):
    """
    Compute PCA without using sklearn
    
    Args:
        data (ndarray): the data to compute PCA on, with shape (n_samples, n_features)

    Returns:
        eigvals (ndarray): the eigenvalues of the covariance matrix
        eigvecs (ndarray): the eigenvectors of the covariance matrix
        explained_variance (ndarray): the explained variance of each principal component

    """
    data = data - np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(cov)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    # sort by eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    explained_variance = eigvals / np.sum(eigvals)
    return eigvals, eigvecs, explained_variance

def fit_ellipse(data):
    """
    Fit an ellipse to a set of points.

    Args:
        data (ndarray): The points to fit an ellipse to, containing n ndarray elements
                        representing each point, each with d elements representing the
                        coordinates for the point.

    Returns:
        width (float): The width of the ellipse.
        height (float): The height of the ellipse.
        angle (float): The angle of the ellipse in radians.

    """
    eigs, vecs, _ = compute_pca(data)
    mean = np.mean(data, axis=0)
    angle = np.arctan2(vecs[1, 0], vecs[0, 0])
    width, height = 2 * np.sqrt(eigs)
    return width, height, angle

def draw_ellipse(points, ax=None, **kwargs):
    """
    Draw an ellipse around a cluster of points in 2D.
    
    Args:
        points (ndarray): The points in a cluster to enclose with an ellipse, containing n
                          ndarray elements representing each point, each with d elements
                          representing the coordinates for the point.
        ax (matplotlib.axes.Axes): The axes to draw the ellipse on.
        **kwargs: Additional keyword arguments to pass to matplotlib.patches.Ellipse.

    Returns:
        ellipse (matplotlib.patches.Ellipse): The ellipse drawn on the axes.

    Example:

        plt.figure()
        all_pairs = np.random.random((100, 2))
        all_pairs = all_pairs @ np.linalg.inv(np.random.random((2, 2)))
        plt.scatter(all_pairs[:, 0], all_pairs[:, 1])
        draw_ellipse(all_pairs, ax=plt.gca(), edgecolor="red")
    """
    if ax is None:
        ax = plt.gca()
    width, height, angle = fit_ellipse(points)
    center = tuple(np.mean(points, axis=0))
    kwargs.setdefault("fill", False)
    if "edgecolor" not in kwargs and "ec" not in kwargs and "color" not in kwargs:
        kwargs["edgecolor"] = _next_cycle_color(ax)
    kwargs.setdefault("linewidth", plt.rcParams["lines.linewidth"])
    ellipse = patches.Ellipse(center, width, height, angle=float(np.degrees(angle)), **kwargs)
    ax.add_patch(ellipse)
    return ellipse

from scipy.stats import spearmanr, pearsonr
def plot_cross(all_pairs, 
               scale=0.1, 
               ax=None, 
               center="mean", 
               aspect=1,
               flip=False, 
               slope="pca", 
               scaling="absolute", 
               **kwargs):
    """
    Plot crosses along the major and minor axes of a data cloud

    Args:
        all_pairs (np.ndarray): array of shape (n, 2) with x and y coordinates
        ax (matplotlib.axes.Axes): axes to plot on. If None, plt.gca() is used
        center (str): "mean" or "median" to determine the center of the cross
        scale (float): scale of the cross
        slope (str): "pca" or "spearman" or "pearson" to determine the slope of the 
            cross. If "pca", the slope is determined by the principal components, if 
            "spearman", the slope is determined by the spearman correlation, and if
            "pearson", the slope is determined by the pearson correlation.
        scaling (str): "absolute" or "relative" or "equal" to determine the scaling of 
            the cross. If "absolute", the cross is scaled to the absolute values of the
            principal components, if "relative", the cross is scaled to a maximum width,
            but the aspect ratio from the principal components is preserved, and if
             "equal", both arms of the cross are scaled to the same length.
        **kwargs: keyword arguments passed to plt.plot

    Returns:
        matplotlib.axes.Axes: axes with the cross
    """
    if center == "median":
        x, y = np.median(all_pairs, axis=0)
    elif center == "mean":
        x, y = np.mean(all_pairs, axis=0)
    else:
        warnings.warn("Unknown centering method. Using mean.")
        x, y = np.mean(all_pairs, axis=0)

    scale1, scale2, ang = fit_ellipse(all_pairs)

    if ax is None:
        ax = plt.gca()

    if scaling == "relative":
        scale_max = np.max([scale1, scale2])
        scale1 = scale1 / scale_max
        scale2 = scale2 / scale_max
    if scaling == "absolute":
        scale1 = scale1
        scale2 = scale2
    if scaling == "equal":
        scale1 = 1
        scale2 = 1

    if slope == "spearman":
        ang = np.arctan(spearmanr(all_pairs[:, 0], all_pairs[:, 1], nan_policy="omit")[0])

    if slope == "pearson":
        ang = np.arctan(np.corrcoef(all_pairs[:, 0], all_pairs[:, 1])[0, 1])

    if flip:
        ang = ang + np.pi / 2
        # scale2 = scale1

    ## plot a line at the angle of the ellipse
    vec1 = np.array([
        [x - scale * scale1 * np.cos(ang) * aspect, x + scale * scale1 * np.cos(ang) * aspect],
        [y - scale * scale1 * np.sin(ang), y + scale * scale1 * np.sin(ang)],
    ])
    ## define a vector at a 90 degree angle
    ang = ang + np.pi / 2
    vec2 = np.array([
        [x - scale * scale2 * np.cos(ang) * aspect, x + scale * scale2 * np.cos(ang) * aspect],
        [y - scale * scale2 * np.sin(ang), y + scale * scale2 * np.sin(ang)],
    ])  

    ax.plot(vec1[0], vec1[1], **kwargs)
    ax.plot(vec2[0], vec2[1], **kwargs)
    return ax


def scatter_lines(x, y, s=None, c=None, linewidth2=None, ax=None, **kwargs):
    """
    Scatter plot with lines connecting the points. The line segments will match
    the color of the point they start at.

    Args:
        x, y (array_like, shape (n, )): Input data
        s (scalar or array_like, shape (n, ), optional): Size of the points.
        c (color, sequence, or sequence of color, optional): The marker colors.
        linewidth2 (float or None): Line width of connecting segments. Defaults to
            ``0.5 * rcParams["lines.linewidth"]``.
        ax (matplotlib.axes.Axes or None): Axes to draw on. Defaults to ``plt.gca()``.
        **kwargs: Additional keyword arguments are passed through to ``ax.scatter``.
    """
    if ax is None:
        ax = plt.gca()
    if linewidth2 is None:
        linewidth2 = 0.5 * plt.rcParams["lines.linewidth"]
    kwargs.setdefault("edgecolors", "none")

    # Create scatter plot
    if c is None:
        base_color = _next_cycle_color(ax)
        scatter = ax.scatter(x, y, s=s, color=base_color, **kwargs)
        colors = [base_color] * len(x)
    else:
        scatter = ax.scatter(x, y, s=s, c=c, **kwargs)
        colors = scatter.to_rgba(c)

    ## Loop to draw lines connecting all points
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], color=colors[i], linewidth=linewidth2)

    return scatter


def plot_err(
    y, 
    errs, 
    x=[], 
    color=None, 
    loglog=False,
    lightness=0.33/.5,
    ax=None,
    fill_kwargs=None,
    **plot_kwargs):
    """
    Plot a curve with shaded error bands.

    Args:
        y (array): A list of values to plot
        errs (array): A list of errors, or a pair of lists of upper and lower errors 
        x (array): A list of x positions
        color (color or None): Line and error-band color (RGB tuple, hex string, or
            named color). If None, uses the next color from the active property cycle.
        loglog (bool): Whether to plot the x and y axes in log scale
        lightness (float): The visual lightness of the error bars on light themes
        fill_kwargs (dict or None): keyword arguments passed to ``ax.fill_between``
        plot_kwargs: passed to the line plot

    Returns:
        ax (matplotlib.axes.Axes): The axes on which the plot was drawn

    Example:
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        errs = np.random.random(100) * .1
        plot_err(y, errs, x=x, color=(0,0,1))

    """
    from matplotlib.colors import to_rgb

    if ax is None:
        ax = plt.gca()
    if len(x) < 1:
        x = np.arange(len(y))
    if color is None:
        color_rgb = _next_cycle_color(ax)
    else:
        color_rgb = to_rgb(color)

    errs = np.asarray(errs)
    y = np.asarray(y)
    if len(errs.shape) > 1:
        if errs.shape[1] == 2:
            err_lo, err_hi = errs[:, 0], errs[:, 1]
    else:
        err_lo = errs
        err_hi = errs
        
    trace_lo, trace_hi = y - err_lo, y + err_hi

    ## Set default colors and other plot properties (adapt band fill to dark vs light theme)
    if _is_dark_theme():
        all_fill_kwargs = dict(
            color=color_rgb,
            alpha=0.28,
            edgecolor="none",
        )
    else:
        all_fill_kwargs = dict(
            color=lighter(color_rgb, lightness),
            edgecolor="none",
        )
    
    all_plot_kwargs = dict(
        color=color_rgb,
    )
    
    all_plot_kwargs.update(plot_kwargs) # update to user-specified plot_kwargs
    lineplot = ax.plot(x, y, **all_plot_kwargs)
    z = lineplot[0].get_zorder()

    all_fill_kwargs["zorder"] = z - 1
    if fill_kwargs:
        all_fill_kwargs.update(fill_kwargs) # update to user-specified fill_kwargs
    ax.fill_between(x, trace_lo, trace_hi, **all_fill_kwargs)
    if loglog:
        ax.set_yscale('log', nonpositive='clip')
        ax.set_xscale('log', nonpositive='clip')
    
    return ax

def plot3dproj(x, y, z, *args, 
    ax=None,
    color=None, 
    shadow_dist=1.0, 
    color_proj=None, 
    elev_azim=(39,-47), 
    show_labels=False, 
    aspect_ratio=1.0,
    **kwargs):
    """
    Create a three dimensional plot, with projections onto the 2D coordinate planes
    
    Parameters
    ----------
    x, y, z : 1D arrays of coordinates to plot
    *args : arguments passed to the matplotlib plt.plot functions
    color : color or None
        Color for the three-dimensional line plot. Defaults to black on light themes
        or ``degas.cornflower_blue`` on dark themes.
    color_proj : color or None
        Color for the two-dimensional projection plots. Defaults to a softer version
        of ``color`` adapted to the active theme.
    shadow_dist : float
        The relative distance of axes to their shadow. If a single value, 
        then the same distance is used for all three axies. If a triple, then 
        different values are used for all axes
    elev_azim : length-2 tuple
        The starting values of elevation and azimuth when viewing the figure
    show_labels : bool
        Whether to show numerical labels on the axes
    aspect_ratio : None or int
        The integer aspect ratio to impose on the axes. If not passed, the default
        aspect ratio is used
    """
    from matplotlib.colors import to_rgb

    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
    if color is None:
        color_rgb = tuple(cornflower_blue) if _is_dark_theme() else (0.0, 0.0, 0.0)
    else:
        color_rgb = to_rgb(color)
    if color_proj is None:
        color_proj = darker(color_rgb, 0.5) if _is_dark_theme() else lighter(color_rgb, 0.6)

    if np.isscalar(shadow_dist) == 1:
        sdist_x = shadow_dist
        sdist_y = shadow_dist
        sdist_z = shadow_dist
    else:
        sdist_x, sdist_y, sdist_z = shadow_dist

    ax.plot(x, z, *args, zdir='y', zs=sdist_y*np.max(y), color=color_proj, **kwargs)
    ax.plot(y, z, *args, zdir='x', zs=sdist_x*np.min(x), color=color_proj, **kwargs)
    ax.plot(x, y, *args, zdir='z', zs=sdist_z*np.min(z), color=color_proj, **kwargs)

    ax.plot(x, y, z, *args, color=color_rgb, **kwargs)

    ax.view_init(elev=elev_azim[0], azim=elev_azim[1])
    ax.set_aspect('auto', adjustable='box') 
    
#     ratio = 1.0
#     xvals, yvals = ax.get_xlim(), ax.get_ylim()
#     xrange = xvals[1]-xvals[0]
#     yrange = yvals[1]-yvals[0]
#     ax.set_aspect(ratio*(xrange/yrange), adjustable='box')
    if aspect_ratio:
        fixed_aspect_ratio(aspect_ratio)

    if not show_labels:
        ax.set_xticklabels([])                               
        ax.set_yticklabels([])                               
        ax.set_zticklabels([])
    #plt.show()

    return ax

def plot_linear_confidence(
    x, y, ax=None, show_ci=True, show_pi=True,
    ci_range=0.95,
    return_model=False,
    extend_fraction=0.0,
    ci_kwargs={},
    pi_kwargs={},
    logx=None,
    logy=None,
    log_base=10.0,
    fit_xlim=True,
    unclip_points=True,
    **kwargs
):
    """
    Plot a linear regression with confidence interval and prediction interval.
    If either axis is logarithmic, the fit is performed in the corresponding
    transformed (log) space so the plotted line is straight on the transformed axes.

    Args:
        x (array-like): x values
        y (array-like): y values
        ax (matplotlib.axes.Axes): axes to plot on
        show_ci (bool): plot confidence interval
        show_pi (bool): plot prediction interval
        ci_range (float): range of confidence interval (e.g., 0.95)
        return_model (bool): return model parameters
        extend_fraction (float): extend intervals by this fraction outside data range
        ci_kwargs (dict): passed to ax.fill_between for confidence interval
        pi_kwargs (dict): passed to ax.fill_between for prediction interval
        logx (bool or None): if True/False, force log-x behavior; if None, read from ax
        logy (bool or None): if True/False, force log-y behavior; if None, read from ax
        log_base (float): base used for log transform/inverse (default 10)
        fit_xlim (bool): if True (default), set ``ax.set_xlim`` to match the exact
            x-span of the regression/interval so bands run flush edge-to-edge.
        unclip_points (bool): if True (default), disable clipping (``clip_on=False``)
            on existing scatter/marker data points on ``ax`` so boundary points that
            define the interval edges render in full rather than being sliced off.
        kwargs: passed to ax.plot

    Returns:
        matplotlib.axes.Axes or (Axes, slope, intercept, r2, mse)
    """
    if ax is None:
        ax = plt.gca()

    # infer log flags from axes if not provided
    if logx is None:
        logx = hasattr(ax, "get_xscale") and (ax.get_xscale() == "log")
    if logy is None:
        logy = hasattr(ax, "get_yscale") and (ax.get_yscale() == "log")

    x, y = np.asarray(x), np.asarray(y)
    n, m = len(x), 2

    ci_kwargs0 = {"color": "gray", "alpha": 0.2, "edgecolor": "none"}
    pi_kwargs0 = {"color": "gray", "alpha": 0.1, "edgecolor": "none"}
    ci_kwargs0.update(ci_kwargs)
    pi_kwargs0.update(pi_kwargs)
    ci_kwargs, pi_kwargs = ci_kwargs0, pi_kwargs0

    # transforms
    ln_base = np.log(log_base)

    def _log(z):
        return np.log(z) / ln_base

    def _exp(z):
        return np.exp(z * ln_base)

    if logx and np.any(x <= 0):
        raise ValueError("logx=True requires all x > 0.")
    if logy and np.any(y <= 0):
        raise ValueError("logy=True requires all y > 0.")

    x_t = _log(x) if logx else x
    y_t = _log(y) if logy else y

    # fit in transformed space
    slope, intercept = np.polyfit(x_t, y_t, 1)
    y_model_t = np.polyval([slope, intercept], x_t)
    x_mean_t, y_mean_t = np.mean(x_t), np.mean(y_t)

    t = scipy.stats.t.ppf(ci_range, n - m)
    std_error_t = np.std(y_t - y_model_t) / np.sqrt(n / (n - m))
    r2 = 1 - np.sum((y_t - y_model_t) ** 2) / np.sum((y_t - y_mean_t) ** 2)
    mse = (1 / n) * np.sum((y_t - y_model_t) ** 2)

    # x-range for line/intervals (in transformed space)
    x_min, x_max = np.min(x), np.max(x)
    if logx:
        # multiplicative extension in data space, then transform
        x_min_ext = x_min * (1 - extend_fraction)
        x_max_ext = x_max * (1 + extend_fraction)
        if x_min_ext <= 0:
            x_min_ext = x_min  # keep valid; user asked for fraction, not hard guarantees
        x_line_t = np.linspace(_log(x_min_ext), _log(x_max_ext), 100)
        x_line = _exp(x_line_t)
    else:
        # preserve original sign-aware behavior
        if np.sign(x_min) == np.sign(x_max) and np.sign(x_max) > 0:
            x_line = np.linspace(x_min * (1 - extend_fraction), x_max * (1 + extend_fraction), 100)
        elif np.sign(x_min) == np.sign(x_max) and np.sign(x_max) < 0:
            x_line = np.linspace(x_min * (1 + extend_fraction), x_max * (1 - extend_fraction), 100)
        else:
            x_line = np.linspace(x_min * (1 + extend_fraction), x_max * (1 + extend_fraction), 100)
        x_line_t = x_line

    y_line_t = intercept + slope * x_line_t

    # intervals (computed in transformed space)
    sxx = np.sum((x_t - x_mean_t) ** 2)
    ci_t = t * std_error_t * (1 / n + (x_line_t - x_mean_t) ** 2 / sxx) ** 0.5
    pi_t = t * std_error_t * (1 + 1 / n + (x_line_t - x_mean_t) ** 2 / sxx) ** 0.5

    # inverse transform for plotting
    if logy:
        y_line = _exp(y_line_t)
        y_ci_lo, y_ci_hi = _exp(y_line_t - ci_t), _exp(y_line_t + ci_t)
        y_pi_lo, y_pi_hi = _exp(y_line_t - pi_t), _exp(y_line_t + pi_t)
    else:
        y_line = y_line_t
        y_ci_lo, y_ci_hi = y_line_t - ci_t, y_line_t + ci_t
        y_pi_lo, y_pi_hi = y_line_t - pi_t, y_line_t + pi_t

    ax.plot(x_line, y_line, **kwargs)
    if show_ci:
        ax.fill_between(x_line, y_ci_lo, y_ci_hi, **ci_kwargs)
    if show_pi:
        ax.fill_between(x_line, y_pi_lo, y_pi_hi, **pi_kwargs)

    if fit_xlim:
        ax.set_xlim(float(x_line[0]), float(x_line[-1]))
    if unclip_points:
        from matplotlib.collections import PathCollection

        for coll in ax.collections:
            if isinstance(coll, PathCollection):
                coll.set_clip_on(False)
        for line in ax.lines:
            if line.get_linestyle() in ("None", "none", "") and line.get_marker() not in (
                None,
                "",
                "None",
            ):
                line.set_clip_on(False)

    if not return_model:
        return ax
    return ax, slope, intercept, r2, mse


# def plot_linear_confidence(
#     x, y, ax=None, show_ci=True, show_pi=True, 
#     ci_range=0.95,
#     return_model=False, 
#     extend_fraction=0.0,
#     ci_kwargs={}, 
#     pi_kwargs={}, 
#     **kwargs
#     ):
#     """
#     Plot a linear regression with confidence interval and prediction interval.

#     Args:
#         x (array-like): x values
#         y (array-like): y values
#         ax (matplotlib.axes.Axes): axes to plot on
#         show_ci (bool): plot confidence interval
#         show_pi (bool): plot prediction interval
#         ci_range (float): range of confidence interval
#         return_model (bool): return model parameters
#         extend_fraction (float): extend confidence/prediction interval by this amount
#             outside of the data range
#         ci_kwargs (dict): passed to plt.fill_between for confidence interval
#         pi_kwargs (dict): passed to plt.fill_between for prediction interval
#         kwargs: passed to plt.plot

#     Returns:
#         matplotlib.axes.Axes: axes with plot
#     """
#     if ax is None:
#         ax = plt.gca()
#     x, y = np.asarray(x), np.asarray(y)
#     n, m = len(x), 2

#     ci_kwargs0 = {"color" : "gray", "alpha" : 0.2}
#     pi_kwargs0 = {"color" : "gray", "alpha" : 0.1}
#     ci_kwargs0.update(ci_kwargs)
#     pi_kwargs0.update(pi_kwargs)
#     ci_kwargs, pi_kwargs = ci_kwargs0, pi_kwargs0

#     slope, intercept = np.polyfit(x, y, 1)
#     y_model = np.polyval([slope, intercept], x)
#     x_mean, y_mean = np.mean(x), np.mean(y)
    
#     t = scipy.stats.t.ppf(ci_range, n - m) # Students statistic of interval confidence
#     std_error = np.std(y - y_model) / np.sqrt(n / (n - m)) # correct for dof
#     r2 = 1 - np.sum((y - y_model)**2) / np.sum((y - y_mean)**2)
#     mse = 1/n * np.sum( (y - y_model)**2 )

#     x_min, x_max = np.min(x), np.max(x)
#     if np.sign(x_min) == np.sign(x_max) and np.sign(x_max) > 0:
#         x_line = np.linspace(
#             np.min(x) * (1 - extend_fraction), 
#             np.max(x) * (1 + extend_fraction), 
#             100
#         )
#     elif np.sign(x_min) == np.sign(x_max) and np.sign(x_max) < 0:
#         x_line = np.linspace(
#             np.min(x) * (1 + extend_fraction), 
#             np.max(x) * (1 - extend_fraction), 
#             100
#         )
#     else:
#         x_line = np.linspace(
#             np.min(x) * (1 + extend_fraction), 
#             np.max(x) * (1 + extend_fraction), 
#             100
#         )
#     print("----\n", flush=True)
#     y_line = intercept + x_line * slope

#     # confidence interval
#     ci = t * std_error * (1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))**.5
#     # prediction interval
#     pi = t * std_error * (1 + 1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))**.5  

#     # plotting
#     ax.plot(x_line, y_line, **kwargs)
#     if show_ci:
#         ax.fill_between(x_line, y_line - ci, y_line + ci, **ci_kwargs)
#     if show_pi:
#         ax.fill_between(x_line, y_line - pi, y_line + pi, **pi_kwargs)
        
#     if not return_model:
#         return ax
#     else:
#         return ax, slope, intercept, r2, mse

def rle(inarray):
    """Run-length encoding returning ``(runlengths, startpositions, values)``."""
    ia = np.asarray(inarray)
    n = len(ia)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=ia.dtype)
    y = ia[1:] != ia[:-1]
    i = np.append(np.where(y)[0], n - 1)
    z = np.diff(np.append(-1, i))
    p = np.cumsum(np.append(0, z))[:-1]
    return z, p, ia[i]


def plot_segments(coords, mask, ax=None, **kwargs):
    """
    Given a set of coordinates, and a boolean mask, find consecutive 
    runs of coordinates and plot them as connected lines

    Args:
        coords (ndarray): The coordinates to plot, with shape (n, 2)
        mask (ndarray): A boolean mask with shape (n,) indicating which 
            coordinates to plot
        ax (matplotlib.axes.Axes): The axes to plot on
        **kwargs: passed to plt.plot

    Returns:
        ax (matplotlib.axes.Axes): The axes with the plot
    """
    if not ax:
        fig = plt.figure()
        ax = fig.gca()
    (runlengths, startpositions, values) = rle(mask)
    runlengths, startpositions = runlengths[values], startpositions[values]
    for start, run in zip(startpositions, runlengths):
        ax.plot(*coords[start:start+run].T, **kwargs)

    return ax


def scientific_ticks(ax=None, axis='y', **kwargs):
    """
    Set scientific notation on the axis.

    Args:
        ax (matplotlib.axis): axis to set the ticks on
        axis (str): 'x' or 'y'
        **kwargs: keyword arguments to pass to set_tick 
    
    """
    if ax is None:
        ax = plt.gca()
    
    if axis == 'x':
        ax.set_xticklabels(
            [f"$10^{{{int(y)}}}$" for y in plt.gca().get_xticks()], 
            **kwargs
        )
    elif axis == 'y':
        ax.set_yticklabels(
            [f"$10^{{{int(y)}}}$" for y in plt.gca().get_yticks()], 
            **kwargs
        )
    else:
        raise ValueError("axis must be 'x' or 'y'")

def dim_axes(ax=None, alpha=0.5):
    """
    Fade all ticks and frames on a plot. 

    Args:
        ax (matplotlib.axes.Axes): The axes to dim. If None, use the current axes.
        alpha (float): The alpha value to use for the dimming.

    """
    if ax is None:
        ax = plt.gca()
    ax.spines["bottom"].set_alpha(alpha)
    ax.spines["left"].set_alpha(alpha)
    ax.spines["top"].set_alpha(alpha)
    ax.spines["right"].set_alpha(alpha)
    ax.xaxis.label.set_alpha(alpha)
    ax.yaxis.label.set_alpha(alpha)
    ax.tick_params(axis="x", grid_alpha=alpha)
    ax.tick_params(axis="y", grid_alpha=alpha)
    ## dim the tick labels
    for tick in ax.get_xticklabels():
        tick.set_alpha(alpha)
    for tick in ax.get_yticklabels():
        tick.set_alpha(alpha)

    ## dim the tick lines
    for tick in ax.get_xticklines():
        tick.set_alpha(alpha)
    for tick in ax.get_yticklines():
        tick.set_alpha(alpha)


    ## for 3d plots
    if  ax.name == "3d":
        print("dimming 3d axes")
        for tick in ax.get_zticklabels():
            tick.set_alpha(alpha)
        for tick in ax.get_zticklines():
            tick.set_alpha(alpha)
        ## axes
        ax.w_xaxis.line.set_alpha(alpha)
        ax.w_yaxis.line.set_alpha(alpha)
        ax.w_zaxis.line.set_alpha(alpha)


def split_log(data, base=10):
	"""
	Map a dataset to split log coordinates, in order to better show
	dynamic range
	"""
	return np.sign(data) * np.log10(1 + np.abs(data))


import matplotlib.collections as mcoll

def colorline(
    x, y, z=None, cmap=None, norm=None,
    linewidth=None, alpha=1.0, ax=None, **kwargs):
    """
    Plot multiple line segments with a continuous color map.

    Args:
        x (array), y (array): Coordinates of shape ``(N, T)`` or ``(T,)``.
        z (array or None): Values mapped through ``cmap`` and ``norm``.
            Defaults to evenly spaced values in ``[0, 1]``.
        cmap (Colormap, str, or None): Matplotlib colormap or name. Defaults to
            the active ``rcParams["image.cmap"]`` (``vernalis`` under degas styles).
        norm (Normalize or None): Normalization for ``z``. Defaults to
            ``plt.Normalize(z.min(), z.max())`` when ``z`` is provided, else ``[0, 1]``.
        linewidth (float or None): Line width. Defaults to ``rcParams["lines.linewidth"]``.
        alpha (float): Line transparency (default ``1.0``).
        ax (matplotlib.axes.Axes or None): Axes to draw on. Defaults to ``plt.gca()``.
        **kwargs: Additional keyword arguments forwarded to ``LineCollection``.
        
    This multisegment function is adapted from several single-segment functions:
    https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, x.shape[-1])
        if norm is None:
            norm = plt.Normalize(0.0, 1.0)
    else:
        if not hasattr(z, "__iter__"):
            z = np.array([z])
        z = np.asarray(z)
        if norm is None:
            z_min, z_max = float(np.nanmin(z)), float(np.nanmax(z))
            norm = plt.Normalize(z_min, z_max if z_max > z_min else z_min + 1.0)

    if cmap is None:
        cmap = plt.get_cmap(plt.rcParams["image.cmap"])
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if linewidth is None:
        linewidth = plt.rcParams["lines.linewidth"]

    segments = make_segments(x, y)
    
    if len(segments.shape) == 4:
        z = np.tile(z, segments.shape[0])
        segments = np.reshape(segments, (-1, 2, 2))
    lc = mcoll.LineCollection(
        segments, array=z, cmap=cmap, norm=norm,
        linewidth=linewidth, alpha=alpha, **kwargs
    )

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)
    ax.autoscale_view()

    return lc




#############################################################
#
#
#   Utilities
#
#
############################################################

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for the LineCollection constructor
    
    Attributes:
        x (array), y (array): A list of x and y coordinates of shape (T,) 
            or (N_segments, T)
    
    Returns:
        segments (array): an array of the form numlines x (points per line) x 2 (x
            and y) array
            
    This multisegment function is adapted from a several single-segment function:
    https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
    """
    if len(np.squeeze(x).shape) > 1:
        #print("tt")
        points = np.dstack([x, y])
        #points = np.reshape(points, (x.shape[0]))
        #print(x.shape, points.shape)
        segments = np.stack((points[:, :-1, :], points[:, 1:, :]), axis=-1)
        segments = np.moveaxis(segments, (-1, -2), (-2, -1))
    else:
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def font_size(size, ax=None):
    """
    Set the font size of all text elements in a plot

    Args:
        size (int): The font size to set
        ax (matplotlib.axes.Axes): The axes to set the font size on. If None, use the
            current axes.

    Returns:
        ax (matplotlib.axes.Axes): The axes with the font size set
    """
    if not ax:
        ax = plt.gca()
    for item in ([ax.axes.title, ax.axes.xaxis.label, ax.axes.yaxis.label] +
                 ax.axes.get_xticklabels() + ax.axes.get_yticklabels()):
        item.set_fontsize(size)
    return ax


def lighter(clr, f=1/3):
    """
    Lighten an RGB color value. Based on Mathematica's Lighter[] function for RGB colors
    
    Args:
        clr (3-tuple): An RGB color
        f (float): The fraction by which to brighten the color
    
    Returns:
        new_clr (3-tuple): The new, lightened color
    """
    gaps = [f*(1 - val) for val in clr]
    new_clr = [val + gap for gap, val in zip(gaps, clr)]
    return new_clr

def darker(clr, f=1/3):
    """
    Darken an RGB color value. Based on Mathematica's Darker[] function for RGB colors

    Args:
        clr (3-tuple): An RGB color
        f (float): The fraction by which to darken the color

    Returns:
        new_clr (3-tuple): The new, darkened color
    """
    gaps = [f*val for val in clr]
    new_clr = [val - gap for gap, val in zip(gaps, clr)]
    return new_clr

def cmap1D(col1, col2, N):
    """
    Generate a continuous colormap between two values
    
    Args:
        col1 (3-tuple): The RGB color at the start of the colormap
        col2 (3-tuple): The RGB color at the end of the colormap
        N (int): The number of colors to interpolate between col1 and col2
    
    Returns:
        col_list (list): A list of RGB colors interpolated between col1 and col2
    """
    col1 = np.array([item/255. for item in col1])
    col2 = np.array([item/255. for item in col2])
    
    vr = list()
    for ii in range(3):
        vr.append(np.linspace(col1[ii],col2[ii],N))
    colist = np.array(vr).T
    return [tuple(thing) for thing in colist]

def color_sort(color_list, method="hue", reverse=False):
    """
    Given a list of RGB colors, sort the colors by their hue.

    Args:
        color_list (list): A list of RGB colors, each of which is a 3-tuple
            of floats between 0 and 1.
        method (str): The method by which to sort the colors. Can be "hue",
            "red", "green", "blue", or "luminance". Defaults to "hue".
        reverse (bool): Whether to reverse the order of the colors. Defaults
            to False.

    Returns:
        list: A list of colors sorted by the specified method.

    """
    if method == "hue":
        from matplotlib.colors import rgb_to_hsv
        hsv_colors = rgb_to_hsv(color_list)
        hue = hsv_colors[:, 0]
        sorted_indices = hue.argsort()
    elif method == "red":
        red = color_list[:, 0]
        sorted_indices = red.argsort()
    elif method == "green":
        green = color_list[:, 1]
        sorted_indices = green.argsort()
    elif method == "blue":
        blue = color_list[:, 2]
        sorted_indices = blue.argsort()
    elif method == "luminance":
        from matplotlib.colors import rgb_to_hsv
        hsv_colors = rgb_to_hsv(color_list)
        luminance = hsv_colors[:, 2]
        sorted_indices = luminance.argsort()
    elif method == "step":
        from matplotlib.colors import rgb_to_hsv
        hsv_colors = rgb_to_hsv(color_list)
        hue = hsv_colors[:, 0]
        sorted_indices = hue.argsort()
        sorted_indices = np.concatenate([sorted_indices[::2], sorted_indices[1::2]])
    else:
        sorted_indices = np.arange(len(color_list))
    if reverse:
        sorted_indices = sorted_indices[::-1]
    return np.array(color_list)[sorted_indices]

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def make_linear_cmap(color_list, name="CustomColormap", alpha=None):
    """
    Create a linear interpolating colormap from a list of colors. For now, stick to
    using the built-in:
    
    from matplotlib.colors import LinearSegmentedColormap
    LinearSegmentedColormap.from_list("CustomColormap", color_list)
    
    Attributes:
        color_list (list): List of (R, G, B) values scaled between 0 and 1
        alpha (None or array): List of alpha (transparency) values for each 
            color in the gradient
        
    Returns:
        cmap: A matplotlib LinearSegmentedColormap object
        
    Development:
        "alpha" is not working, likely due to draworder issues
    
    """
    
    return LinearSegmentedColormap.from_list(name, color_list)
    
    # ## code below is for development purposes
    
    # n = len(color_list)
    # color_list = np.array(color_list)
    
    # linear_gradient = np.linspace(0, 1, n)
    # template = np.zeros((n, 3))
    
    
    # template[:, 0] = linear_gradient
    
    # template_r, template_g, template_b = (np.copy(template), 
    #                                       np.copy(template), 
    #                                       np.copy(template))
    
    # #print(template_r, color_list[0, :])
    # template_r[:, 1:3] = color_list[:, 0][:, None]
    # template_g[:, 1:3] = color_list[:, 1][:, None]
    # template_b[:, 1:3] = color_list[:, 2][:, None]
    
    
    # cdict = {'red': template_r,
    #      'green': template_g,
    #      'blue': template_b
    #     }
    
    # if alpha:
    #     #print("t")
    #     cdict["alpha"] = np.vstack([linear_gradient, linear_gradient, linear_gradient]).T
    # #print(cdict)
    # cmap =  LinearSegmentedColormap(name=name, segmentdata=cdict)
    # return cmap




def coords_to_image(x, y, z, **kwargs):
    """Given a list of x, y, z values, interpolate onto a regular grid for plotting as 
    an image. This function is based on Mathematica's ListDensityPlot
    
    Args:
        x, y, z (np.ndarray): The x, y, z values to interpolate, all of shape (N,)
        **kwargs: Additional keyword arguments to pass to scipy.interpolate.griddata, 
            such as the interpolation order and filling rules
    
    Returns:
        pp_im (np.ndarray): The image of the interpolated values, of shape (D1, D2)
    """
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    xs, ys = (
        np.median(np.diff(x_sorted)[np.diff(x_sorted)>0]), 
        np.median(np.diff(y_sorted)[np.diff(y_sorted)>0])
    )
    xlo, xhi = np.min(x), np.max(x)
    ylo, yhi = np.min(y), np.max(y)
    
    nx, ny = int(1 + np.ceil((xhi - xlo)/xs)), int(1 + np.ceil((yhi - ylo)/ys))
    xp, yp = np.linspace(xlo, xhi, nx), np.linspace(ylo, yhi, ny)
    
    xx, yy = np.meshgrid(xp, yp)
    
    grid_vals = np.vstack([np.ravel(item) for item in [xx, yy]]).T
    grid_vals = np.vstack([np.ravel(item) for item in [xx, yy]]).T
    pp = griddata(np.vstack([x, y]).T, z, grid_vals, "nearest")
    pp_im = np.reshape(pp, (ny, nx))
    return pp_im

#############################################################
#
#
#   I/O utilities
#
#
############################################################


def vanish_axes(gca=None):
    """Make all axes disappear from a plot
    
    Args:
        gca (matplotlib.axes.Axes): The axes to remove. If None, use the current axes.
    """
    if not gca:
        gca = plt.gca()
    gca.set_axis_off()
    gca.xaxis.set_major_locator(plt.NullLocator())
    gca.yaxis.set_major_locator(plt.NullLocator())


def savefig_exact(arr, file_name, target_shape=None, dpi=400, **kwargs):
    """
    Save a figure to an exact size in pixels

    Args:
        arr (np.ndarray): The array to plot
        file_name (str): The file name to save to
        target_shape (tuple): The target shape of the figure in pixels
        dpi (int): The dpi to use
        **kwargs: Additional arguments to pass to plt.imshow

    """
    if target_shape is not None:
        if has_scipy:
            from scipy import signal
            arr = signal.resample(arr, target_shape[0], axis=0)
            arr = signal.resample(arr, target_shape[1], axis=1)
        else:
            warnings.warn("Scipy not installed; skipping resampling")

    fig = plt.figure(frameon=False)
    fig.set_size_inches(arr.shape[1] / dpi, arr.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr, **kwargs)
    plt.savefig(file_name, dpi=(dpi ))


from datetime import datetime
import os
def better_savefig(
        name, 
        dpi=300, 
        pad=0.0, 
        pad_inches=0.02, 
        remove_border=False, 
        dryrun=False,
        file_format=None,
        unique_tag=False,
        **kwargs
    ):
    """This function is for saving images without a bounding box and at the proper resolution
        The tiff files produced are huge because compression is not currently supported by 
        matplotlib
    
    Args:
        name (str): The string containing the name of the desired save file and its 
            resolution
        dpi (int): The desired dots per linear inch
        pad (float): Add a tiny amount of whitespace if necessary
        remove_border (bool): Whether to remove axes and padding (e.g. for images)
        file_format (str or None): The file format to save the figure as. If a format is 
            specified in the name, this will override it.
        unique_tag (bool): Tag a unique date and time to the file name
        dryrun (bool): If True, don't actually save the file
        **kwargs: passed on to matplotlib's built-in "savefig" function

    """
    if remove_border:
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1+pad, bottom = 0+pad, right = 1+pad, left = 0+pad, 
                    hspace = 0, wspace = 0)
        plt.margins(0+pad,0+pad)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # Split name into name and extenstion
    fname, fext = os.path.splitext(name)
    if unique_tag:
        fname = fname + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    name = fname + fext


    if not dryrun:
        plt.savefig(name, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi, **kwargs)

# Blue-Black-Red colormap
blbkrd = make_linear_cmap([blue, 'k', red])

def plt_sphere(pts, radius, ax=None):
    """
    Plot a set of spheres with given centers and radii.

    Args:
        pts (ndarray): the centers of the spheres
        radius (float): the radius of the spheres
        ax (Axes3D): the axes on which to plot the spheres

    Returns:
        ax (Axes3D): the axes on which the spheres were plotted

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for pt in pts:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v)) + pt[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + pt[1]  
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pt[2]
        # Plot sphere surfaces    
        ax.plot_surface(x, y, z, color='b', alpha=0.8)
    
    return ax

def save_colormap(cmap, filename, **kwargs):
    """
    Save a colormap as an image file.

    Args:
        cmap: The colormap to save (can be a string or a matplotlib colormap object).
        filename: The name of the file to save the colormap to.
        **kwargs: Additional keyword arguments to pass to plt.savefig().

    """
    plt.figure(figsize=(10, 0.5))
    plt.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=cmap, aspect='auto')
    vanish_axes()
    better_savefig(filename, **kwargs)
    plt.close()



def sliderplot(arr, figsize=(6, 6), width='100%', **kwargs):
    """
    Given a 3-dimensional array, plot the last two dimensions and make a slider along
    the first dimension.

    Args:
        arr (ndarray): a 3-dimensional array
        figsize (tuple): the size of the figure
        width (str): the width of the slider
        **kwargs: keyword arguments to pass to plt.imshow
    """
    from ipywidgets import interact, IntSlider, Layout

    n = arr.shape[0]
    def plotter(i):
        fig = plt.figure(figsize=figsize)
        plt.imshow(arr[i], **kwargs)
        plt.show()
    interact(
        plotter, 
        i=IntSlider(
            min=0, max=n-1, step=1, value=0, layout=Layout(width=width)
        )
    )



#############################################################
#
#
#   Image analysis
#
#
############################################################



def crop_by_value(image, target=1.0):
    """
    Crops an image to only include the columns where the pixel value is 1.0.
    
    Args:
        image (numpy.ndarray): The input image to be cropped.

    Returns:
        numpy.ndarray: The cropped image.
    """
    # Find columns where all pixel values are 1.0
    mask = np.logical_not(np.all(image == target, axis=0))
    
    # Find the indices of the first and last columns that satisfy the condition
    leftmost = np.argmax(mask)
    rightmost = len(mask) - np.argmax(mask[::-1]) - 1
    
    # Crop the image based on these indices
    cropped_image = image[:, leftmost:rightmost+1]
    
    return cropped_image





def save_matrix_pixels(A, filename="matrix.png", **kwargs):
    """
    Save a matrix as an image with exactly one output pixel per matrix element.

    Args:
        A (array-like): Matrix with shape (M, M) or (N, M).
        filename (str, optional): Output image filename.
        **kwargs: Additional arguments to pass to plt.imshow

    Returns:
        None
    """
    h, w = A.shape

    fig = plt.figure(figsize=(w, h), dpi=1, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(A, origin="lower", interpolation="nearest", aspect="auto", **kwargs)
    ax.set_axis_off()
    fig.savefig(filename, dpi=1, bbox_inches=None, pad_inches=0)
    plt.close(fig)

from scipy.interpolate import griddata
def xyz_to_image(x, y, z, method="linear", fill_value=np.nan):
    """
    Convert scattered x, y, z samples into an image matrix for imshow.

    Args:
        x (array_like): x coordinates.
        y (array_like): y coordinates.
        z (array_like): z values.
        method (str, optional): Interpolation method. Options are
            "nan", "nearest", "zero", "linear", and "cubic". "zero" is
            equivalent to 0th-order nearest-neighbor interpolation.
        fill_value (float, optional): Value used outside the convex hull for
            "linear" and "cubic" interpolation.

    Returns:
        (tuple): A tuple ``(img, extent, xi, yi)``, where ``img`` is the
        2D image matrix, ``extent`` is suitable for ``imshow(..., extent=extent)``,
        and ``xi``, ``yi`` are the grid coordinates.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    dx = np.min(np.diff(np.unique(np.sort(x))))
    dy = np.min(np.diff(np.unique(np.sort(y))))

    xi = np.arange(x.min(), x.max() + 0.5 * dx, dx)
    yi = np.arange(y.min(), y.max() + 0.5 * dy, dy)

    X, Y = np.meshgrid(xi, yi)

    img = np.full(X.shape, np.nan, dtype=float)

    x_idx = np.rint((x - xi[0]) / dx).astype(int)
    y_idx = np.rint((y - yi[0]) / dy).astype(int)
    img[y_idx, x_idx] = z

    if method == "nan":
        pass
    elif method in {"nearest", "zero"}:
        missing = np.isnan(img)
        img[missing] = griddata((x, y), z, (X[missing], Y[missing]), method="nearest")
    elif method in {"linear", "cubic"}:
        missing = np.isnan(img)
        img[missing] = griddata(
            (x, y),
            z,
            (X[missing], Y[missing]),
            method=method,
            fill_value=fill_value,
        )
    else:
        raise ValueError(
            "method must be one of {'nan', 'nearest', 'zero', 'linear', 'cubic'}"
        )

    extent = (xi[0] - dx / 2, xi[-1] + dx / 2, yi[0] - dy / 2, yi[-1] + dy / 2)

    return img, extent, xi, yi