import os, warnings
import numpy as np
import matplotlib.pyplot as plt


try:
    import scipy
    from scipy.interpolate import griddata
    has_scipy = True
except ImportError:
    has_scipy = False


## TODO
## Easy split y axis function



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
royal_purple = np.array((120, 81, 169))/255.

# Blue-Black-Red colormap
blbkrd = make_linear_cmap([blue, 'k', red])

style_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "styles")
def set_style(style_name="default"):
    """
    Set the style of the plot to one of the styles in the styles folder. This wraps
    matplotlib's style.use function.
    """
    plt.style.use(os.path.join(style_path, style_name + ".mplstyle"))


def plot_splity(x, y1, y2, ax=None, kwargs1=None, kwargs2=None):
    """
    Plot two y-axes on the same plot with a split y axis. This is useful for
    plotting two quantities with very different scales on the same plot. Separate ticks
    and labels are used for each y axis
    """
    if not ax:
        ax = plt.gca()
    if not kwargs1:
        kwargs1 = {}
    if not kwargs2:
        kwargs2 = {}
    ax.plot(x, y1, **kwargs1)
    ax2 = ax.twinx()
    ax2.plot(x, y2, **kwargs2)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax2.patch.set_visible(True)
    return ax, ax2

def fixed_aspect_ratio(ratio, ax=None, 
	log=False, semilogy=False, semilogx=False):
    '''
    Set a fixed aspect ratio on matplotlib plots 
    regardless of axis units
    '''
    if not ax:
        ax = plt.gca()
    xvals, yvals = ax.axes.get_xlim(), ax.axes.get_ylim()
    xrange = xvals[1] - xvals[0]
    yrange = yvals[1] - yvals[0]
    if log:
        xrange = np.log10(xvals[1]) - np.log10(xvals[0])
        yrange = np.log10(yvals[1]) - np.log10(yvals[0])
    if semilogy:
        yrange = np.log10(yvals[1]) - np.log10(yvals[0])
    if semilogx:
        xrange = np.log10(xvals[1]) - np.log10(xvals[0])
    try:
        ax.set_aspect(ratio*(xrange/yrange), adjustable='box')
    except NotImplementedError:
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

from matplotlib import patches
def draw_ellipse(points, ax=None, **kwargs):
    """
    Draw an ellipse around a cluster of points in 2D.
    
    Args:
        ax (matplotlib.axes.Axes): The axes to draw the ellipse on.
        points (ndarray): The points in a cluster to enclose with an ellipse, containing n
                          ndarray elements representing each point, each with d elements
                          representing the coordinates for the point.
        **kwargs: Additional keyword arguments to pass to matplotlib.patches.Ellipse.

    Returns:
        ellipse (matplotlib.patches.Ellipse): The ellipse drawn on the axes.

    Example:

        plt.figure()
        all_pairs = np.random.random((100, 2))
        all_pairs = all_pairs @ np.linalg.inv(np.random.random((2, 2)))
        plt.scatter(all_pairs[:, 0], all_pairs[:, 1])
        draw_ellipse(plt.gca(), all_pairs, fill=False, edgecolor="red")
    """
    if ax is None:
        ax = plt.gca()
    width, height, angle = fit_ellipse(points)
    center = tuple(points.mean(axis=0))
    ellipse = patches.Ellipse(center, width, height, angle, **kwargs)
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

def plot_err(y, 
    errs, 
    x=[], 
    color=(0,0,0), 
    alpha=.4, 
    linewidth=1, 
    loglog=False,
    **kwargs):
    """
    Plot a curve with error bars

    Args:
        y (array): A list of values to plot
        errs (array): A list of errors, or a pair of lists of upper and lower errors 
        x (array): A list of x positions
        color (3-tuple): The color of the plot lines and error bars
        alpha (float): The transparency level of the error bars
        kwargs: passed to plot

    Returns:
        ax (matplotlib.axes.Axes): The axes on which the plot was drawn

    Example:
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        errs = np.random.random(100) * .1
        plot_err(y, errs, x=x, color=(0,0,1))

    """
    if len(x) < 1:
        x = np.arange(len(y))
    
    if len(errs.shape) > 1:
        if errs.shape[1] == 2:
            err_lo, err_hi = errs[:, 0], errs[:, 1]
    else:
        err_lo = errs
        err_hi = errs
        
    trace_lo, trace_hi = y - err_lo, y + err_hi
    
    plt.fill_between(x, trace_lo, trace_hi, color=lighter(color), alpha=alpha)
    plt.plot(x, y, color=color, linewidth=linewidth, **kwargs)
    if loglog:
        plt.yscale('log', nonposy='clip')
        plt.xscale('log', nonposy='clip')
    # return ax

def plot3dproj(x, y, z, *args, 
    ax=None,
    color=(0,0,0), 
    shadow_dist=1.0, 
    color_proj=None, 
    elev_azim=(39,-47), 
    show_labels=False, 
    aspect_ratio=1.0,
    **kwargs):
    """
    Create a three dimensional plot, with projections onto the 2D coordinate
    planes
    
    Parameters
    ----------
    x, y, z : 1D arrays of coordinates to plot
    *args : arguments passed to the matplotlib plt.plot functions
    color : length-3 tuple
        The RGB color (with each element in [0,1]) to use for the
        three dimensional line plot
    color_proj : length-3 tuple
        The RGB color (with each element in [0,1]) to use for the
        two dimensional projection plots. Defaults to a lighter version of the 
        plotting color
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
    if ax is None:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection= '3d')
    if color_proj is None:
        color_proj = lighter(color, .6)

    if np.isscalar(shadow_dist) == 1:
        sdist_x = shadow_dist
        sdist_y = shadow_dist
        sdist_z = shadow_dist
    else:
        sdist_x, sdist_y, sdist_z = shadow_dist

    ax.plot(x, z, *args, zdir='y', zs=sdist_y*np.max(y), color=color_proj, **kwargs)
    ax.plot(y, z, *args, zdir='x', zs=sdist_x*np.min(x), color=color_proj, **kwargs)
    ax.plot(x, y, *args, zdir='z', zs=sdist_z*np.min(z), color=color_proj, **kwargs)
    ax.plot(x, y, z, *args, color=color, **kwargs)

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

# def plot_err(y, errs, 
#     color=(0,0,0), 
#     x=[], 
#     alpha=.4, 
#     linewidth=1, 
#     ax=None,
#     **kwargs):
#     """
#     Plot a curve with error bars

#     Args:
#         y (array): A list of values
#         errs (array): A list of errors, or a pair of lists of upper and lower errors
#         x (array): A list of x positions
#         color (3-tuple): The color of the plot lines and error bars
#         alpha (float): The transparency level of the error bars
#         kwargs: passed to plot

#     """
#     if not ax:
#         ax = plt.gca()
#     if len(x)<1:
#         x = np.arange(len(y))
    
#     if len(errs.shape)>1:
#         if errs.shape[1]==2:
#             err_lo, err_hi = errs[:, 0], errs[:, 1]
#     else:
#         err_lo = errs
#         err_hi = errs
        
#     trace_lo, trace_hi = y - err_lo, y + err_hi
    
#     plt.fill_between(x, trace_lo, trace_hi, color=lighter(color), alpha=alpha)
#     plt.plot(x, y, color=color, linewidth=linewidth, **kwargs)

#     return ax

def plot_linear_confidence(
    x, y, ax=None, show_ci=True, show_pi=True, 
    ci_range=0.95,
    return_model=False, 
    extend_fraction=0.0,
    ci_kwargs={}, 
    pi_kwargs={}, 
    **kwargs
    ):
    """
    Plot a linear regression with confidence interval and prediction interval.

    Args:
        x (array-like): x values
        y (array-like): y values
        ax (matplotlib.axes.Axes): axes to plot on
        show_ci (bool): plot confidence interval
        show_pi (bool): plot prediction interval
        ci_range (float): range of confidence interval
        return_model (bool): return model parameters
        extend_fraction (float): extend confidence/prediction interval by this amount
            outside of the data range
        ci_kwargs (dict): passed to plt.fill_between for confidence interval
        pi_kwargs (dict): passed to plt.fill_between for prediction interval
        kwargs: passed to plt.plot

    Returns:
        matplotlib.axes.Axes: axes with plot
    """
    if ax is None:
        ax = plt.gca()
    x, y = np.asarray(x), np.asarray(y)
    n, m = len(x), 2

    ci_kwargs0 = {"color" : "gray", "alpha" : 0.2}
    pi_kwargs0 = {"color" : "gray", "alpha" : 0.1}
    ci_kwargs0.update(ci_kwargs)
    pi_kwargs0.update(pi_kwargs)
    ci_kwargs, pi_kwargs = ci_kwargs0, pi_kwargs0

    slope, intercept = np.polyfit(x, y, 1)
    y_model = np.polyval([slope, intercept], x)
    x_mean, y_mean = np.mean(x), np.mean(y)
    
    t = scipy.stats.t.ppf(ci_range, n - m) # Students statistic of interval confidence
    std_error = np.std(y - y_model) / np.sqrt(n / (n - m)) # correct for dof
    r2 = 1 - np.sum((y - y_model)**2) / np.sum((y - y_mean)**2)
    mse = 1/n * np.sum( (y - y_model)**2 )

    x_min, x_max = np.min(x), np.max(x)
    if np.sign(x_min) == np.sign(x_max) and np.sign(x_max) > 0:
        x_line = np.linspace(
            np.min(x) * (1 - extend_fraction), 
            np.max(x) * (1 + extend_fraction), 
            100
        )
    elif np.sign(x_min) == np.sign(x_max) and np.sign(x_max) < 0:
        x_line = np.linspace(
            np.min(x) * (1 + extend_fraction), 
            np.max(x) * (1 - extend_fraction), 
            100
        )
    else:
        x_line = np.linspace(
            np.min(x) * (1 + extend_fraction), 
            np.max(x) * (1 + extend_fraction), 
            100
        )
    print("----\n", flush=True)
    y_line = intercept + x_line * slope

    # confidence interval
    ci = t * std_error * (1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))**.5
    # prediction interval
    pi = t * std_error * (1 + 1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))**.5  

    # plotting
    ax.plot(x_line, y_line, **kwargs)
    if show_ci:
        ax.fill_between(x_line, y_line - ci, y_line + ci, **ci_kwargs)
    if show_pi:
        ax.fill_between(x_line, y_line - pi, y_line + pi, **pi_kwargs)
        
    if not return_model:
        return ax
    else:
        return ax, slope, intercept, r2, mse

def plot_segments(coords, mask, ax=None, **kwargs):
    """
    Given a set of coordinates, and a boolean mask, find consecutive 
    runs of coordinates and plot them as connected lines
    """
    if not ax:
        fig = plt.figure()
        ax = fig.gca()
    (runlengths, startpositions, values) = rle(mask)
    runlengths, startpositions = runlengths[values], startpositions[values]
    for start, run in zip(startpositions, runlengths):
        ax.plot(*coords[start:start+run].T, **kwargs)

    return ax

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
import matplotlib.path as mpath

def colorline(
    x, y, z=None, cmap=plt.get_cmap('viridis'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    Plot multiple line segments with a continuous color map
    Attributes:
        x (array), y (array): Lists of values of shape (N, T) or (T)
        z (array): A list of values between 0 and 1, corresponding to the color map 
            locations
        norm: The normalization of the colormap
        cmap: A matplotlib colormap object
        **kwargs: keyword arguments passed to the LineCollection object. Accepts 
            standard line properties typically passed to plt.plot
        
    This multisegment function is adapted from several single-segment functions:
    https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, x.shape[-1])

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    
    if len(segments.shape) == 4:
        z = np.tile(z, segments.shape[0])
        segments = np.reshape(segments, (-1, 2, 2))
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

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
    size : int
        Font size in pts
    """
    if not ax:
        ax = plt.gca()
    for item in ([ax.axes.title, ax.axes.xaxis.label, ax.axes.yaxis.label] +
                 ax.axes.get_xticklabels() + ax.axes.get_yticklabels()):
        item.set_fontsize(size)


def lighter(clr, f=1/3):
    """
    An implementation of Mathematica's Lighter[] 
    function for RGB colors
    clr : 3-tuple or list, an RGB color
    f : float, the fraction by which to brighten
    """
    gaps = [f*(1 - val) for val in clr]
    new_clr = [val + gap for gap, val in zip(gaps, clr)]
    return new_clr

def darker(clr, f=1/3):
    """
    An implementation of Mathematica's Darker[] 
    function for RGB colors
    clr : 3-tuple or list, an RGB color
    f : float, the fraction by which to brighten
    """
    gaps = [f*val for val in clr]
    new_clr = [val - gap for gap, val in zip(gaps, clr)]
    return new_clr

def cmap1D(col1, col2, N):
    """
    Generate a continuous colormap between two values
    
    Parameters
    ----------
    col1 : tuple of ints
        RGB values of final color
        
    col2 : tuple of ints
        RGB values of final color
    
    N : int
        The number of values to interpolate
        
    Returns
    -------
    col_list : list of tuples
        An ordered list of colors for the colormap
    
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
    """Given a list of x, y, z values, interpolate onto 
    a regular grid for plotting as an image
    
    Parameters
    ----------
    x, y, z : N x 1
        Lists of data coordinates
    
    kwargs : int
        Parameters passed to scipy.interpolate.griddata,
        such as the interpolation order and filling rules
        
    Returns
    -------
    pp_im : D1 x D2 ndarray
        An image created from the 2D coordinates
    
    Notes
    -----
    
    Based on Mathematica's ListDensityPlot
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
    """Make all axes disappear from a plot"""
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
            arr = scipy.signal.resample(arr, target_shape[0], axis=0)
            arr = scipy.signal.resample(arr, target_shape[1], axis=1)
        else:
            warnings,warn("Scipy not installed; skipping resampling")

    fig = plt.figure(frameon=False)
    fig.set_size_inches(arr.shape[1] / dpi, arr.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr, **kwargs)
    plt.savefig(f_name, dpi=(dpi ))



def better_savefig(
        name, 
        dpi=300, 
        pad=0.0, 
        pad_inches=0.02, 
        remove_border=False, 
        dryrun=False,
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

    if not dryrun:
        plt.savefig(name, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi)

