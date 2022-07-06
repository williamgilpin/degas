import os, warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


## TODO
## Easy split y axis function

# degas high contrast color scheme
blue, red, turquoise, purple, magenta, orange, gray  = [[0.372549, 0.596078, 1], 
                                                  [1.0, .3882, .2784], 
                                                  [0.20784314, 0.67843137, 0.6], 
                                                  [0.59607843, 0.25882353, 0.89019608],
                                                  [0.803922, 0.0627451, 0.462745], 
                                                  [0.917647, 0.682353, 0.105882],
                                                  [0.7, 0.7, 0.7]
                                                  ]


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

# degas line plot colors
royal_purple = np.array((120, 81, 169))/255.

style_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "styles")
def set_style(style_name="default"):
    plt.style.use(os.path.join(style_path, style_name + ".mplstyle"))


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

def plot_err(y, errs, 
    color=(0,0,0), 
    x=[], 
    alpha=.4, 
    linewidth=1, 
    log=False,
    **kwargs):
    """
    errs : Nx1 or Nx2 ndarray
    kwargs : passed to plot
    """
    if len(x) < 1:
        x = np.arange(len(y))
    
    if len(errs.shape)>1:
        if errs.shape[1]==2:
            err_lo, err_hi = errs[:, 0], errs[:, 1]
    else:
        err_lo = errs
        err_hi = errs
        
    trace_lo, trace_hi = y - err_lo, y + err_hi
    
    plt.fill_between(x, trace_lo, trace_hi, color=lighter(color), alpha=alpha)
    plt.plot(x, y, color=color, linewidth=linewidth, **kwargs)
    if log:
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

def plot_err(y, errs, 
    color=(0,0,0), 
    x=[], 
    alpha=.4, 
    linewidth=1, 
    ax=None,
    **kwargs):
    """
    errs : Nx1 or Nx2 ndarray
    kwargs : passed to plot
    """
    if not ax:
        ax = plt.gca()
    if len(x)<1:
        x = np.arange(len(y))
    
    if len(errs.shape)>1:
        if errs.shape[1]==2:
            err_lo, err_hi = errs[:, 0], errs[:, 1]
    else:
        err_lo = errs
        err_hi = errs
        
    trace_lo, trace_hi = y - err_lo, y + err_hi
    
    plt.fill_between(x, trace_lo, trace_hi, color=lighter(color), alpha=alpha)
    plt.plot(x, y, color=color, linewidth=linewidth, **kwargs)

    return ax

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


def better_savefig(name, dpi=300, pad=0.0, pad_inches=0.02, remove_border=False, **kwargs):
    """This function is for saving images without a bounding box and at the proper resolution
        The tiff files produced are huge because compression is not currently supported by 
        matplotlib
    
    Parameters
    ----------
    name : str
        The string containing the name of the desired save file and its resolution
        
    dpi : int
        The desired dots per linear inch
    
    pad : float
        Add a tiny amount of whitespace if necessary
        
    remove_border : bool
        Whether to remove axes and padding (for example, for images)

    kwargs : passed on to matplotlib's built-in "savefig" function
    
    """
    if remove_border:
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1+pad, bottom = 0+pad, right = 1+pad, left = 0+pad, 
                    hspace = 0, wspace = 0)
        plt.margins(0+pad,0+pad)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig(name, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi)

