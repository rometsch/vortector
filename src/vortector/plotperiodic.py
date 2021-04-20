import numpy as np


def plot_periodic(ax, x, y, m=None, bnd=(-np.pi, np.pi), **kwargs):
    if m is not None:
        plot_periodic_mask(ax, x, y, m, **kwargs)
        return
    L = bnd[1] - bnd[0]
    m = np.logical_and(x <= bnd[1], x >= bnd[0])
    line, = ax.plot(x[m], y[m], **kwargs)
    kwa = kwargs.copy()
    kwa["color"] = line.get_color()
    kwa["ls"] = line.get_linestyle()

    m = x < bnd[0]
    line, = ax.plot(x[m] + L, y[m], **kwa)
    line.set_label(None)

    m = x > bnd[1]
    line, = ax.plot(x[m] - L, y[m], **kwa)
    line.set_label(None)


def plot_vline_periodic(ax, x, y, dy, **kwargs):
    bup = np.pi
    blow = -np.pi
    L = bup - blow
    y = (y-blow) % L + blow

    at_upper_bnd = y + dy > bup
    at_lower_bnd = y - dy < blow
    if at_upper_bnd:
        line, = ax.plot([x, x], [y-dy, bup], **kwargs)
        c = line.get_color()
        ls = line.get_linestyle()
        lw = line.get_linewidth()
        ax.plot([x, x], [blow, y-L+dy], ls=ls, lw=lw, c=c)
    elif at_lower_bnd:
        line, = ax.plot([x, x], [y+dy, blow], **kwargs)
        c = line.get_color()
        ls = line.get_linestyle()
        lw = line.get_linewidth()
        ax.plot([x, x], [bup, y+L-dy], ls=ls, lw=lw, c=c)
    else:
        ax.plot([x, x], [y-dy, y+dy], **kwargs)


def plot_periodic_mask(ax, x, y, m, **kwargs):
    #     print(m)
    if m[0] and m[-1] and not all(m):
        bnd = np.where(m == False)[0][0]
        xl = x[:bnd]
        yl = y[:bnd]
        line, = ax.plot(xl, yl, **kwargs)
        bnd = np.where(m == False)[0][-1]
#         print(len(x))
#         print(bnd)
        xr = x[bnd:]
        yr = y[bnd:]
        kwa = kwargs.copy()
        kwa["color"] = line.get_color()
        kwa["ls"] = line.get_linestyle()
        line, = ax.plot(xr, yr, **kwa)
        line.set_label(None)
    else:
        ax.plot(x[m], y[m], **kwargs)


def combine_periodic(x, y, m, bnd=(-np.pi, np.pi)):
    """ Combine an array split at a periodic boundary. 

    This is used for vortices that stretch over the periodic boundary
    in azimuthal directions in disks.
    The x values at the left boundary are appended to the right of the
    values at the right boundary and the periodicity is added.

    The following sketch shows how the arrays are combined.

    |+++++_________xxx|
    to
    |______________xxx|+++++

    Parameters
    ----------
    x: array
        Coordinate values (full domain).
    y: array
        Values (full domain).
    m: array (boolean)
        Mask to select the active values.
    bnd: tuple of two float
        Position of the left and right boundary.

    Returns
    -------
    x : array
        coordinates
    y : array
        values
    """
    lb, rb = bnd
    if m[0] and m[-1] and not all(m):
        b = np.where(m == False)[0][0]
        xl = x[:b]
        yl = y[:b]
        b = np.where(m == False)[0][-1]
        xr = x[b:]
        yr = y[b:]
        xcom = np.append(xr, xl+(rb-lb))
        ycom = np.append(yr, yl)
        return xcom, ycom
    else:
        return x[m], y[m]


def plot_ellipse_periodic(ax, x, y, w, h, crosshair=False, bnd=(-np.pi, np.pi), **kwargs):
    """ Show an Ellipse in a plot periodic in y direction.

    Parameters
    ----------
    ax : plt.axes
        Axes to plot on.
    x : float
        Center in x.
    y : float
        Center in y.
    w : float
        Width in x.
    h : float
        Height in y.
    crosshair : bool
        Show a cross indicating the center.
    bnd : (float, float)
        Locations of the periodic boundary.
    """
    from matplotlib.patches import Ellipse
    L = bnd[1] - bnd[0]
    C = 0.5*(bnd[1] + bnd[0])

    plot_args = kwargs.copy()
    if "color" in kwargs:
        plot_args["edgecolor"] = kwargs["color"]
        del plot_args["color"]
    else:
        kwargs["edgecolor"] = "C0"
    e = Ellipse(xy=[x, y], width=w, height=h,
                fc="None", **kwargs)
    ax.add_artist(e)
    e.set_zorder(1000)
    e.set_clip_box(ax.bbox)
    y_clone = y+L*(1 if y < C else -1)
    e = Ellipse(xy=[x, y_clone], width=w,
                height=h, angle=0, fc="None", **kwargs)
    ax.add_artist(e)
    e.set_zorder(1000)
    e.set_clip_box(ax.bbox)

    lw = e.get_linewidth()

    if crosshair:
        line_args = kwargs.copy()
        for k in ["lw", "linewidth"]:
            if k in line_args:
                del line_args[k]
        line_args["lw"] = lw/2
        if "path_effects" in line_args:
            del line_args["path_effects"]
        ax.plot([x + w/2, x - w/2],
                [y, y], **line_args)
        plot_vline_periodic(
            ax, x, y, h/2, **line_args)
