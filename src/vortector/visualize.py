import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

from .gaussfit import gauss

warnings.simplefilter("ignore")


def switch_axes_xy(ax):
    """ Switch the x and y axis of an axes. """
    lines = ax.get_lines()
    for line in lines:
        x = line.get_xdata()
        y = line.get_ydata()
        line.set_xdata(y)
        line.set_ydata(x)

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.set_xticks(yticks)
    ax.set_xticklabels(yticklabels)
    ax.set_yticks(xticks)
    ax.set_yticklabels(xticklabels)
    ax.set_xlim(ylim)
    ax.set_ylim(xlim)

    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    ax.set_xlabel(ylabel)
    ax.set_ylabel(xlabel)


def show_fit_overview_1D(vt, n, axes=None):
    if axes is None:
        fig, axes = plt.subplots(2, 2, dpi=150, figsize=(
            8, 6), constrained_layout=True, sharex="col", sharey="row")
        axes = axes.flatten()
    else:
        if len(axes) != 4:
            raise ValueError(
                "You need to pass a 1D array with 4 pyplot axes!")

    ax = axes[1]
    key = "vortensity"
    ref = "contour"
    show_radial_fit(vt, ax, key, n, ref=ref)
    ax.set_title(f"rad, ref={ref}")
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("")

    ax = axes[0]
    show_azimuthal_fit(vt, ax, key, n, ref=ref)
    ax.set_title(f"phi, ref={ref}")
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("")

    ax = axes[3]
    key = "surface_density"
    ref = "surface_density"
    center = "surface_density"
    show_radial_fit(vt, ax, key, n, ref=ref, center=center)

    ax = axes[2]
    show_azimuthal_fit(vt, ax, key, n, ref=ref, center=center)


def show_fit_overview_2D(vt, n=None, axes=None, bnd_lines=False, bnd_pnts=False, show_fits=True, fit_contours=True, xscale=None):
    if axes is None:
        fig, axes = plt.subplots(2, 5, figsize=(10, 6), dpi=150,
                                 #  sharex="col",
                                 gridspec_kw={"height_ratios": [1, 4],
                                              "width_ratios": [3, 1, 1, 3, 1]})
    else:
        if len(axes) != 8:
            raise ValueError(
                "You need to pass an array with 2 pyplot axes!")

    main_vortex = vt.guess_main_vortex()
    

    plt.subplots_adjust(hspace=.001, wspace=0.001)

    show_fit_overview_2D_single(vt, "vortensity", ax=axes[1, 0], n=n,
                                bnd_lines=bnd_lines, bnd_pnts=bnd_pnts,
                                show_fits=show_fits, fit_contours=fit_contours,
                                cbar_axes=[axes[1, 0], axes[1, 1]])

    yticklabels = axes[1, 0].get_yticklabels()

    show_fit_overview_2D_single(vt, "surface_density", ax=axes[1, 3], n=n,
                                bnd_lines=bnd_lines, bnd_pnts=bnd_pnts,
                                show_fits=show_fits, fit_contours=fit_contours,
                                cbar_axes=[axes[1, 3], axes[1, 4]])

    for ax in [axes[0, 1], axes[0, 4], axes[0, 2], axes[1, 2]]:
        ax.axis("off")

    if n is None:
        for k in range(len(vt.vortices)):
            if main_vortex == vt.vortices[k]:
                n = k
                break
    if n is None:
        n = 0

    if len(vt.vortices) > 0:
        ax = axes[0, 0]
        show_radial_fit(vt, ax, "vortensity", n, ref="contour")

        ax = axes[1, 1]
        show_azimuthal_fit(vt, ax, "vortensity", n, ref="contour")

        ax = axes[0, 3]
        show_radial_fit(vt, ax, "surface_density", n, ref="contour")

        ax = axes[1, 4]
        show_azimuthal_fit(vt, ax, "surface_density", n, ref="contour")
        switch_axes_xy(ax)

        sharex = [0, 3, 5, 8]
        sharey = [5, 6, 8, 9]
        axes_flat = axes.ravel()
        for n in range(1, 4):
            axes_flat[sharex[n]].sharex(axes_flat[sharex[n-1]])
            axes_flat[sharey[n]].sharey(axes_flat[sharey[n-1]])

    ax = axes[0, 0]
    ax.tick_params(labelbottom=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    if len(vt.vortices) > 0:
        ax.set_ylim(-0.5, 1)
        xticks = ax.get_yticks()
        vallim = ax.get_ylim()
        try:
            x = ax.get_lines()[0].get_data()[0]
            ax.set_xlim(np.min(x), np.max(x))
        except IndexError:
            pass

    ax = axes[1, 1]
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelleft=False)
    if len(vt.vortices) > 0:
        switch_axes_xy(ax)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks], rotation=270)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlim(vallim)

    ax = axes[0, 3]
    ax.tick_params(labelbottom=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    if len(vt.vortices) > 0:
        xticks = ax.get_yticks()
        ax.set_ylim(bottom=0)

    ax = axes[1, 4]
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelleft=False)
    if len(vt.vortices) > 0:
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1e}" for x in xticks],
                           rotation=270, fontsize=8)
        ax.set_xlim(left=0)
        ax.set_ylim(-np.pi, np.pi)
        
    if xscale is not None:
        axes[1,0].set_xscale(xscale)

    # for ax in [axes[1, 0], axes[1, 3]]:
    #     ax.set_yticklabels(yticklabels)


def show_fit_overview_2D_single(vt, varname, ax, n=None, bnd_lines=False,
                                bnd_pnts=True, show_fits=True, fit_contours=True,
                                cbar_axes=None):
    import matplotlib.patheffects as pe
    Xc = vt.radius
    Yc = vt.azimuth

    contour_colors = "darkgray"
    contour_lw = 0.5
    if varname == "vortensity":
        label = r"$\varpi/\varpi_0$"

        levels = vt.levels

        Z = vt.vortensity
        cmap = "magma"
        norm = matplotlib.colors.Normalize(vmin=levels[0], vmax=levels[-1])
        img = ax.pcolormesh(
            Xc, Yc, Z, cmap=cmap, norm=norm, rasterized=True, shading="auto")

        ax.contour(
            Xc, Yc, Z, levels=levels[::2], colors=contour_colors, linewidths=contour_lw)

    elif varname == "surface_density":
        label = r"$\Sigma$"

        Z = vt.surface_density
        cmap = "magma"

        # try:
        #     vmax = vt.vortices[0]["fits"]["surface_density"]["c"] + \
        #         vt.vortices[0]["fits"]["surface_density"]["a"]
        # except (KeyError, IndexError):
        #     vmax = np.max(Z)

        vmax = np.max(Z)
        vmin = 1e-3*vmax

        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        img = ax.pcolormesh(
            Xc, Yc, Z, cmap=cmap, norm=norm, rasterized=True, shading="auto")

        ax.contour(Xc, Yc, Z, levels=np.arange(
            0, vmax, vmax/20), colors=contour_colors, linewidths=contour_lw)
    else:
        raise ValueError(
            f"{varname} not supported. Only 'vortensity' and 'sigma'.")

    main_vortex = vt.guess_main_vortex()
    if n is None:
        if main_vortex is not None:
            vortices = [main_vortex] + [v for v in vt.vortices if v != main_vortex]
        else:
            vortices = []
    else:
        vortices = [vt.vortices[n]]

    for n, vort in enumerate(vortices):
        cnt = vort["contour"]
        try:
            ax.contour(Xc, Yc, cnt["mask"], levels=[
                0, 1, 2], linewidths=1, colors="white")
        except KeyError:
            print("Contour mask is not included in the output. Call 'detect_vortices' with 'include_mask=True' to show the outline of the contour.")
            pass
        x, y = cnt["stats"]["vortensity_min_pos"]
        if not show_fits:
            ax.plot([x], [y], "x")

        if bnd_lines:
            for key in ["rmin", "rmax"]:
                ax.axvline(cnt[key])
            for key in ["phimin", "phimax"]:
                ax.axhline(cnt[key])

        if bnd_pnts:
            for key in ["top", "bottom", "left", "right"]:
                x = Xc[cnt[key]]
                y = Yc[cnt[key]]
                ax.plot([x], [y], "x")

        blow = -np.pi
        bup = np.pi
        L = bup - blow
        if show_fits:
            try:
                lw = 1
                path_effects = [
                    pe.Stroke(linewidth=2*lw, foreground='w'), pe.Normal()]

                r0 = vort["fits"]["vortensity"]["r0"]
                sigma_r = vort["fits"]["vortensity"]["sigma_r"]
                w = 2*np.sqrt(2*np.log(2))*sigma_r
                phi0 = vort["fits"]["vortensity"]["phi0"]
                sigma_phi = vort["fits"]["vortensity"]["sigma_phi"]
                phi0 = (phi0 - blow) % L + blow
                h = 2*np.sqrt(2*np.log(2))*sigma_phi

                color_vortensity = "C0"
                lw = 1
                if n == 0:
                    plot_ellipse_periodic(
                        ax, r0, phi0, w, h, crosshair=True, color=color_vortensity, ls="-", lw=lw, path_effects=path_effects)
                else:
                    plot_ellipse_periodic(
                        ax, r0, phi0, w, h, color=color_vortensity, ls="-", lw=lw)

                r0 = vort["fits"]["surface_density"]["r0"]
                sigma_r = vort["fits"]["surface_density"]["sigma_r"]
                w = 2*np.sqrt(2*np.log(2))*sigma_r
                phi0 = vort["fits"]["surface_density"]["phi0"]
                phi0 = (phi0 - blow) % L + blow
                sigma_phi = vort["fits"]["surface_density"]["sigma_phi"]
                h = 2*np.sqrt(2*np.log(2))*sigma_phi

                if n == 0:
                    plot_ellipse_periodic(
                        ax, r0, phi0, w, h, color="C2", ls="-", lw=lw, crosshair=True, path_effects=path_effects)
                else:
                    plot_ellipse_periodic(
                        ax, r0, phi0, w, h, color="C2", ls="-", lw=lw)
            except KeyError:
                pass

    ax.set_xlabel(r"$r$ [au]")
    ax.set_ylabel(r"$\phi$")
    ax.set_yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi])
    ax.set_yticklabels(
        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    # ax.set_xlim(5.2, 10)

    ax.set_ylim(-np.pi, np.pi)

    if cbar_axes is None:
        cbar = ax.get_figure().colorbar(img, ax=ax, orientation="horizontal")
    else:
        cbar = ax.get_figure().colorbar(img, ax=cbar_axes, orientation="horizontal")
    cbar.set_label(label)


def select_fit_quantity(vt, key):
    """ Return values by name.

    Parameters
    ----------
    vt : vortector.Vortector
        Vortector object.
    key : str
        Name of the quantity.

    Returns
    -------
    array of floats
        Data array.
    """
    if key == "vortensity":
        return vt.vortensity
    elif key == "surface_density":
        return vt.surface_density
    else:
        raise AttributeError(
            f"'{key}' is not a valid choice for a fit quantity.")


def vortex_mask_r(vt, c, hw):
    """ Construct a radial vortex mask from center and width value.

    Parameters
    ----------
    vt : vortector.Vortector
        Vortector object.
    c : float
        Center position.
    hw : float
        Half of the width.

    Returns
    -------
    array of bool
        Mask indicating the vortex region.
    """
    wf = 1.5
    r = vt.radius[:, 0]
    mask = np.zeros(len(r), dtype=bool)
    cind = np.argmin(np.abs(r - c))
    lr = c - wf*hw
    lind = np.argmin(np.abs(r - lr))
    ur = c + wf*hw
    uind = np.argmin(np.abs(r - ur))
    mask = np.zeros(len(r), dtype=bool)
    mask[lind:uind+1] = True
    return mask


def show_radial_fit(vt, ax, key, n, ref="contour", center=None):
    """ Show a plot of a radial gaussian fit for the nth vortex.

    Parameters
    ----------
    vt : vortector.Vortector
        Vortector object.
    ax : plt.axes
        Axes to plot on.
    key : str
        Name of the variable.
    n : int
        Number of the vortex in the sorted vortex array.
    ref : str
        Reference for the vortex region (contour/vortensity/sigma).
    """
    try:
        vortex = vt.vortices[n]
    except IndexError:
        return

    try:
        center = ref if center is None else center
        inds = select_center_inds(vt, vortex, center)
        mask_r, mask_phi = select_fit_region(vt, vortex, ref)
        vals = select_fit_quantity(vt, key)

        mask = mask_r

        y = vals[:, inds[1]]
        x = vt.radius[:, 0]

        ax.plot(x, y, label=f"data slice n={n}")
        ax.plot(x[mask], y[mask], label="vortex region")

        y0 = vortex["fits"][key]["c"]
        x0 = vortex["fits"][key]["r0"]
        a = vortex["fits"][key]["a"]
        sig = vortex["fits"][key]["sigma_r"]
        popt = [y0, a, x0, sig]

        ax.plot(x[mask], gauss(x[mask], *popt),
                ls="--", color="C2", lw=2, label=f"fit")
        ax.plot(x, gauss(x, *popt), color="C3", alpha=0.3)
        ax.plot([x0], [y[inds[0]]], "x")
    except KeyError as e:
        print(f"Warning: KeyError encountered in showing r fit: {e}")
        return

    ax.set_xlabel(r"$r$")
    ax.set_ylabel(f"{key}")
    ax.legend()
    ax.grid()


def show_azimuthal_fit(vt, ax, key, n, ref="contour", center=None):
    """ Show a plot of a azimuthal gaussian fit for the nth vortex.

    Parameters
    ----------
    vt : vortector.Vortector
        Vortector object.
    ax : plt.axes
        Axes to plot on.
    key : str
        Name of the variable.
    n : int
        Number of the vortex in the sorted vortex array.
    ref : str
        Reference for the vortex region (contour/vortensity/sigma).
    """
    try:
        c = [c for c in vt.vortices][n]
    except IndexError:
        return

    try:
        center = ref if center is None else center
        inds = select_center_inds(vt, c, center)
        mask_r, mask_phi = select_fit_region(vt, c, ref)
        vals = select_fit_quantity(vt, key)
        mask = mask_phi

        y = vals[inds[0], :]
        x = vt.azimuth[0, :]

        ax.plot(x, y, label=f"data slice n={n}")
        plot_periodic(ax, x, y, mask, label="vortex region")
        y0 = c["fits"][key]["c"]
        x0 = c["fits"][key]["phi0"]
        a = c["fits"][key]["a"]
        sig = c["fits"][key]["sigma_phi"]

        bnd = vt.azimuthal_boundaries
        L = bnd[1] - bnd[0]

        xc, yc = combine_periodic(x, y, mask)

        if x0 < np.min(xc):
            x0 += L
        if x0 > np.max(xc):
            x0 -= L
        popt = [y0, a, x0, sig]

        plot_periodic(ax, xc, gauss(xc, *popt), bnd=bnd,
                      ls="--", lw=2, color="C2", label=f"fit")

        xfull = np.linspace(x0-L/2, x0+L/2, endpoint=True)
        plot_periodic(ax, xfull, gauss(xfull, *popt), bnd=bnd,
                      ls="-", lw=1, color="C3", alpha=0.3)
        ax.plot([clamp_periodic(x0, bnd)], y[[inds[1]]], "x")
    except KeyError as e:
        print(f"Warning: KeyError encountered in showing phi fit: {e}")
        return

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(f"{key}")
    ax.set_xticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.legend()
    ax.grid()


def vortex_mask_phi(vt, c, hw):
    """ Construct a azimuthal vortex mask from center and width value.

    This function takes the periodic boundary into account.

    Parameters
    ----------
    c : float
        Center position.
    hw : float
        Half of the width.

    Returns
    -------
    array of bool
        Mask indicating the vortex region.
    """
    wf = 1.5
    phi = vt.azimuth[0, :]
    bnd = vt.azimuthal_boundaries

    mask = np.zeros(len(phi), dtype=bool)
    cind = np.argmin(np.abs(phi - c))
    lphi = clamp_periodic(c - wf*hw, bnd)
    lind = np.argmin(np.abs(phi - lphi))
    uphi = clamp_periodic(c + wf*hw, bnd)
    uind = np.argmin(np.abs(phi - uphi))
    mask = np.zeros(len(phi), dtype=bool)
    if c + wf*hw > bnd[1] or c - wf*hw < bnd[0]:
        mask[lind:] = True
        mask[:uind+1] = True
    else:
        mask[lind:uind+1] = True
    return mask


def select_center_inds(vt, vortex, ref):
    """ Select the indices which indicate the fit region.

    The source which specifies the vortex region is either
    the extracted contour or the fits of vortensity or surface density.
    ref = contour, vortensity, sigma


    Parameters
    ----------
    vt : vortector.Vortector
        Vortector object.
    vortex : dict
        Vortex candidate.
    ref : str
        Name of the variable to take the mask from.

    Returns
    -------
    inds : tuple of two int
        Radial and azimuthal index of center.
    """
    if ref == "contour":
        r0, phi0 = vortex["contour"]["stats"]["vortensity_min_pos"]
    elif ref == "vortensity":
        r0 = vortex["fits"]["vortensity"]["r0"]
        phi0 = vortex["fits"]["vortensity"]["phi0"]
    elif ref == "surface_density":
        r0 = vortex["fits"]["surface_density"]["r0"]
        phi0 = vortex["fits"]["surface_density"]["phi0"]
    else:
        raise AttributeError(
            f"'{ref}' is not a valid reference for fitting.")
    rind = position_index(vt.radius[:, 0], r0)
    phiind = position_index(vt.azimuth[0, :], phi0)
    inds = (rind, phiind)
    return inds


def select_fit_region(vt, vortex, ref):
    """ Select the indices and masks which indicate the fit region.

    The source which specifies the vortex region is either
    the extracted contour or the fits of vortensity or surface density.
    ref = contour, vortensity, sigma


    Parameters
    ----------
    c : dict
        Vortex candidate.
    ref : str
        Name of the variable to take the mask from.

    Returns
    -------
    inds : tuple of two int
        Radial and azimuthal index of center.
    mask_r : array of bool
        Mask indicating vortex region in radial direction.
    mask_phi : array of bool
        Mask indicating vortex region in azimuthal direction.
    """
    inds = select_center_inds(vt, vortex, ref)
    if ref == "contour":
        try:
            mask = vortex["contour"]["mask"]
            mask_r = mask[:, inds[1]]
            mask_phi = mask[inds[0], :]
        except KeyError:
            mask_r, mask_phi = direction_masks_from_extent(
                vt.radius.shape, vortex["contour"])
    elif ref in ["vortensity", "surface_density"]:
        center = vortex["fits"][ref]["r0"]
        hw = vortex["fits"][ref]["sigma_r"]
        mask_r = vortex_mask_r(vt, center, hw)
        center = vortex["fits"][ref]["phi0"]
        hw = vortex = vortex["fits"][ref]["sigma_phi"]
        mask_phi = vortex_mask_phi(vt, center, hw)
    else:
        raise AttributeError(
            f"'{ref}' is not a valid reference for fitting.")
    return mask_r, mask_phi


def direction_masks_from_extent(shape, contour):
    left = contour["left"]
    right = contour["right"]
    top = contour["top"]
    bottom = contour["bottom"]

    mask_r = np.zeros(shape[0], dtype=bool)
    mask_r[left[0]:right[0]+1] = True

    mask_phi = np.zeros(shape[1], dtype=bool)
    if bottom[1] < top[1]:
        mask_phi[bottom[1]:top[1]+1] = True
    else:
        mask_phi[0:top[1]+1] = True
        mask_phi[bottom[1]:] = True

    return mask_r, mask_phi


def position_index(x, x0):
    """ Index of position x0 in array x .

    Parameters
    ----------
    x : array
        Position array.
    x0 : float
        Position of interest.

    Returns
    -------
    int
        Index of position x0.
    """
    return int(np.argmin(np.abs(x-x0)))


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


def clamp_periodic(x, azimuthal_boundaries):
    """ Make sure a periodic quantity is inside the domain.

    Parameters
    ----------
    x : float
        Position.

    Returns
    -------
    float
        Position moved into the boundary values.
    """
    bnd = azimuthal_boundaries
    rv = (x - bnd[0]) % (bnd[1]-bnd[0]) + bnd[0]
    return rv


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
