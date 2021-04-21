import numpy as np


def calc_vortex_mass(c, mass):
    mask = c["mask"]
    c["stats"]["mass"] = np.sum(mass[mask])


def calc_vortensity(c, vortensity):
    mask = c["mask"]
    s = c["stats"]
    s["vortensity_avg"] = np.mean(vortensity[mask])
    s["vortensity_med"] = np.median(vortensity[mask])
    s["vortensity_min"] = np.min(vortensity[mask])
    s["vortensity_max"] = np.max(vortensity[mask])


def calc_sigma(c, surface_density):
    mask = c["mask"]
    s = c["stats"]
    s["surface_density_avg"] = np.mean(surface_density[mask])
    s["surface_density_med"] = np.median(surface_density[mask])
    s["surface_density_min"] = np.min(surface_density[mask])
    s["surface_density_max"] = np.max(surface_density[mask])


def calc_vortex_extent(c, area, r, phi):
    mask = c["mask"]
    s = c["stats"]
    s["area"] = np.sum(area[mask])
    s["rmax"] = r[c["left"]]
    s["rmin"] = r[c["right"]]
    s["phimin"] = phi[c["top"]]
    s["phimax"] = phi[c["bottom"]]
    if s["phimax"] < s["phimin"]:
        s["height"] = s["phimax"] + 2*np.pi - s["phimin"]
    else:
        s["height"] = s["phimax"] - s["phimin"]


def find_vortensity_min_position(contour, r, phi, vortensity):
    # Calculate the position of minimum vortensity
    mask = np.logical_not(contour["mask"])
    ind = np.argmin(np.ma.masked_array(
        vortensity, mask=mask), axis=None)
    inds = np.unravel_index(ind, mask.shape)
    x = r[inds]
    y = phi[inds]
    contour["stats"]["vortensity_min_pos"] = (x, y)
    contour["stats"]["vortensity_min_inds"] = inds


def find_density_max_position(contour, r, phi, surface_density):
    # Calculate the position of maximum density
    mask = np.logical_not(contour["mask"])
    ind = np.argmax(np.ma.masked_array(
        surface_density, mask=mask), axis=None)
    inds = np.unravel_index(ind, mask.shape)
    x = r[inds]
    y = phi[inds]
    contour["stats"]["surface_density_max_pos"] = (x, y)
    contour["stats"]["surface_density_max_inds"] = inds


def calc_azimuthal_statistics(stats, surface_density, vortensity):
    """ Calculate statistics along the azimuthal direction at the location
    of vortensity minimum and surface density maximum."""
    inds = stats["vortensity_min_inds"]
    stats["azimuthal_at_vortensity_min"] = {
        "vortensity_max": np.max(vortensity[inds[0], :]),
        "vortensity_avg": np.mean(vortensity[inds[0], :]),
        "vortensity_med": np.median(vortensity[inds[0], :])
    }

    inds = stats["surface_density_max_inds"]
    stats["azimuthal_at_surface_density_max"] = {
        "surface_density_min": np.max(surface_density[inds[0], :]),
        "surface_density_avg": np.mean(surface_density[inds[0], :]),
        "surface_density_med": np.median(surface_density[inds[0], :])
    }


def calc_orbital_elements_vortector(vt, n, vrad, vphi, mu, region="contour"):
    """ Calculate mass-averaged semi-major axis and eccentricity for a vortex
    of a Vortector object.

    The selected region can be the contour or the ellipse region (FWHM) of either
    the vortensity or surface density fit.

    Parameters
    ----------
    vt : vortector.Vortector
        Vortector instance used as basis.
    n : int
        Number of the vortex inside the Vortector vortices list.
    vrad : np.array 2D 
        Radial velocity values as 2D array matching the shape of the data in vt.
    vphi : np.array 2D 
        Azimuth values as 2D array matching the shape of the data in vt.
    mu : float
        Gravitational constant * stellar mass in the used unit system.
    region : str
        Region to analyze, either 'contour', 'vortensity', 'surface_density', 
        by default 'contour'

    Returns
    -------
    dict
        Dictionary containing the semi-major axis ("a") and eccentricity ("e").
    """
    vortex = vt.vortices[n]
    if region == "contour":
        mask = vortex["contour"]["mask"]
    elif region in ["vortensity", "surface_density"]:
        r0 = vortex["fits"][region]["r0"]
        hr = np.sqrt(2*np.log(2))*vortex["fits"][region]["sigma_r"]
        phi0 = vortex["fits"][region]["phi0"]
        hphi = np.sqrt(2*np.log(2))*vortex["fits"][region]["sigma_phi"]
        mask = ((vt.radius - r0)/hr)**2 + ((vt.azimuth - phi0)/hphi)**2 <= 1
    else:
        raise ValueError(
            f"region = '{region}' is not supported. Choose from 'contour'/'vortensity'/'surface_density'")

    return calc_orbital_elements(vt.radius, vt.azimuth, vt.area, vt.surface_density, vrad, vphi, mu, mask=mask)


def calc_orbital_elements(r, phi, area, sigma, vrad, vphi, mu, mask=None):
    """ Calculate mass-averaged semi-major axis and eccentricity for 2D data.

    A subregion of the domain can be selected by specifying a boolean mask.

    Parameters
    ----------
    r : np.array 2D 
        Radius values as 2D array.
    phi : np.array 2D 
        Azimuth values as 2D array in radians.
    area : np.array 2D 
        Cell area values as 2D array.
    sigma : np.array 2D 
        Surface density values as 2D array.
    vrad : np.array 2D 
        Radial velocity values as 2D array.
    vphi : np.array 2D 
        Azimuth values as 2D array.
    mu : float
        Gravitational constant * stellar mass in the used unit system.
    mask : np.array of type bool 2D, optional
        2D array of boolean values indicating a subregion, by default None

    Returns
    -------
    dict
        Dictionary containing the semi-major axis ("a") and eccentricity ("e").
    """
    # mass of each cell
    mass = sigma * area

    # velocities in cartesian for each cell
    vx = vrad*np.cos(phi) - vphi*np.sin(phi)
    vy = vrad*np.sin(phi) + vphi*np.cos(phi)

    # cartesian coordinates
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # specific angular momentum and speed squared
    h2 = (x*vy - y*vx)**2
    v2 = vx*vx + vy*vy

    # smj axis from energy
    eng = 0.5*v2 - mu/r
    a = -0.5*mu/eng

    # eccentricity
    ecc = np.sqrt(1.0-h2/(mu*a))

    # weight by mass, calculate weighted eccentricity of each cell
    if mask is None:
        mask = np.ones(mass.shape, dtype=bool)

    mass = mass[mask]
    a = a[mask]
    ecc = ecc[mask]

    total_mass = np.sum(mass)

    weighted_a = np.sum(a*mass)/total_mass
    weighted_ecc = np.sum(ecc*mass)/total_mass

    return {"a": weighted_a, "e": weighted_ecc}
