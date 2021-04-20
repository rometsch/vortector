import numpy as np


def calc_vortex_mass(c, mass):
    mask = c["mask"]
    c["mass"] = np.sum(mass[mask])


def calc_vortensity(c, vortensity):
    mask = c["mask"]
    c["vortensity_mean"] = np.mean(vortensity[mask])
    c["vortensity_median"] = np.median(vortensity[mask])
    c["vortensity_min"] = np.min(vortensity[mask])
    c["vortensity_max"] = np.max(vortensity[mask])


def calc_sigma(c, surface_density):
    mask = c["mask"]
    c["sigma_mean"] = np.mean(surface_density[mask])
    c["sigma_median"] = np.median(surface_density[mask])
    c["sigma_min"] = np.min(surface_density[mask])
    c["sigma_max"] = np.max(surface_density[mask])


def calc_vortex_extent(c, area, r, phi):
    mask = c["mask"]
    c["area"] = np.sum(area[mask])
    c["rmax"] = r[c["left"]]
    c["rmin"] = r[c["right"]]
    c["phimin"] = phi[c["top"]]
    c["phimax"] = phi[c["bottom"]]
    if c["phimax"] < c["phimin"]:
        c["height"] = c["phimax"] + 2*np.pi - c["phimin"]
    else:
        c["height"] = c["phimax"] - c["phimin"]


def find_vortensity_min_position(contour, r, phi, vortensity):
    # Calculate the position of minimum vortensity
    mask = np.logical_not(contour["mask"])
    ind = np.argmin(np.ma.masked_array(
        vortensity, mask=mask), axis=None)
    inds = np.unravel_index(ind, mask.shape)
    x = r[inds]
    y = phi[inds]
    contour["vortensity_min_pos"] = (x, y)
    contour["vortensity_min_inds"] = inds


def find_density_max_position(contour, r, phi, surface_density):
    # Calculate the position of maximum density
    mask = np.logical_not(contour["mask"])
    ind = np.argmax(np.ma.masked_array(
        surface_density, mask=mask), axis=None)
    inds = np.unravel_index(ind, mask.shape)
    x = r[inds]
    y = phi[inds]
    contour["surface_density_max_pos"] = (x, y)
    contour["surface_density_max_inds"] = inds


def orbital_elements(r, phi, area, sigma, vrad, vphi, mu, mask=None):
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
    #mass of each cell
    mass = sigma * area

    #velocities in cartesian for each cell
    vx = vrad*np.cos(phi) - vphi*np.sin(phi)
    vy = vrad*np.sin(phi) + vphi*np.cos(phi)
    
    #cartesian coordinates
    x  = r * np.cos(phi)
    y  = r * np.sin(phi)

    #specific angular momentum and speed squared
    h2  = (x*vy - y*vx)**2
    v2  = vx*vx + vy*vy

    #smj axis from energy
    eng = 0.5*v2 - mu/r
    a   = -0.5*mu/eng
    
    #eccentricity
    ecc = np.sqrt(1.0-h2/(mu*a))
    
    #weight by mass, calculate weighted eccentricity of each cell
    if mask is None:
        mask = np.ones(mass.shape, dtype=bool)
        
    mass = mass[mask]
    a = a[mask]
    ecc = ecc[mask]
    
    total_mass = np.sum(mass)
    
    weighted_a = np.sum(a*mass)/total_mass
    weighted_ecc = np.sum(ecc*mass)/total_mass

    return {"a" : weighted_a, "e" : weighted_ecc}