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
