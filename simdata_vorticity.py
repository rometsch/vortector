import numpy as np
import astropy
import astropy.units as u

def calc_quantities(simulation, Noutput, normalize_by = "initial"):
    M_star = 1*u.solMass
    rho = simulation.fluids["gas"].get("2d", "mass density", Noutput)
    r = rho.grid.get_centers("r")
    phi = rho.grid.get_interfaces("phi")
    phi = map_angles(phi.to_value("rad"))
    if np.isclose(phi[-1],-np.pi):
        phi[-1] = np.pi
    phi = phi*u.rad
    PHI_rho, R_rho = np.meshgrid(phi, r)
    x_rho = R_rho*np.cos(PHI_rho)
    y_rho = R_rho*np.sin(PHI_rho)
    phi_c = rho.grid.get_centers("phi")
    phi_c = map_angles((phi_c).to_value("rad"))
    if phi_c[-1] == 0.0:
        phi_c[-1] = 2*np.pi
    phi_c = phi_c*u.rad
    PHI_c, R_c = np.meshgrid(phi_c, r)
    xc = R_c*np.cos(PHI_c)
    yc = R_c*np.sin(PHI_c)

    dr = rho.grid.get_sizes("r").to_value("au")
    dphi = rho.grid.get_sizes("phi").to_value("rad")
    DPHI, DR = np.meshgrid(dphi, dr)

    Rho_background = simulation.fluids["gas"].get("2d", "mass density", 0).data.to_value("solMass/au2")
    Rho = simulation.fluids["gas"].get("2d", "mass density", Noutput).data.to_value("solMass/au2")
    
    vorticity = vorticity_simdata(simulation, Noutput)
    Omega_Kepler = np.sqrt(astropy.constants.G * M_star / R_c**3).decompose()
    vorticity_Kepler = (0.5*Omega_Kepler).to_value("1/s")
    vorticity = vorticity.to_value(1/u.s)

    if normalize_by == "initial":
        vortensity = vorticity/Rho * Rho_background / vorticity_Kepler
    elif normalize_by == "median":
        vortensity = vorticity/Rho
        vortensity_med = np.median(vortensity, axis=1)
        vortensity = vortensity/np.tile(vortensity_med,(vortensity.shape[1], 1)).T
    elif normalize_by == "avg":
        vortensity = vorticity/Rho
        vortensity_avg = np.average(vortensity, axis=1)
        vortensity = vortensity/np.tile(vortensity_avg,(vortensity.shape[1], 1)).T
    else:
        raise ValueError(f"Invalid choice for vortensity normalization: '{normalize_by}'")
    
#     vortensity[vortensity < -1] = -1

    X = R_rho.to_value("au")
    Y = PHI_rho.to_value("rad")
    Xc = R_c.to_value("au")
    Yc = PHI_c.to_value("rad")
    R = Xc
    A = DR*R*DPHI

    N_roll = -np.argmax(phi[1:] - phi[:-1] < 0)
    X = np.roll(X, N_roll, axis=1)
    Y = np.roll(Y, N_roll, axis=1)
    Xc = np.roll(Xc, N_roll, axis=1)
    Yc = np.roll(Yc, N_roll, axis=1)
    vorticity = np.roll(vorticity, N_roll, axis=1)
    vortensity = np.roll(vortensity, N_roll, axis=1)
    Rho = np.roll(Rho, N_roll, axis=1)
    Rho_background = np.roll(Rho_background, N_roll, axis=1)
    return X, Y, Xc, Yc, A, vortensity, vorticity, Rho, Rho_background

def map_angles(phi, phi_min=-np.pi):
    """ Map angles to the range [phi_min, phi_min + 2pi]

    Parameters
    ----------
    phi: float
        Angles to map.
    phi_min: float
        Lower bound.
    """
    phi_max = phi_min + 2*np.pi
    phi = phi % (2*np.pi)
    if isinstance(phi, np.ndarray):
        phi[phi > phi_max] -= 2*np.pi
    else:
        if phi > phi_max:
            phi -= 2*np.pi
    return phi


def vorticity_simdata(data, Noutput, rref=None):
    vrad = data.fluids["gas"].get("2d", "velocity radial", Noutput)
    rs = vrad.grid.get_centers("r")
    phi = vrad.grid.get_centers("phi")

    PHI, R = np.meshgrid(phi, rs)

    t = vrad.get_time()
    try:
        omega_frame = data.planets[0].get(
            "omega frame").get_closest_to_time(t).to("s-1")
    except KeyError:
        omega_frame = (1/data.loader.units["time"]).to("s-1")

    dvrad_dphi = derivative_vrad_phi(data, Noutput)
    duazi_dr, uazi = derivative_vazi_r(data, Noutput, rref=rref)

    vorticity = 2*omega_frame + uazi/R + duazi_dr - dvrad_dphi/R
    return vorticity


def derivative_vrad_phi(data, Noutput):
    vrad = data.fluids["gas"].get("2d", "velocity radial", Noutput)

    rsi = vrad.grid.get_coordinates("r")
    phii = vrad.grid.get_coordinates("phi")

    PHIi, Ri = np.meshgrid(phii, rsi)

    rsc = vrad.grid.get_centers("r")
    phic = vrad.grid.get_centers("phi")
    PHIc, Rc = np.meshgrid(phic, rsc)

    if data.loader.code_info[0].lower().startswith("fargo"):
        Rplus = Ri[1:, :]
        Rminus = Ri[:-1, :]
        vplus = vrad.data[1:, :]
        vminus = vrad.data[:-1, :]
        # interpolate to cell centers
        vr = vminus + (vplus - vminus)/(Rplus - Rminus)*(Rc-Rminus)
    elif data.loader.code_info[0].lower().startswith("pluto"):
        vr = vrad.data
    dvrad_dphi = np.gradient(vr.to_value('cm/s'), phic.to_value('rad'), axis=1)
    return dvrad_dphi*u.cm/u.s


def derivative_vazi_r(data, Noutput, rref=None):
    vazi = data.fluids["gas"].get("2d", "velocity azimuthal", Noutput)

    try:
        Mstar = data.planets[0].get("mass")[0]
    except KeyError:
        Mstar = 1*u.solMass
        print("Warning: Assumed a start mass of 1 solar mass!")

    if rref is not None:
        rref = rref*u.au
        v_K = np.sqrt(astropy.constants.G*Mstar/rref).to("cm/s")
        omega_frame = data.planets[0].get(
            "omega frame").get_closest_to_time(vazi.get_time()).to("s-1")
        v_Frame = omega_frame*rref
        v_azi_rot = vazi.data + v_Frame - v_K
    else:
        v_azi_rot = vazi.data

    rsi = vazi.grid.get_centers("r")
    phii = vazi.grid.get_interfaces("phi")

    PHIi, Ri = np.meshgrid(phii, rsi)
    rsc = vazi.grid.get_centers("r")
    phic = vazi.grid.get_centers("phi")
    PHIc, Rc = np.meshgrid(phic, rsc)

    PHIplus = PHIi[:, 1:]
    PHIminus = PHIi[:, :-1]
    vplus = np.roll(v_azi_rot, 1, axis=1)
    vminus = v_azi_rot

    va = vminus + (vplus - vminus)/(PHIplus - PHIminus)*(PHIc-PHIminus)

    dvazi_dr = np.gradient(va.to_value('cm/s'), rsc.to_value('cm'), axis=0)
    return dvazi_dr/u.s, va


def velocity_cartesian_simdata(data, Noutput):
    vrad = data.fluids["gas"].get("2d", "velocity radial", Noutput)
    vazi = data.fluids["gas"].get("2d", "velocity azimuthal", Noutput)

    rs = vrad.grid.get_centers("r")
    phi = vrad.grid.get_centers("phi")

    PHI, R = np.meshgrid(phi, rs)

    x = R*np.cos(PHI)
    y = R*np.sin(PHI)

    # centered velocities
    vrad_c = vrad.data[:-1, :]
    #vazi_c = vazi.data - np.tile(np.mean(vazi.data, axis=1), (vazi.data.shape[1], 1) ).transpose()
    G = astropy.constants.G
    try:
        Mstar = data.planets[0].get("mass")[0]
    except KeyError:
        Mstar = 1*u.solMass
        print("Warning: Assumed a start mass of 1 solar mass!")
    v_K = np.sqrt(G*Mstar/R).to("cm/s")
    v_Frame = 0.5*1.6792048484108891e-08*1/u.s*R
    vazi_c = vazi.data - v_K + v_Frame

    vx = vrad_c*np.cos(PHI) - vazi_c*np.sin(PHI)
    vy = vrad_c*np.sin(PHI) + vazi_c*np.cos(PHI)

    return (x, y, vx, vy)


