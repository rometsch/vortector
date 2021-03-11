import numpy as np
import astropy
import astropy.units as u

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