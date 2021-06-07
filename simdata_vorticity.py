import numpy as np
import astropy
import astropy.units as u
import simdata
import os
import pickle


cache_dir = "simulation_data_cache"

def provide_simulation_data(simid, Noutput, skip_cache=False, calc_kwargs=dict()):
    cache = DataCache(cache_dir, f"{simid}")
    if Noutput in cache.data and not skip_cache:
        rv = cache.data[Noutput]
    else:
        simulation = simdata.SData(simid)
        rv = calc_quantities(simulation, Noutput, **calc_kwargs)
        cache.data[Noutput] = rv
        cache.save()
    return rv

def calc_quantities(simulation, Noutput, normalize_by = None, return_Nroll=False):
    M_star = 1*u.solMass
    rho = simulation.fluids["gas"].get("2d", "mass density", Noutput)
    r_c = rho.grid.get_centers("r")
    phi_c = rho.grid.get_centers("phi")
    phi_c = map_angles(phi_c.to_value("rad"))*u.rad
    PHI_c, R_c = np.meshgrid(phi_c, r_c)

    dr = rho.grid.get_sizes("r").to_value("au")
    dphi = rho.grid.get_sizes("phi").to_value("rad")
    DPHI, DR = np.meshgrid(dphi, dr)

    Rho_background = simulation.fluids["gas"].get("2d", "mass density", 0).data.to_value("solMass/au2")
    Rho = simulation.fluids["gas"].get("2d", "mass density", Noutput).data.to_value("solMass/au2")
    
    vorticity = vorticity_simdata(simulation, Noutput)
    Omega_Kepler = np.sqrt(astropy.constants.G * M_star / R_c**3).decompose()
    vorticity_Kepler = (0.5*Omega_Kepler).to_value("1/yr")
    vorticity = vorticity.to_value("1/yr")

    if normalize_by is None:
        vortensity = vorticity/Rho
    elif normalize_by == "initial":
        vortensity = vorticity/Rho * Rho_background / vorticity_Kepler
        vorticity = vorticity / vorticity_Kepler
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

    R = R_c.to_value("au")
    Phi = PHI_c.to_value("rad")
    R = R
    A = DR*R*DPHI

    rv = roll_data(phi_c, R, Phi, A, vorticity, vorticity_Kepler, Rho, Rho_background, return_Nroll=return_Nroll)
    return rv


def roll_data(phi, *args, return_Nroll=False):
    sign_changes = phi[1:] - phi[:-1] < 0
    sign_change_pos = np.argmax(sign_changes)
    N_roll = len(sign_changes) - sign_change_pos
    if sign_change_pos == 0 and (~sign_changes).all():
        N_roll = 0
        rv = [x for x in args]
    else:
        rv = [np.roll(x, N_roll, axis=1) for x in args]
    if return_Nroll:
        rv += [N_roll]
    return rv


def map_angles(phi, interval = (-np.pi, np.pi)):
    """ Map angles to the range [phi_min, phi_min + 2pi]

    Parameters
    ----------
    phi: float
        Angles to map.
    phi_min: float
        Lower bound.
    """
    phimin = interval[0]
    phimax = interval[1]    
    return (phi - phimin) % (phimax - phimin) + phimin


def vorticity_simdata(data, Noutput, rref=None):
    vrad = data.fluids["gas"].get("2d", "velocity radial", Noutput)
    rs = vrad.grid.get_centers("r")
    phi = vrad.grid.get_centers("phi")

    PHI, R = np.meshgrid(phi, rs)

    t = vrad.get_time()
    try:
        omega_frame = data.planets[0].get(
            "omega frame").get_closest_to_time(t).to("s-1")
    except (KeyError, IndexError):
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
    except (KeyError, IndexError):
        Mstar = 1*u.solMass
        print("Warning: Assumed a star mass of 1 solar mass!")

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
    phi = map_angles(phi.to_value("rad"))

    PHI, R = np.meshgrid(phi, rs)

    x = R*np.cos(PHI)
    y = R*np.sin(PHI)

    # centered velocities
    if vrad.data.shape[0] != len(rs):
        vrad_c = vrad.data[:-1, :]
    else:
        vrad_c = vrad.data    #vazi_c = vazi.data - np.tile(np.mean(vazi.data, axis=1), (vazi.data.shape[1], 1) ).transpose()
    G = astropy.constants.G
    try:
        Mstar = data.planets[0].get("mass")[0]
    except (KeyError, IndexError):
        Mstar = 1*u.solMass
        print("Warning: Assumed a star mass of 1 solar mass!")
    try:
        omega_frame = data.planets[0].get("omega frame")[0]
    except (KeyError, IndexError):
        omega_frame = (1/data.loader.units["time"]).to("s-1")
    v_K = np.sqrt(G*Mstar/R).to("cm/s")
    v_Frame = omega_frame.to("s-1")*R
    v_K = np.sqrt(G*Mstar/R).to("cm/s")
    v_Frame = omega_frame*R
    vazi_c = vazi.data - v_K + v_Frame

    vx = vrad_c*np.cos(PHI) - vazi_c*np.sin(PHI)
    vy = vrad_c*np.sin(PHI) + vazi_c*np.cos(PHI)

    return roll_data(phi ,x, y, vx, vy)


def velocity_polar_simdata(data, Noutput):
    vrad = data.fluids["gas"].get("2d", "velocity radial", Noutput)
    vazi = data.fluids["gas"].get("2d", "velocity azimuthal", Noutput)

    rs = vrad.grid.get_centers("r")
    phi = vrad.grid.get_centers("phi")
    phi = map_angles(phi.to_value("rad"))

    PHI, R = np.meshgrid(phi, rs)

    # centered velocities
    if vrad.data.shape[0] != len(rs):
        vrad_c = vrad.data[:-1, :]
    else:
        vrad_c = vrad.data
    #vazi_c = vazi.data - np.tile(np.mean(vazi.data, axis=1), (vazi.data.shape[1], 1) ).transpose()
    G = astropy.constants.G
    try:
        Mstar = data.planets[0].get("mass")[0]
    except (KeyError, IndexError):
        Mstar = 1*u.solMass
        print("Warning: Assumed a star mass of 1 solar mass!")
    try:
        omega_frame = data.planets[0].get("omega frame")[0]
    except (KeyError, IndexError):
        omega_frame = (1/data.loader.units["time"]).to("s-1")
    v_K = np.sqrt(G*Mstar/R).to("cm/s")
    v_Frame = omega_frame.to("s-1")*R
    vazi_c = vazi.data + v_Frame

    return roll_data(phi, vrad_c, vazi_c, v_K)


class DataCache:

    def __init__(self, cache_dir, name, info=""):
        self.cache_dir = cache_dir
        self.name = name
        self.create_cache_dir()
        self.data_file = os.path.join(cache_dir, name + ".cache.pickle")
        self.load()

    def create_cache_dir(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def load(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, "rb") as in_file:
                self.data = pickle.load(in_file)
        else:
            self.data = {}

    def save(self):
        with open(self.data_file, "wb") as out_file:
            pickle.dump(self.data, out_file)

    def update_info(self, info):
        if info == "":
            return
        if "info" in self.data:
            if self.data["info"][-1] != info:
                self.data["info"].append(info)
        else:
            self.data["info"] = [info]
