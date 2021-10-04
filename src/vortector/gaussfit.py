import numpy as np
from scipy.optimize import curve_fit
from numba import njit

def gauss(x, c, a, x0, sigma):
    """ A gaussian bell function.

    Parameters
    ----------
    x: array
        Coordinates
    c: float
        Offset
    a: float
        Amplitute
    x0: float
        Center of the bell curve.
    sigma : float
        Standard deviation.
    Returns
    -------
    array
        Function values.  
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + c


@njit(cache=True)
def gauss2D(v, c, a, x0, y0, sx, sy):
    """ A 2D version of the gaussian bell function.

    Parameters
    ----------
    v: tuple of two array
        x and y coordinates
    c: float
        Offset
    a: float
        Amplitute
    x0: float
        Center in x.
    y0: float
        Center in y.
    sx : float
        Standard deviation in x.
    sy : float
        Standard deviation in y.
    Returns
    -------
    array
        Function values.  
    """
    x, y = v
    argx = -(x - x0)**2 / (2 * sx**2)
    argy = -(y - y0)**2 / (2 * sy**2)
    return c + a*np.exp(argx+argy)


@njit(cache=True)
def gauss2D_jac(v, c, a, x0, y0, sx, sy):
    """ Jacobian of the 2D version of the gaussian bell function.

    Parameters
    ----------
    v: tuple of two array
        x and y coordinates
    c: float
        Offset
    a: float
        Amplitute
    x0: float
        Center in x.
    y0: float
        Center in y.
    sx : float
        Standard deviation in x.
    sy : float
        Standard deviation in y.
    Returns
    -------
    array
        Function values.  
    """
    x, y = v
    dx = x-x0
    dy = y-y0
    argx = -dx**2 / (2 * sx**2)
    argy = -dy**2 / (2 * sy**2)
    e = np.exp(argx+argy)
    df_dc = np.ones_like(x)
    df_da = e
    df_dx0 = a*dx*e/sx**2
    df_dy0 = a*dy*e/sy**2
    df_dsx = a*dx**2*e/sx**3
    df_dsy = a*dy**2*e/sy**3
    rv = np.empty((6,len(x)))
    rv[0] = df_dc
    rv[1] = df_da
    rv[2] = df_dx0
    rv[3] = df_dy0
    rv[4] = df_dsx
    rv[5] = df_dsy
    return rv.T


def extract_fit_values(contour, r, phi, vals, reference_point, periodicity=None):
    mask = contour["mask"]
    inds = contour["stats"][reference_point]
    r0 = r[inds]
    phi0 = phi[inds]

    mask_phi = mask[inds[0], :]

    R = r[mask]
    PHI = phi[mask]
    Z = vals[mask]

    if periodicity is not None:
        up = periodicity["upper"]
        L = periodicity["L"]
        # capture contours streching over the periodic boundary
        phi_low = phi[contour["pnt_ylow"]]
        phi_high = phi[contour["pnt_yhigh"]]
        if phi_high < phi_low:
            # always place the contour at the higher boundary
            PHI[PHI <= phi_high] += 2*np.pi
            if phi0 < phi_high:
                phi0 += L

    c_ref = np.median(vals[inds[0], np.logical_not(mask_phi)])

    return R, PHI, Z, r0, phi0, c_ref


def fit_2D_gaussian_surface_density(r, phi, Z, c_ref, periodicity=None):

    c_guess = c_ref
    a_guess = np.max(Z) - c_guess

    n = np.argmax(Z)
    r0 = r[n]
    phi0 = phi[n]

    dr = np.max(r) - np.min(r)
    dphi = np.max(phi) - np.min(phi)
    fitter = Gauss2DFitter(r, phi, Z,
                           p0={"x0": r0, "y0": phi0,
                               "c": 0.8*c_ref, "a": a_guess},
                           blow={"c": 0.75*c_ref, "a": 0.5*a_guess,
                                 "x0": r0-0.25*dr, "y0": phi0-0.25*dphi},
                           bup={"c": 1.25*c_ref, "a": 2*a_guess,
                                "x0": r0+0.25*dr, "y0": phi0+0.25*dphi},
                           )
    p, p0, cov = fitter.fit()

    if periodicity is not None:
        up = periodicity["upper"]
        L = periodicity["L"]
        if p[3] > up:
            p[3] -= L

    fit = {"c": p[0], "a": p[1], "r0": p[2], "phi0": p[3],
           "sigma_r": p[4], "sigma_phi": p[5], "popt": p, "pcov": cov,
           "bounds_low": fitter.blow, "bounds_high": fitter.bup, "p0" : p0}
    return fit


def fit_2D_gaussian_vortensity(r, phi, Z, c_ref, periodicity=None):

    n = np.argmin(Z)
    r0 = r[n]
    phi0 = phi[n]

    c_guess = c_ref
    a_guess = np.min(Z) - c_guess

    fitter = Gauss2DFitter(r, phi, Z,
                           p0={"x0": r0, "y0": phi0,
                               "c": c_ref, "a": a_guess},
                           blow={"c": 0.75*c_ref, "a": 1.25*a_guess},
                           bup={"c": 1.25*c_ref, "a": 0.75*a_guess})
    p, p0, cov = fitter.fit()

    if periodicity is not None:
        up = periodicity["upper"]
        L = periodicity["L"]
        if p[3] > up:
            p[3] -= L

    fit = {"c": p[0], "a": p[1], "r0": p[2], "phi0": p[3],
           "sigma_r": p[4], "sigma_phi": p[5], "popt": p, "pcov": cov,
           "bounds_low": fitter.blow, "bounds_high": fitter.bup, "p0" : p0}
    return fit


class Gauss2DFitter:
    def __init__(self, x, y, z, weights=None, autoweight=False, blow=None, bup=None, p0=None):
        blow = {} if blow is None else blow
        bup = {} if bup is None else bup
        p0 = {} if p0 is None else p0

        self.x = x
        self.y = y
        self.z = z
        self.autoweight = autoweight
        self.weights = weights

        self.parameters = ["c", "a", "x0", "y0", "sx", "sy"]

        self.blow = {key: -np.inf for key in self.parameters}
        self.bup = {key: np.inf for key in self.parameters}
        self.bup["sx"] = np.max(x)-np.min(x)
        self.blow["sx"] = self.bup["sx"]/20
        self.blow["x0"] = np.min(x)
        self.bup["x0"] = np.max(x)

        self.bup["sy"] = min((np.max(y)-np.min(y)), np.pi/2)
        self.blow["sy"] = self.bup["sy"]/20
        self.blow["y0"] = np.min(y)
        self.bup["y0"] = np.max(y)

        for key, val in blow.items():
            self.set_lower_bound(key, val)
        for key, val in bup.items():
            self.set_upper_bound(key, val)

        self.p0 = p0

    def set_lower_bound(self, key, value):
        if not key in self.parameters:
            raise KeyError(f"{key} is not a member of the lower bounds dict.")
        self.blow[key] = value

    def set_upper_bound(self, key, value):
        if not key in self.parameters:
            raise KeyError(f"{key} is not a member of the upper bounds dict.")
        self.bup[key] = value

    def guess_p0(self):
        x = self.x
        y = self.y
        z = self.z
        v = np.array([x,y])
        sx = 0.5*(np.max(x) - np.min(x))
        sy = 0.5*(np.max(y) - np.min(y))

        v0 = np.sum(v* np.abs(z-np.median(z)), axis=1)/len(x)
        x0, y0 = v0

        zavg = np.average(self.z)
        c = zavg
        a = zavg
        p0 = {
            "c": c,
            "a": a,
            "x0": x0,
            "y0": y0,
            "sx": sx,
            "sy": sy
        }

        # apply manually passed guesses
        for key, val in self.p0.items():
            p0[key] = val

        # restrict guess to allowed range of values
        for key, val in p0.items():
            val = min(val, self.bup[key])
            val = max(val, self.blow[key])
            p0[key] = val

        self.p0 = p0

    def fit(self):
        self.guess_p0()
        popt, p0, pcov = self.fit_single()

        if self.weights is None and self.autoweight:
            peak_value = popt[0] + popt[1]  # y0 + a
            self.calc_weights(peak_value)
            popt, p0, pcov = self.fit_single()
        return popt, p0, pcov

    def calc_weights(self, peak_value):
        difference = np.abs(self.y - peak_value)

        x0 = self.p0["x0"]
        y0 = self.p0["y0"]
        sx = self.p0["sx"]
        sy = self.p0["sy"]
        dx = np.abs(x0 - self.x)
        dy = np.abs(y0 - self.y)
        self.weights = np.exp(-difference/np.max(difference)) * \
            np.exp(-(dx/sx)**2 - (dy/sy)**2)

    def fit_single(self):
        x = np.array(self.x, dtype=np.float64)
        y = np.array(self.y, dtype=np.float64)
        z = np.array(self.z, dtype=np.float64)
        weights = np.array(self.weights, dtype=np.float64)
        lower = np.array([self.blow[key] for key in self.parameters], dtype=np.float64)
        upper = np.array([self.bup[key] for key in self.parameters], dtype=np.float64)
        p0 = np.array([self.p0[key] for key in self.parameters], dtype=np.float64)

        f = gauss2D

        bounds = (lower, upper)
        popt, pcov = curve_fit(
            f, (x, y), z, p0=p0, bounds=bounds, #sigma=weights, 
            jac=gauss2D_jac, method="trf")

        return popt, p0, pcov
