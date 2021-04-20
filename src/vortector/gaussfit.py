import numpy as np
from scipy.optimize import curve_fit


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
    ex = np.exp(-(x - x0)**2 / (2 * sx**2))
    ey = np.exp(-(y - y0)**2 / (2 * sy**2))
    return c + a*ex*ey


class Gauss2DFitter:
    def __init__(self, x, y, z, weights=None, autoweight=False, blow=None, bup=None, fixed=None, p0=None):
        blow = {} if blow is None else blow
        bup = {} if bup is None else bup
        fixed = {} if fixed is None else fixed
        p0 = {} if p0 is None else p0

        self.x = x
        self.y = y
        self.z = z
        self.autoweight = autoweight
        self.weights = weights

        self.parameters = ["c", "a", "x0", "y0", "sx", "sy"]

        self.blow = {key: -np.inf for key in self.parameters}
        self.bup = {key: np.inf for key in self.parameters}
        self.blow["sx"] = 0
        self.bup["sx"] = np.max(x)-np.min(x)
        self.blow["x0"] = np.min(x)
        self.bup["x0"] = np.max(x)

        self.blow["sy"] = 0
        self.bup["sy"] = min((np.max(y)-np.min(y)), np.pi/2)
        self.blow["y0"] = np.min(y)
        self.bup["y0"] = np.max(y)

        for key, val in blow.items():
            self.set_lower_bound(key, val)
        for key, val in bup.items():
            self.set_upper_bound(key, val)

        self.fixed = fixed
        self.p0 = p0
        self.guess_p0()

    def set_lower_bound(self, key, value):
        if not key in self.parameters:
            raise KeyError(f"{key} is not a member of the lower bounds dict.")
        self.blow[key] = value

    def set_upper_bound(self, key, value):
        if not key in self.parameters:
            raise KeyError(f"{key} is not a member of the upper bounds dict.")
        self.bup[key] = value

    def set_fixed(self, key, value):
        if not key in self.parameters:
            raise KeyError(f"{key} is not a valid parameter.")
        self.fixed[key] = value

    def guess_p0(self):
        x = self.x
        y = self.y
        x0_guess = 0.5*(np.max(x) + np.min(x))
        sx_guess = 0.5*(np.max(x) - np.min(x))

        y0_guess = 0.5*(np.max(y) + np.min(y))
        sy_guess = 0.5*(np.max(y) - np.min(y))

        zavg = np.average(self.z)
        c_guess = zavg
        a_guess = zavg
        p0 = {
            "c": c_guess,
            "a": a_guess,
            "x0": x0_guess,
            "y0": y0_guess,
            "sx": sx_guess,
            "sy": sy_guess
        }
        for key, val in self.p0.items():
            p0[key] = val
        self.p0 = p0

    def fit(self):
        popt, pcov = self.fit_single()

        if self.weights is None and self.autoweight:
            peak_value = popt[0] + popt[1]  # y0 + a
            self.calc_weights(peak_value)
            popt, pcov = self.fit_single()
        return popt, pcov

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
        x = self.x
        y = self.y
        z = self.z
        fixed = self.fixed
        weights = self.weights
        lower = [self.blow[key] for key in self.parameters]
        upper = [self.bup[key] for key in self.parameters]
        p0 = [self.p0[key] for key in self.parameters]

        f = gauss2D

        bounds = (lower, upper)
        popt, pcov = curve_fit(f, (x, y), z, p0=p0,
                               bounds=bounds, sigma=weights)

        return popt, pcov