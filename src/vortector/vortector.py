import numpy as np

from . import analyze
from .gaussfit import gauss2D, Gauss2DFitter
from .contour import detect_elliptic_contours


class Vortector:
    def __init__(self, radius, azimuth, area, vortensity, surface_density,
                 levels=[float(x) for x in np.arange(-1, 1.5, 0.05)],
                 med=0.15, mear=np.inf, mvd=0.01, verbose=False, azimuthal_boundaries=[-np.pi, np.pi]):

        self.vortensity = vortensity

        self.surface_density = surface_density

        self.radius = radius
        self.azimuth = azimuth
        self.area = area

        self.azimuthal_boundaries = azimuthal_boundaries

        # Parameters
        self.levels = levels
        self.med = med  # max_ellipse_deviation
        self.mear = mear  # max_ellipse_aspect_ratio
        self.mvd = mvd  # min_vortensity_drop

        self.verbose = verbose

        self.mass = self.area*self.surface_density

    def detect_vortices(self, include_mask=False, keep_internals=False):

        self.candidates = detect_elliptic_contours(self.vortensity,
                                                   self.levels,
                                                   self.mear,
                                                   self.med,
                                                   self.verbose)

        self.calculate_contour_properties()

        self.vortices = []
        for c in self.candidates.values():
            self.vortices.append({"contour": c})

        self.fit_contours()
        self.remove_non_vortex_candidates()
        self.remove_duplicates_by_min_vort_pos()

        self.sort_vortices_by_mass()

        self.remove_intermediate_data(include_mask, keep_internals)

        return self.vortices

    def sort_vortices_by_mass(self):
        """ Sort the vortices by mass decending. """
        self.vortices = [v for v in reversed(
            sorted(self.vortices, key=lambda v: v["contour"]["mass"]))]

    def remove_intermediate_data(self, include_mask=False, keep_internals=False):
        for v in self.vortices:
            c = v["contour"]
            if not keep_internals:
                for key in ["contour", "mask_extended", "bounding_hor",
                            "bounding_vert", "pixel_arcLength", "pixel_area",
                            "top_extended", "left_extended", "bottom_extended", "right_extended",
                            "ancestors", "decendents"]:
                    del c[key]
            if not include_mask:
                del c["mask"]

    def remove_non_vortex_candidates(self):
        """ Remove candidates not having vortex properties
        A vortex should have at least a small dip in vortensity.
        Exclude vortices for which the minimum vorticity is not at least 0.05 lower than the maximum vorticity.
        Also check that vortensity < 1
        """

        no_min = []
        for n, vortex in enumerate(self.vortices):
            c = vortex["contour"]
            cid = c["opencv_contour_number"]
            try:
                if c["vortensity_min"] > 1:
                    if self.verbose:
                        print(
                            f"Check candidates: excluding {cid} because of min_vortensity > 1")
                    no_min.append(n)
                    continue
                vortensity_drop = c["vortensity_max"] - c["vortensity_min"]
                if vortensity_drop < self.mvd:
                    if self.verbose:
                        print(
                            f"Check candidates: excluding {cid} vortensity drop is {vortensity_drop} < {self.mvd}")
                    no_min.append(n)
            except KeyError:
                # remove candiates causing errors
                no_min.append(n)
        for k, n in enumerate(no_min):
            del self.vortices[n-k]

    def remove_duplicates_by_min_vort_pos(self):
        """ Remove remaining duplicates.

        Because of the way the mirror counter lines are generated,
        the mirror shape can be shifted slightly.
        Still both mirror shapes should share the same location of minimum.
        The shape containing the lower mass is deleted.
        """
        to_del = []
        for v in self.vortices:
            c = v["contour"]
            inds = c["vortensity_min_inds"]
            mass = c["mass"]

            for n, o in enumerate(self.vortices):
                oc = o["contour"]
                o_inds = oc["vortensity_min_inds"]
                o_mass = oc["mass"]
                if o_inds == inds and o_mass < mass:
                    to_del.append(n)
        for k, n in enumerate(set(to_del)):
            del self.vortices[n-k]

    def calculate_contour_properties(self):
        # Get the mass and vortensity inside the candidates

        for c in self.candidates.values():
            try:
                analyze.calc_vortex_mass(c, self.mass)
                analyze.calc_vortensity(c, self.vortensity)
                analyze.calc_sigma(c, self.surface_density)
                analyze.calc_vortex_extent(
                    c, self.area, self.radius, self.azimuth)
                analyze.find_vortensity_min_position(
                    c, self.radius, self.azimuth, self.vortensity)
                analyze.find_density_max_position(
                    c, self.radius, self.azimuth, self.surface_density)
            except (ValueError, RuntimeError) as e:
                # print("Warning: ValueError encountered in calculating vortex properties:", e)
                pass

    def fit_contours(self):
        for vortex in self.vortices:
            try:
                self.fit_gaussians(vortex)
            except (ValueError, RuntimeError) as e:
                # print("Warning: ValueError encountered in calculating vortex properties:", e)
                pass
            try:
                self.calc_fit_difference_2D(vortex)
            except KeyError as e:
                # print("Warning: KeyError encountered in calculating fit differences:", e)
                pass

    def fit_gaussians(self, vortex):

        contour = vortex["contour"]
        inds = contour["vortensity_min_inds"]

        def get_pos(inds):
            r = self.radius[inds]
            phi = self.azimuth[inds]
            return (r, phi)

        top = get_pos(contour["top"])
        left = get_pos(contour["left"])
        right = get_pos(contour["right"])
        bottom = get_pos(contour["bottom"])

        mask = contour["mask"]
        mask_r = mask[:, inds[1]]
        mask_phi = mask[inds[0], :]

        vals = self.surface_density
        R = self.radius[mask]
        PHI = self.azimuth[mask]

        Z = vals[mask]
        Z_r = vals[:, inds[1]]
        Z_phi = vals[inds[0], :]

        r0 = self.radius[inds]
        phi0 = self.azimuth[inds]

        c_ref = np.average(Z_phi[np.logical_not(mask_phi)])

        c_guess = c_ref
        a_guess = np.max(Z) - c_guess

        if bottom[1] > top[1]:
            if phi0 > 0:
                mask_up = PHI < phi0-np.pi
                mask_low = np.logical_not(mask_up)

                R_up = R[mask_up]
                PHI_up = PHI[mask_up]+2*np.pi
                Z_up = Z[mask_up]

                R_low = R[mask_low]
                PHI_low = PHI[mask_low]
                Z_low = Z[mask_low]

                R_fit = np.append(R_low, R_up)
                PHI_fit = np.append(PHI_low, PHI_up)
                Z_fit = np.append(Z_low, Z_up)
            else:
                mask_up = PHI > phi0+np.pi
                mask_low = np.logical_not(mask_up)

                R_up = R[mask_up]
                PHI_up = PHI[mask_up]-2*np.pi
                Z_up = Z[mask_up]

                R_low = R[mask_low]
                PHI_low = PHI[mask_low]
                Z_low = Z[mask_low]

                R_fit = np.append(R_low, R_up)
                PHI_fit = np.append(PHI_low, PHI_up)
                Z_fit = np.append(Z_low, Z_up)
        else:
            R_fit = R
            PHI_fit = PHI
            Z_fit = Z

        dr = np.max(R_fit) - np.min(R_fit)
        dphi = np.max(PHI_fit) - np.min(PHI_fit)
        fitter = Gauss2DFitter(R_fit, PHI_fit, Z_fit,
                               p0={"x0": r0, "y0": phi0,
                                   "c": 0.8*c_ref, "a": a_guess},
                               blow={"c": 0.75*c_ref, "a": 0.75*a_guess,
                                     "x0": r0-0.25*dr, "y0": phi0-0.25*dphi},
                               bup={"c": 1.25*c_ref, "a": 1.25*a_guess, "x0": r0+0.25*dr, "y0": phi0+0.25*dphi})
        p, cov = fitter.fit()

        fit = {"c": p[0], "a": p[1], "r0": p[2], "phi0": p[3],
               "sigma_r": p[4], "sigma_phi": p[5], "popt": p, "pcov": cov}
        vortex["fits"] = {"surface_density": fit}

        # print("Sigma fit parameters")
        # for name, val, guess, low, up in zip(fitter.parameters, popt_rho, fitter.p0.values(), fitter.blow.values(), fitter.bup.values()):
        #     print(f"{name:5s} {val: .2e} ({guess: .2e}) [{low: .2e}, {up: .2e}]")

        vals = self.vortensity
        R = self.radius[mask]
        PHI = self.azimuth[mask]
        Z = vals[mask]
        Z_r = vals[:, inds[1]]
        Z_phi = vals[inds[0], :]

        r0 = self.radius[inds]
        phi0 = self.azimuth[inds]

        if bottom[1] > top[1]:
            Z_up = Z[mask_up]
            Z_low = Z[mask_low]
            Z_fit = np.append(Z_low, Z_up)
        else:
            Z_fit = Z

        c_ref = np.average(Z_phi[np.logical_not(mask_phi)])

        c_guess = c_ref
        a_guess = np.min(Z) - c_guess

        fitter = Gauss2DFitter(R_fit, PHI_fit, Z_fit,
                               p0={"x0": r0, "y0": phi0,
                                   "c": c_ref, "a": a_guess},
                               blow={"c": 0.75*c_ref, "a": 1.25*a_guess},
                               bup={"c": 1.25*c_ref, "a": 0.75*a_guess})
        p, cov = fitter.fit()

        fit = {"c": p[0], "a": p[1], "r0": p[2], "phi0": p[3],
               "sigma_r": p[4], "sigma_phi": p[5], "popt": p, "pcov": cov}
        vortex["fits"]["vortensity"] = fit

    def calc_fit_difference_2D(self, v, varname="surface_density"):
        """ Calculate the difference of the fit to the data.

        Parameters
        ----------
        c : dict
            Vortex info.
        varname : str
            Name of the fit variable.
        """
        y0 = v["fits"][varname]["c"]
        a = v["fits"][varname]["a"]
        phi0 = v["fits"][varname]["phi0"]
        sigma_phi = v["fits"][varname]["sigma_phi"]
        r0 = v["fits"][varname]["r0"]
        sigma_r = v["fits"][varname]["sigma_r"]
        popt = v["fits"][varname]["popt"]

        def f(r, phi):
            return gauss2D((r, phi), *popt)

        if varname == "surface_density":
            vals = self.surface_density
        elif varname == "vortensity":
            vals = self.vortensity
        else:
            raise AttributeError(
                f"Can't calculate fit difference in 2D for '{varname}'!")

        R = self.radius
        PHI = self.azimuth
        hr = sigma_r*np.sqrt(2*np.log(2))
        hphi = sigma_phi*np.sqrt(2*np.log(2))

        mc = v["contour"]["mask"]
        me = ((R-r0)/hr)**2 + ((PHI-phi0)/hphi)**2 <= 1

        Ae = np.sum(self.area[me])
        Ac = v["contour"]["area"]
        v["fits"][varname]["properties"] = dict()
        v["fits"][varname]["properties"]["ellipse_area_numerical"] = Ae
        v["fits"][varname]["properties"]["area_ratio_ellipse_to_contour"] = Ae/Ac

        for region, mask, area in zip(["contour", "ellipse"], [mc, me], [Ac, Ae]):
            fitvals = f(R[mask], PHI[mask])
            numvals = vals[mask]
            diff = np.sum(np.abs(fitvals - numvals))
            reldiff = diff/(area*a)
            v["fits"][varname]["properties"][f"{region}_diff"] = diff
            v["fits"][varname]["properties"][f"{region}_reldiff"] = reldiff
            v["fits"][varname]["properties"][f"{region}_mass"] = np.sum(
                numvals*area)
            v["fits"][varname]["properties"][f"{region}_mass_fit"] = np.sum(
                fitvals*area)
