import numpy as np

from . import analyze
from .gaussfit import fit_2D_gaussian_vortensity, fit_2D_gaussian_surface_density, gauss2D
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
        self.periodicity = {
            "upper": azimuthal_boundaries[1],
            "lower": azimuthal_boundaries[0],
            "L": azimuthal_boundaries[1] - azimuthal_boundaries[0]
        }

        # Parameters
        self.levels = levels
        self.med = med  # max_ellipse_deviation
        self.mear = mear  # max_ellipse_aspect_ratio
        self.mvd = mvd  # min_vortensity_drop

        self.verbose = verbose

        self.mass = self.area*self.surface_density

    def print(self, *args, **kwargs):
        """ Print when verbose flag is set. """
        if self.verbose:
            print(*args, **kwargs)

    def detect(self, include_mask=False, keep_internals=False):

        self.candidates = detect_elliptic_contours(self.vortensity,
                                                   self.levels,
                                                   self.mear,
                                                   self.med,
                                                   self.verbose)

        self.calculate_contour_properties()

        self.vortices = []
        for c in self.candidates:
            self.vortices.append({"contour": c})

        self.fit_gaussians()
        self.remove_non_vortex_candidates()
        self.remove_duplicates_by_min_vort_pos()

        self.sort_vortices_by_mass()

        self.remove_intermediate_data(include_mask, keep_internals)

        return self.vortices

    def guess_main_vortex(self):
        return choose_main_vortex(self.vortices)

    def sort_vortices_by_mass(self):
        """ Sort the vortices by mass decending. """
        self.vortices = [v for v in reversed(
            sorted(self.vortices, key=lambda v: v["contour"]["stats"]["mass"]))]

    def remove_intermediate_data(self, include_mask=False, keep_internals=False):
        for v in self.vortices:
            if not keep_internals:
                del v["contour"]["detection"]["boundary"]
                del v["contour"]["detection"]["mask_extended"]
            if not include_mask:
                del v["contour"]["mask"]

    def remove_non_vortex_candidates(self):
        """ Remove candidates not having vortex properties
        A vortex should have at least a small dip in vortensity.
        Exclude vortices for which the minimum vorticity is not at least 0.05 lower than the maximum vorticity.
        Also check that vortensity < 1
        """

        no_min = []
        for n, vortex in enumerate(self.vortices):
            c = vortex["contour"]
            cid = c["detection"]["opencv_contour_number"]
            try:
                if c["stats"]["vortensity_min"] > 1:
                    self.print(
                        f"Check candidates: excluding {cid} because of min_vortensity > 1")
                    no_min.append(n)
                    continue
                vortensity_drop = c["stats"]["vortensity_max"] - \
                    c["stats"]["vortensity_min"]
                if vortensity_drop < self.mvd:
                    self.print(
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
            inds = c["stats"]["vortensity_min_inds"]
            mass = c["stats"]["mass"]

            for n, o in enumerate(self.vortices):
                oc = o["contour"]
                o_inds = oc["stats"]["vortensity_min_inds"]
                o_mass = oc["stats"]["mass"]
                if o_inds == inds and o_mass < mass:
                    to_del.append(n)
        for k, n in enumerate(set(to_del)):
            del self.vortices[n-k]

    def calculate_contour_properties(self):
        # Get the mass and vortensity inside the candidates

        for c in self.candidates:
            try:
                c["stats"] = dict()
                analyze.calc_vortex_mass(c, self.mass)
                analyze.calc_vortensity(c, self.vortensity)
                analyze.calc_sigma(c, self.surface_density)
                analyze.calc_vortex_extent(
                    c, self.area, self.radius, self.azimuth)
                analyze.find_vortensity_min_position(
                    c, self.radius, self.azimuth, self.vortensity)
                analyze.find_density_max_position(
                    c, self.radius, self.azimuth, self.surface_density)
                analyze.calc_azimuthal_statistics(
                    c["stats"], self.surface_density, self.vortensity)
            except (ValueError, RuntimeError) as e:
                self.print(
                    "Warning: ValueError encountered in calculating vortex properties:", e)
                pass

    def fit_gaussians(self):
        for vortex in self.vortices:
            try:
                fit = fit_2D_gaussian_vortensity(
                    vortex["contour"],
                    self.radius,
                    self.azimuth,
                    self.vortensity,
                    "vortensity_min_inds",
                    periodicity=self.periodicity)
                vortex["fits"] = {"vortensity": fit}
                fit = fit_2D_gaussian_surface_density(
                    vortex["contour"],
                    self.radius,
                    self.azimuth,
                    self.surface_density,
                    "surface_density_max_inds",
                    periodicity=self.periodicity)
                vortex["fits"]["surface_density"] = fit
            except (ValueError, RuntimeError) as e:
                self.print(
                    "Warning: ValueError encountered in calculating vortex properties:", e)
                pass
            try:
                self.calc_fit_difference_2D(vortex)
            except KeyError as e:
                self.print(
                    "Warning: KeyError encountered in calculating fit differences:", e)
                pass

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
        Ac = v["contour"]["stats"]["area"]
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


def choose_main_vortex(vortices):
    """ Choose the most likely largest legitimate vortex from a list of candidates. """
    if len(vortices) == 0:
        return None
    if len(vortices) == 1:
        return vortices[0]

    large_vortices = [vortices[0]]
    ref_mass = vortices[0]["contour"]["stats"]["mass"]
    # keep vortices that have 20% of most massive's mass
    for vortex in vortices[1:]:
        if vortex["contour"]["stats"]["mass"] > ref_mass/5:
            large_vortices.append(vortex)

    vortices_with_fit = []
    for vortex in large_vortices:
        if "fits" in vortex:
            vortices_with_fit.append(vortex)

    if len(vortices_with_fit) > 0:
        return vortices_with_fit[0]

    sorted_vortices = sorted(
        large_vortices, key=lambda x: x["contour"]["stats"]["vortensity_min"])
    return sorted_vortices[0]
