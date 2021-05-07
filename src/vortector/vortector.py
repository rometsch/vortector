import numpy as np

from . import analyze
from .gaussfit import fit_2D_gaussian_vortensity, fit_2D_gaussian_surface_density, gauss2D
from .contour import detect_elliptic_contours
from .interpolation import *


class Vortector:
    def __init__(self, radius, azimuth, area, vortensity, surface_density,
                 levels=None,
                 med=0.15, mear=np.inf, mvd=0.1, verbose=False, azimuthal_boundaries=[-np.pi, np.pi]):

        self.vortensity = vortensity

        self.surface_density = surface_density

        self.radius = radius
        self.radius_min = np.min(radius)
        self.radius_max = np.max(radius)

        self.azimuth = azimuth
        self.area = area

        self.azimuthal_boundaries = azimuthal_boundaries
        self.periodicity = {
            "upper": azimuthal_boundaries[1],
            "lower": azimuthal_boundaries[0],
            "L": azimuthal_boundaries[1] - azimuthal_boundaries[0]
        }

        # Parameters
        if levels is None:
            levels = np.linspace(-1, 1, 41)
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

    def detect(self, include_mask=False, keep_internals=False, linear_radius=False, autofit=True, **kwargs):

        if linear_radius:
            contour_data, reverse_trafo, trafo = interpolate_to_linear(
                self.vortensity, self.radius)
        else:
            contour_data = self.vortensity

        self.candidates = detect_elliptic_contours(
            contour_data,
            self.levels,
            self.mear,
            self.med,
            verbose=self.verbose,
            periodic=True,
            **kwargs)
        self.print(f"Detected {len(self.candidates)} elliptic contours.")

        if linear_radius:
            for c in self.candidates:
                c["mask"] = interpolate_radial(c["mask"], reverse_trafo)
                for s in ["pnt_xlow", "pnt_xhigh", "pnt_ylow", "pnt_yhigh"]:
                    c[s] = transform_indices(c[s], trafo)

        self.remove_duplicates_by_geometry()
        self.calculate_contour_properties()
        self.vortices = []
        for c in self.candidates:
            self.vortices.append({"contour": c})

        # self.remove_duplicates_by_boundary_area()
        self.sort_vortices_by_mass()
        self.create_hierarchy_by_min_vort_pos()

        # select only the parents
        self.candidates = {v["contour"]["detection"]
                           ["uuid"]: v for v in self.vortices}
        self.vortices = [c for c in self.vortices if len(
            c["hierarchy"]["parents"]) == 0]

        if autofit:
            for n in range(len(self.vortices)):
                self.fit(n)

        self.remove_non_vortex_candidates()

        if not keep_internals:
            del self.candidates
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
                del v["contour"]["detection"]["mask_img"]
            if not include_mask:
                del v["contour"]["mask"]

    def remove_non_vortex_candidates(self):
        """ Remove candidates not having vortex properties
        A vortex should have at least a small dip in vortensity.
        Exclude vortices for which the minimum vorticity is not at least 0.05 lower than the maximum vorticity.
        Also check that vortensity < 1
        """
        to_del = []
        for n, vortex in enumerate(self.vortices):
            c = vortex["contour"]
            cid = c["detection"]["opencv_contour_number"]
            try:
                if c["stats"]["vortensity_min"] > 1:
                    self.print(
                        f"Check candidates: excluding {cid} because of min_vortensity > 1")
                    to_del.append(n)
                    continue
                vortensity_drop = c["stats"]["vortensity_max"] - \
                    c["stats"]["vortensity_min"]
                if vortensity_drop < self.mvd:
                    to_del.append(n)
            except KeyError:
                # remove candiates causing errors
                to_del.append(n)
        to_del = set(to_del)
        self.vortices = [v for n, v in enumerate(
            self.vortices) if not n in to_del]

        self.print(
            f"Removed {len(to_del)} candidates by due to vortensity drop < {self.mvd}.")

    def remove_duplicates_by_min_vort_pos(self):
        """ Remove remaining duplicates. 

        Because of the way the mirror counter lines are generated,
        the mirror shape can be shifted slightly.
        Still both mirror shapes should share the same location of minimum.
        The shape containing the lower mass is deleted.
        """
        to_del = []
        for n_vort, v in enumerate(self.vortices):
            if n_vort in to_del:
                continue
            c = v["contour"]
            inds = c["stats"]["vortensity_min_inds"]
            area = int(c["detection"]["pixel_area"])

            for n_other, o in enumerate(self.vortices):
                if n_other == n_vort:
                    continue
                oc = o["contour"]
                o_inds = oc["stats"]["vortensity_min_inds"]
                o_area = oc["detection"]["pixel_area"]
                if o_inds == inds and o_area == area:
                    to_del.append(n_other)
        to_del = set(to_del)
        self.vortices = [v for n, v in enumerate(
            self.vortices) if not n in to_del]
        self.print(
            f"Removed {len(to_del)} candidates by common vortensity min position. {len(self.vortices)} remaining")

    def remove_duplicates_by_boundary_area(self):
        """ Remove remaining duplicates. 

        Hash the array of boundary points and look for matches.
        Remove the contour with the lower contour value.
        """
        to_del = []

        for n_vort, v in enumerate(self.vortices):
            if n_vort in to_del:
                continue
            c = v["contour"]
            v_area = c["detection"]["pixel_area"]

            for n_other, o in enumerate(self.vortices):
                if n_other == n_vort:
                    continue
                oc = o["contour"]
                o_area = oc["detection"]["pixel_area"]

                if v_area == o_area:
                    to_del.append(n_other)

        to_del = set(to_del)
        self.vortices = [v for n, v in enumerate(
            self.vortices) if not n in to_del]

        self.print(
            f"Removed {len(to_del)} candidates by common boundary area. {len(self.vortices)} remaining")
        
    def remove_duplicates_by_geometry(self):
        """ Remove remaining duplicates. 

        Hash the array of boundary points and look for matches.
        Remove the contour with the lower contour value.
        """
        to_del = []

        for n_can, c in enumerate(self.candidates):
            if n_can in to_del:
                continue
            bbox = [c[f"pnt_{key}"] for key in ["xlow", "xhigh", "ylow", "yhigh"]]
            area = c["detection"]["pixel_area"]

            for n_other, o in enumerate(self.candidates):
                if n_other == n_can:
                    continue
                o_bbox = [o[f"pnt_{key}"] for key in ["xlow", "xhigh", "ylow", "yhigh"]]
                o_area = o["detection"]["pixel_area"]

                if area == o_area and bbox == o_bbox:
                    to_del.append(n_other)

        to_del = set(to_del)
        self.candidates = [v for n, v in enumerate(
            self.candidates) if not n in to_del]

        self.print(
            f"Removed {len(to_del)} candidates by common area and bbox. {len(self.candidates)} remaining")

    def create_hierarchy_by_min_vort_pos(self):
        """ Create a hierarchy of the vortex candidates.

        For every candidate we look at all other candidates 
        and check whether they have the same position of the vortex minimum.
        Then we create a hierarchy by appending the smaller one to the larger one as child
        and the larger one to the smaller one as a parent.
        """
        for vort in self.vortices:
            vort["hierarchy"] = {"parents": [], "children": []}
        for n_vort, v in enumerate(self.vortices):
            c = v["contour"]
            inds = c["stats"]["vortensity_min_inds"]
            vortmin = c["stats"]["vortensity_min"]
            mass = c["stats"]["mass"]
            v_area = c["detection"]["pixel_area"]
            contour_val = c["detection"]["contour_value"]

            vort_info = (contour_val, mass, v_area, inds,
                         vortmin, c["detection"]["uuid"])

            for n_other, o in enumerate(self.vortices):
                if n_vort == n_other:
                    continue
                oc = o["contour"]
                o_inds = oc["stats"]["vortensity_min_inds"]
                o_vortmin = oc["stats"]["vortensity_min"]
                o_mass = oc["stats"]["mass"]
                o_area = oc["detection"]["pixel_area"]
                o_contour_val = oc["detection"]["contour_value"]

                o_info = (o_contour_val, o_mass, o_area, o_inds,
                          o_vortmin, oc["detection"]["uuid"])

                if o_inds == inds:
                    if (o_contour_val < contour_val) and (o_mass < mass):
                        v["hierarchy"]["children"].append(o_info)
                    else:
                        v["hierarchy"]["parents"].append(o_info)

        for vort in self.vortices:
            h = vort["hierarchy"]
            h["parents"] = sorted(set(h["parents"]), key=lambda x: -x[0])
            h["children"] = sorted(set(h["children"]), key=lambda x: -x[0])

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

    def fit(self, n, region="contour"):
        vortex = self.vortices[n]
        mask_full = self.get_mask(vortex, region=region)
        r, phi, vort, dens, mask = self.extract_data(vortex, region=region)
        r = r.ravel()
        phi = phi.ravel()
        vort = vort.ravel()
        dens = dens.ravel()
        try:
            r0, phi0 = vortex["contour"]["stats"]["vortensity_min_pos"]
            r0_ind = vortex["contour"]["stats"]["vortensity_min_inds"][0]
            oazi = self.vortensity[r0_ind, ~mask_full[r0_ind]]
            background_val = np.sort(oazi)[-len(oazi)//4]
            fit = fit_2D_gaussian_vortensity(
                vortex["contour"],
                r, phi, vort,
                r0, phi0, background_val,
                periodicity=self.periodicity)
            vortex["fits"] = {"vortensity": fit}
            vortex["fits"]["vortensity"]["fit_region"] = region
        except (ValueError, RuntimeError) as e:
            self.print(
                "Warning: Error while fitting vortensity:", e)
            pass
        try:
            background_val = 1.0
            r0, phi0 = vortex["contour"]["stats"]["surface_density_max_pos"]
            r0_ind = vortex["contour"]["stats"]["surface_density_max_inds"][0]
            oazi = self.surface_density[r0_ind, ~mask_full[r0_ind]]
            background_val = np.sort(oazi)[len(oazi)//4]
            fit = fit_2D_gaussian_surface_density(
                vortex["contour"],
                r, phi, dens,
                r0, phi0, background_val,
                periodicity=self.periodicity)
            vortex["fits"]["surface_density"] = fit
            vortex["fits"]["surface_density"]["fit_region"] = region
        except (ValueError, RuntimeError) as e:
            self.print(
                "Warning: Error while fitting surface density:", e)
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
            diff = np.sum(np.abs(fitvals - numvals)*self.area[mask])
            reldiff = diff/(area*a)
            v["fits"][varname]["properties"][f"{region}_diff"] = diff
            v["fits"][varname]["properties"][f"{region}_reldiff"] = reldiff
            v["fits"][varname]["properties"][f"{region}_mass"] = np.sum(
                numvals*area)
            v["fits"][varname]["properties"][f"{region}_mass_fit"] = np.sum(
                fitvals*area)

    def extract_data(self, vortex, region="contour"):
        """Extract the data inside the vortex region. 

        Parameters
        ----------
        vortex : dict
            Vortex dictionary as returned by Vortector.
        region : str, optional
            Select the region to be used, by default "contour".
            If region=="contour":
                Use the contour as region.
            If region=="vortensity":
                Use the vortensity fit FWHM ellipse as region.
            If region=="surface_density":
                Use the surface density fit FWHM ellipse as region.
            If region=="combined":
                Use the combination of all of the above regions.
        """
        mask = self.get_mask(vortex, region=region)
        bbox = self.get_bbox(vortex, region=region)

        xlow = bbox["xlow"]["inds"][0]
        xhigh = bbox["xhigh"]["inds"][0]
        ylow = bbox["ylow"]["inds"][1]
        yhigh = bbox["yhigh"]["inds"][1]

        if ylow < yhigh:
            r = self.radius[xlow:xhigh, ylow:yhigh]
            phi = self.azimuth[xlow:xhigh, ylow:yhigh]
            dens = self.surface_density[xlow:xhigh, ylow:yhigh]
            vort = self.vortensity[xlow:xhigh, ylow:yhigh]
            if mask is not None:
                mask = mask[xlow:xhigh, ylow:yhigh]
            else:
                mask = np.ones(r.shape, dtype=bool)
        else:
            Ny = self.radius.shape[1]
            r = np.pad(self.radius, [[0, 0], [0, yhigh]], mode="wrap")[
                xlow:xhigh, ylow:Ny+yhigh]
            phi = np.pad(self.azimuth, [[0, 0], [0, yhigh]], mode="wrap")[
                xlow:xhigh, ylow:Ny+yhigh]
            phi[phi < self.azimuth[0, ylow]] += self.periodicity["L"]
            dens = np.pad(self.surface_density, [[0, 0], [0, yhigh]], mode="wrap")[
                xlow:xhigh, ylow:Ny+yhigh]
            vort = np.pad(self.vortensity, [[0, 0], [0, yhigh]], mode="wrap")[
                xlow:xhigh, ylow:Ny+yhigh]
            if mask is not None:
                mask = np.pad(mask, [[0, 0], [0, yhigh]], mode="wrap")[
                    xlow:xhigh, ylow:Ny+yhigh]
            else:
                mask = np.ones(r.shape, dtype=bool)

        return r, phi, vort, dens, mask

    def merge_bounding_boxes(self, bb1, bb2):
        bbox = {}
        # left
        if bb1["ylow"]["inds"][0] < bb2["ylow"]["inds"][0]:
            bbox["ylow"] = dict(bb1["ylow"])
        else:
            bbox["ylow"] = dict(bb2["ylow"])
        # right
        if bb1["yhigh"]["inds"][0] > bb2["yhigh"]["inds"][0]:
            bbox["yhigh"] = dict(bb1["yhigh"])
        else:
            bbox["yhigh"] = dict(bb2["yhigh"])

        N = self.radius.shape[1]
        t1 = bb1["xlow"]["inds"][1]
        b1 = bb1["xhigh"]["inds"][1]
        t2 = bb2["xlow"]["inds"][1]
        b2 = bb2["xhigh"]["inds"][1]

        # consider periodicity
        if b1 > t1:
            t1 += N
        if b2 > t2:
            t2 += N

        if t1 > t2:
            bbox["xlow"] = dict(bb1["xlow"])
        else:
            bbox["xlow"] = dict(bb2["xlow"])

        if b1 < b2:
            bbox["xhigh"] = dict(bb1["xhigh"])
        else:
            bbox["xhigh"] = dict(bb2["xhigh"])

        return bbox

    def contour_bounding_box(self, vortex):
        """ Extract the bounding indices of the contour of the vortex.

        Parameters
        ----------
        vortex : dict
            Vortex as returned by the Vortector.
        """
        box = {
            "xlow": {"inds": vortex["contour"]["pnt_xlow"]},
            "xhigh": {"inds": vortex["contour"]["pnt_xhigh"]},
            "ylow": {"inds": vortex["contour"]["pnt_ylow"]},
            "yhigh": {"inds": vortex["contour"]["pnt_yhigh"]},
        }

        rs = self.radius[:, 0]
        phis = self.azimuth[0]
        for d in box.values():
            inds = d["inds"]
            d["pos"] = [rs[inds[0]], phis[inds[1]]]

        return box

    def ellipse_bounding_box(self, fit):
        """ Extract the bounding indices of the fit ellipse.

        Parameters
        ----------
        fit : dict
            Dict containing the fit parameters: r0, phi0, sigma_r, sigma_phi.
        """
        r0 = fit["r0"]
        hr = np.sqrt(2*np.log(2))*fit["sigma_r"]
        phi0 = fit["phi0"]
        hphi = np.sqrt(2*np.log(2))*fit["sigma_phi"]

        r_up = min(r0 + hr, self.radius_max)
        r_low = max(r0 - hr, self.radius_min)

        bnd_low = self.periodicity["lower"]
        L = self.periodicity["L"]
        phi_up = ((phi0 + hphi) - bnd_low) % L + bnd_low
        phi_low = ((phi0 - hphi) - bnd_low) % L + bnd_low

        box = {
            "xlow": {"pos": [r_low, phi0]},
            "xhigh": {"pos": [r_up, phi0]},
            "ylow": {"pos": [r0, phi_low]},
            "yhigh": {"pos": [r0, phi_up]}
        }

        rs = self.radius[:, 0]
        phis = self.azimuth[0]
        for d in box.values():
            r, phi = d["pos"]
            r_ind = np.argmin(np.abs(rs-r))
            phi_ind = np.argmin(np.abs(phis-phi))
            d["inds"] = [r_ind, phi_ind]

        return box

    def ellipse_mask(self, fit):
        """Create a mask indicating the fit ellipse for varname.

        Parameters
        ----------
        fit : dict
            Dict containing the fit parameters: r0, phi0, sigma_r, sigma_phi.

        Returns
        -------
        np.array, 2D, dtype=bool
            Array mask indicating the ellipse region.

        Raises
        ------
        IndexError
            If the selected fit does not exist.
        """
        r0 = fit["r0"]
        hr = np.sqrt(2*np.log(2))*fit["sigma_r"]
        phi0 = fit["phi0"]
        hphi = np.sqrt(2*np.log(2))*fit["sigma_phi"]

        mask = ((self.radius - r0)/hr)**2 + \
            ((self.azimuth - phi0)/hphi)**2 <= 1

        return mask

    def combined_mask(self, vortex):
        """Create a mask indicating the combined area of the contour and both fit ellipses.

        Parameters
        ----------
        vortex : dict
            Vortex dict as returned by Vortector.

        Returns
        -------
        np.array, 2D, dtype=bool
            Array mask indicating the combined region.

        Raises
        ------
        IndexError
            If the fits do not exist.
        """
        mask_contour = vortex["contour"]["mask"]
        mask_vort = self.ellipse_mask(vortex["fits"]["vortensity"])
        mask_dens = self.ellipse_mask(vortex["fits"]["surface_density"])

        mask_ellipses = np.logical_or(mask_vort, mask_dens)
        mask = np.logical_or(mask_contour, mask_ellipses)

        return mask

    def get_bbox(self, vortex, region="contour"):
        """A boundary box for the vortex region.

        Parameters
        ----------
        vortex : dict
            Vortex dictionary as returned by Vortector.
        region : str, optional
            Select the region to be used, by default "contour".
            If region=="contour":
                Use the contour as region.
            If region=="vortensity":
                Use the vortensity fit FWHM ellipse as region.
            If region=="surface_density":
                Use the surface density fit FWHM ellipse as region.
            If region=="combined":
                Use the combination of all of the above regions.
        """
        if region == "contour":
            bbox = self.contour_bounding_box(vortex)
        elif region in ["vortensity", "surface_density"]:
            bbox = self.ellipse_bounding_box(vortex["fits"][region])
        elif region == "combined":
            bbox = self.contour_bounding_box(vortex)
            try:
                bbox2 = self.ellipse_bounding_box(vortex["fits"]["vortensity"])
                bbox = self.merge_bounding_boxes(bbox, bbox2)
            except KeyError:
                pass

            try:
                bbox3 = self.ellipse_bounding_box(
                    vortex["fits"]["surface_density"])
                bbox = self.merge_bounding_boxes(bbox, bbox3)
            except KeyError:
                pass
        else:
            raise ValueError(f"Invalid region '{region}'")

        return bbox

    def get_mask(self, vortex, region="contour"):
        """A 2d array indicating the vortex.

        Parameters
        ----------
        vortex : dict
            Vortex dictionary as returned by Vortector.
        region : str, optional
            Select the region to be used, by default "contour".
            If region=="contour":
                Use the contour as region.
            If region=="vortensity":
                Use the vortensity fit FWHM ellipse as region.
            If region=="surface_density":
                Use the surface density fit FWHM ellipse as region.
            If region=="combined":
                Use the combination of all of the above regions.
        """
        if region == "contour":
            mask = vortex["contour"]["mask"]
        elif region in ["vortensity", "surface_density"]:
            mask = self.ellipse_mask(vortex["fits"][region])
        elif region == "combined":
            mask = self.get_mask(vortex, region="contour")
            try:
                mask2 = self.get_mask(vortex, region="vortensity")
                mask = np.logical_or(mask, mask2)
            except KeyError:
                pass

            try:
                mask3 = self.get_mask(vortex, region="surface_density")
                mask = np.logical_or(mask, mask3)
            except KeyError:
                pass
        else:
            raise ValueError(f"Invalid region '{region}'")

        return mask


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
        if vortex["contour"]["stats"]["mass"] > ref_mass/10:
            large_vortices.append(vortex)

    # sort by relative azimuthal vortensity
    rel_vort = [v["contour"]["stats"]["vortensity_min"]/v["contour"]["stats"]
                ["azimuthal_at_vortensity_min"]["vortensity_med"]for v in large_vortices]
    inds = np.argsort(rel_vort)
    large_vortices = [large_vortices[n] for n in inds]

    vortices_with_fit = []
    for vortex in large_vortices:
        if "fits" in vortex:
            vortices_with_fit.append(vortex)

    if len(vortices_with_fit) > 0:
        return vortices_with_fit[0]

    sorted_vortices = sorted(
        large_vortices, key=lambda x: x["contour"]["stats"]["vortensity_min"])
    return sorted_vortices[0]
