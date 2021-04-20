import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

from .gaussfit import gauss, gauss2D, Gauss2DFitter
from . import plotperiodic as plotper
from . import plot


class Vortector:
    def __init__(self, Xc, Yc, A, vortensity, Sigma,
                 levels=[float(x) for x in np.arange(-1, 1.5, 0.05)],
                 med=0.15, mear=np.inf, mvd=0.01, verbose=False, azimuthal_boundaries=[-np.pi, np.pi]):

        self.Vortensity = vortensity

        self.SurfaceDensity = Sigma

        self.Rc = Xc
        self.Xc = Xc
        self.Phic = Yc
        self.Yc = Yc
        self.Area = A

        self.azimuthal_boundaries = azimuthal_boundaries

        # Parameters
        self.levels = levels
        self.max_ellipse_deviation = med
        self.max_ellipse_aspect_ratio = mear
        self.min_vortensity_drop = mvd

        self.verbose = verbose

    def contour_image_dimensions(self):

        self.Nx, self.Ny = self.Vortensity.shape
        self.int_aspect = int(np.max([self.Nx/self.Ny, self.Ny/self.Nx]))

        if self.int_aspect >= 2:
            if self.Nx < self.Ny:
                self.CNx = self.int_aspect*self.Nx
                self.CNy = self.Ny
            else:
                self.CNx = self.Nx
                self.CNy = self.int_aspect*self.Ny
        else:
            self.CNx = self.Nx
            self.CNy = self.Ny

        if min(self.CNx, self.CNy) < 1000:
            self.supersample = int(np.ceil(1000/min(self.CNx, self.CNy)))
        else:
            self.supersample = 1

        self.CNx *= self.supersample
        self.CNy *= self.supersample

        if self.verbose:
            print(
                f"Contour image dimensions: Nx {self.Nx}, Ny {self.Ny}, int_aspect {self.int_aspect}, supersample {self.supersample}, CNx {self.CNx}, CNy {self.CNy}")

    def contour_image(self):
        fig = plt.figure(frameon=False, figsize=(self.CNx, 2*self.CNy), dpi=1)
        # fig.set_size_inches(w,h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # periodically extend vortensity
        vort = self.Vortensity
        Hhalf = int(vort.shape[1]/2)
        vort_pe = np.concatenate(
            [vort[:, Hhalf:],
             vort,
             vort[:, :Hhalf]],
            axis=1
        )

        ax.contour(vort_pe.transpose(),
                   levels=self.levels, linewidths=self.CNx/1000)

        img_data = fig2rgb_array(fig)
        plt.close(fig)

        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        # Threshold contour image for full contrast
        _, self.thresh = cv2.threshold(img_data, 250, 255, 0)

    def find_contours(self):

        # Extract contours and construct hierarchy

        contours, self.hierarchy = cv2.findContours(
            self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if self.verbose:
            print("Number of found contours:", len(contours))

        contours_dict = {n: {"contour": cnt, "opencv_contour_number": n}
                         for n, cnt in enumerate(contours)}

        areas = [cv2.contourArea(c) for c in contours]
        for n, d in enumerate(contours_dict.values()):
            d["pixel_area"] = areas[n]

        sort_inds = np.argsort(areas)

        # take the up to 100 largest patches
        self.contours_largest = [contours_dict[i]
                                 for i in [n for n in sort_inds[::-1]][:100]]

    def extract_closed_contours(self):
        # Extract closed contours

        self.contours_closed = []
        for n, contour in enumerate(self.contours_largest):
            cnt = contour["contour"]
            l = cv2.arcLength(cnt, True)
            contour["pixel_arcLength"] = l
            a = contour["pixel_area"]
            leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
            rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
            topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
            bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
            dx = rightmost[0] - leftmost[0]
            dy = bottommost[1] - topmost[1]

            Nh = int(self.thresh.shape[0]/2)
            Nq = int(self.thresh.shape[0]/4)

            # sort out mirrors of contours fully contained in original area
            if bottommost[1] < Nq or topmost[1] > 3*Nq:
                continue

            is_not_too_elongated = dx > 0 and dy > 0 and max(
                dx/dy, dy/dx) < self.max_ellipse_aspect_ratio
            is_area_larget_delimiter = l > 0 and a > l
            is_not_spanning_whole_height = dy < 0.5*0.95*self.thresh.shape[0]

            if not(is_not_too_elongated and is_area_larget_delimiter and is_not_spanning_whole_height):
                continue

            # sort out the lower of mirror images
            bounding_hor = np.array([rightmost[0], leftmost[0]])
            bounding_vert = np.array([topmost[1], bottommost[1]])
            contour["bounding_hor"] = bounding_hor
            contour["bounding_vert"] = bounding_vert
            # save the bounding points
            contour["bottom_extended"] = bottommost
            contour["top_extended"] = topmost
            contour["left_extended"] = leftmost
            contour["right_extended"] = rightmost

            to_del = None
            found_mirror = False
            for k, c in enumerate(self.contours_closed):
                same_hor = (bounding_hor == c["bounding_hor"]).all()
                same_vert = (np.abs(bounding_vert %
                                    Nh - c["bounding_vert"] % Nh) < 20).all()
                if same_hor and same_vert:
                    if bounding_vert[1] > c["bounding_vert"][1]:
                        to_del = k
                    found_mirror = True
                    break

            if found_mirror:
                if to_del is not None:
                    del self.contours_closed[to_del]
                    self.contours_closed.append(contour)
            else:
                self.contours_closed.append(contour)

        if self.verbose:
            print("Number of closed contours:", len(self.contours_closed))
            print("Area of contours:", [c["pixel_area"]
                                        for c in self.contours_closed])

    def extract_ellipse_contours(self):
        # Extract contours that match an ellipse

        self.candidates = {}
        for contour in self.contours_closed:
            cnt = contour["contour"]
            ellipse = cv2.fitEllipse(cnt)

            im_shape = np.zeros(self.thresh.shape)
            cv2.drawContours(im_shape, [cnt], 0, (255, 255, 255), -1)

            im_ellipse = np.zeros(self.thresh.shape)
            im_ellipse = cv2.ellipse(im_ellipse, ellipse, (255, 255, 255), -1)

            difference = np.abs(im_shape - im_ellipse)
            difference_area = np.sum(difference/255)

            rel_delta = difference_area / contour["pixel_area"]

            if rel_delta > self.max_ellipse_deviation:
                continue

            contour["mask_extended"] = im_shape
            self.candidates[contour["opencv_contour_number"]] = contour

    def create_vortex_mask(self):
        # Transform the image from ellipse fitting images back to match the grid

        for contour in self.candidates.values():
            mask_extended = contour["mask_extended"]
            # reduce back to normal image size
            Nq = int(mask_extended.shape[0]/4)
            mask_lower = mask_extended[3*Nq:, :]
            mask_upper = mask_extended[:Nq, :]
            mask_repeated = np.concatenate([mask_lower, mask_upper])
            mask_orig = mask_extended[Nq:3*Nq, :]
            mask_reduced = np.logical_or(mask_orig, mask_repeated)

            # fit back to original data shape
            mask = mask_reduced.transpose()[:, ::-1]
            mask = mask[::self.supersample, ::self.supersample]
            if self.int_aspect >= 2:
                if self.Nx < self.Ny:
                    mask = mask[::self.int_aspect, :]
                else:
                    mask = mask[:, ::self.int_aspect]
            mask = np.array(mask, dtype=bool)
            contour["mask"] = mask

            # map the bounding points to view and data shape
            for key in ["bottom", "top", "left", "right"]:
                pnt = contour[key + "_extended"]
                x, y = map_ext_pnt_to_orig(pnt, Nq)
                y = 2*Nq - y
                x /= self.supersample
                y /= self.supersample
                if self.Nx < self.Ny:
                    x /= self.int_aspect
                else:
                    y /= self.int_aspect
                x = int(x)
                y = int(y)
                contour[key] = (x, y)

        if self.verbose:
            print(
                f"Mapping mask: mask.shape = {mask.shape}, mask_orig.shape {mask_orig.shape}")

    def calc_cell_masses(self):
        self.mass = self.Area*self.SurfaceDensity

    def generate_ancestors(self):
        # Generate ancestor list

        # The hierarchy generated by opencv in the contour finder outputs a list with the syntax
        # ```
        # [Next, Previous, First_Child, Parent]
        # ```
        # If any of those is not available its encoded  by -1.

        for c in self.candidates.values():
            ancestors = []
            n_parent = c["opencv_contour_number"]
            for n in range(1000):
                n_parent = self.hierarchy[0, n_parent, 3]
                if n_parent == -1 or n_parent not in self.candidates:
                    break
                ancestors.append(n_parent)
            c["ancestors"] = ancestors
            if self.verbose:
                print("Ancestors:", c["opencv_contour_number"], ancestors)

    def generate_decendents(self):
        # Construct decendents from ancestor list
        # This is done to avoid causing trouble when an intermediate contour is missing.

        decendents = {}
        for c in self.candidates.values():
            ancestors = c["ancestors"]
            for k, n in enumerate(ancestors):
                if not n in decendents or len(decendents[n]) < k:
                    decendents[n] = [i for i in reversed(ancestors[:k])]

        for c in self.candidates.values():
            if c["opencv_contour_number"] in decendents:
                dec = decendents[c["opencv_contour_number"]]
            else:
                dec = []
            c["decendents"] = dec
            if self.verbose:
                print("Descendents:", c["opencv_contour_number"], dec)

    def prune_candidates_by_hierarchy(self):
        # Remove children from candidates

        decendents = []
        for c in self.candidates.values():
            decendents += c["decendents"].copy()
        decendents = set(decendents)
        for n in decendents:
            del self.candidates[n]

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
                if vortensity_drop < self.min_vortensity_drop:
                    if self.verbose:
                        print(
                            f"Check candidates: excluding {cid} vortensity drop is {vortensity_drop} < {self.min_vortensity_drop}")
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

    def calculate_contour_properties(self):
        # Get the mass and vortensity inside the candidates

        for c in self.candidates.values():
            try:
                self.calc_vortex_mass(c)
                self.calc_vortensity(c)
                self.calc_sigma(c)
                self.calc_vortex_extent(c)
                self.calc_vortensity_flux(c)
                self.find_vortensity_min_position(c)
                self.find_density_max_position(c)
            except (ValueError, RuntimeError) as e:
                # print("Warning: ValueError encountered in calculating vortex properties:", e)
                pass

    def fit_gaussians(self, vortex):

        contour = vortex["contour"]
        inds = contour["vortensity_min_inds"]

        def get_pos(inds):
            r = self.Xc[inds]
            phi = self.Yc[inds]
            return (r, phi)

        top = get_pos(contour["top"])
        left = get_pos(contour["left"])
        right = get_pos(contour["right"])
        bottom = get_pos(contour["bottom"])

        mask = contour["mask"]
        mask_r = mask[:, inds[1]]
        mask_phi = mask[inds[0], :]

        vals = self.SurfaceDensity
        R = self.Xc[mask]
        PHI = self.Yc[mask]

        Z = vals[mask]
        Z_r = vals[:, inds[1]]
        Z_phi = vals[inds[0], :]

        r0 = self.Xc[inds]
        phi0 = self.Yc[inds]

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

        vals = self.Vortensity
        R = self.Xc[mask]
        PHI = self.Yc[mask]
        Z = vals[mask]
        Z_r = vals[:, inds[1]]
        Z_phi = vals[inds[0], :]

        r0 = self.Xc[inds]
        phi0 = self.Yc[inds]

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
            vals = self.SurfaceDensity
        elif varname == "vortensity":
            vals = self.Vortensity
        else:
            raise AttributeError(
                f"Can't calculate fit difference in 2D for '{varname}'!")

        R = self.Xc
        PHI = self.Yc
        hr = sigma_r*np.sqrt(2*np.log(2))
        hphi = sigma_phi*np.sqrt(2*np.log(2))

        mc = v["contour"]["mask"]
        me = ((R-r0)/hr)**2 + ((PHI-phi0)/hphi)**2 <= 1

        Ae = np.sum(self.Area[me])
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

    def calc_vortex_mass(self, c):
        mask = c["mask"]
        c["mass"] = np.sum(self.mass[mask])

    def calc_vortensity(self, c):
        mask = c["mask"]
        c["vortensity_mean"] = np.mean(self.Vortensity[mask])
        c["vortensity_median"] = np.median(self.Vortensity[mask])
        c["vortensity_min"] = np.min(self.Vortensity[mask])
        c["vortensity_max"] = np.max(self.Vortensity[mask])

    def calc_sigma(self, c):
        mask = c["mask"]
        c["sigma_mean"] = np.mean(self.SurfaceDensity[mask])
        c["sigma_median"] = np.median(self.SurfaceDensity[mask])
        c["sigma_min"] = np.min(self.SurfaceDensity[mask])
        c["sigma_max"] = np.max(self.SurfaceDensity[mask])

    def calc_vortex_extent(self, c):
        mask = c["mask"]
        c["area"] = np.sum(self.Area[mask])
        c["rmax"] = self.Xc[c["left"]]
        c["rmin"] = self.Xc[c["right"]]
        c["phimin"] = self.Yc[c["top"]]
        c["phimax"] = self.Yc[c["bottom"]]
        if c["phimax"] < c["phimin"]:
            c["height"] = c["phimax"] + 2*np.pi - c["phimin"]
        else:
            c["height"] = c["phimax"] - c["phimin"]

    def calc_vortensity_flux(self, c):
        mask = c["mask"]
        A = self.Area
        c["vortensity_flux"] = np.sum((A*self.Vortensity)[mask])
        c["vortensity_exp_flux"] = np.sum(
            (A*np.exp(-self.Vortensity))[mask])

    def find_vortensity_min_position(self, contour):
        # Calculate the position of minimum vortensity
        mask = np.logical_not(contour["mask"])
        ind = np.argmin(np.ma.masked_array(
            self.Vortensity, mask=mask), axis=None)
        inds = np.unravel_index(ind, mask.shape)
        x = self.Xc[inds]
        y = self.Yc[inds]
        contour["vortensity_min_pos"] = (x, y)
        contour["vortensity_min_inds"] = inds
        if self.verbose:
            print(f"Location of minimum vortensity: (x,y) = ({x}, {y})")

    def find_density_max_position(self, contour):
        # Calculate the position of maximum density
        mask = np.logical_not(contour["mask"])
        ind = np.argmax(np.ma.masked_array(
            self.SurfaceDensity, mask=mask), axis=None)
        inds = np.unravel_index(ind, mask.shape)
        x = self.Xc[inds]
        y = self.Yc[inds]
        contour["sigma_max_pos"] = (x, y)
        contour["sigma_max_inds"] = inds
        if self.verbose:
            print(
                f"Location of maximum surface density (x,y) = ({x}, {y})")

    def sort_vortices_by_mass(self):
        """ Sort the vortices by mass decending. """
        self.vortices = [v for v in reversed(
            sorted(self.vortices, key=lambda v: v["contour"]["mass"]))]

    def print_properties(self, n=None):
        if n is None:
            n = len(self.candidates)
        for k, vort in enumerate(self.candidates.values()):
            if k >= n:
                break
            try:
                for v in ["mass", "vortensity_min", "vortensity_median", "vortensity_mean", "vortensity_max"]:
                    print(v, vort["contour"][v])
                strength = np.exp(-vort["vortensity_median"])*vort["mass"]
                print("strength", strength)
            except KeyError:
                pass
            print()

    def detect_vortex(self, include_mask=False, keep_internals=False):

        self.contour_image_dimensions()
        self.contour_image()
        self.find_contours()
        self.extract_closed_contours()
        self.extract_ellipse_contours()
        self.create_vortex_mask()

        self.calc_cell_masses()

        self.generate_ancestors()
        self.generate_decendents()
        self.prune_candidates_by_hierarchy()

        self.calculate_contour_properties()

        self.vortices = []
        for n, c in self.candidates.items():
            self.vortices.append({"contour": c})

        self.fit_contours()
        self.remove_non_vortex_candidates()
        self.remove_duplicates_by_min_vort_pos()

        self.sort_vortices_by_mass()

        if self.verbose:
            self.print_properties()

        self.remove_intermediate_data(include_mask, keep_internals)

        return self.vortices

    def show_contours(self):
        """ Plot the contour image. """
        _, ax = plt.subplots()
        ax.imshow(self.thresh)

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


def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)


def map_ext_pnt_to_orig(pnt, Nq):
    x = pnt[0]
    y = pnt[1]
    if y > Nq and y <= 3*Nq:
        y -= Nq
    elif y < Nq:
        y += Nq
    elif y > 3*Nq:
        y -= 3*Nq
    return (x, y)
