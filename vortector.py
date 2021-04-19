import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy.optimize import curve_fit


class Vortector:
    def __init__(self, Xc, Yc, A, vortensity, Sigma,
                 levels=[float(x) for x in np.arange(-1, 1.5, 0.05)],
                 med=0.15, mear=np.inf, mvd=0.01, verbose=False, azimuthal_boundaries=[-np.pi, np.pi]):

        self.vortensity = vortensity

        self.Sigma = Sigma

        self.Rc = Xc
        self.Xc = Xc
        self.Phic = Yc
        self.Yc = Yc
        self.cell_area = A

        self.azimuthal_boundaries = azimuthal_boundaries

        # Parameters
        self.levels = levels
        self.max_ellipse_deviation = med
        self.max_ellipse_aspect_ratio = mear
        self.min_vortensity_drop = mvd

        self.verbose = verbose

    def contour_image_dimensions(self):

        vortensity = self.vortensity
        self.Nx, self.Ny = vortensity.shape
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
        vort = self.vortensity
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
        self.mass = self.cell_area*self.Sigma

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
        for n, c in self.candidates.items():
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
        for n in no_min:
            del self.candidates[n]

    def remove_duplicates_by_min_vort_pos(self):
        """ Remove remaining duplicates.

        Because of the way the mirror counter lines are generated,
        the mirror shape can be shifted slightly.
        Still both mirror shapes should share the same location of minimum.
        The shape containing the lower mass is deleted.
        """
        to_del = []
        for c in self.candidates.values():
            inds = c["vortensity_min_inds"]
            mass = c["mass"]

            for key, o in self.candidates.items():
                o_inds = o["vortensity_min_inds"]
                o_mass = o["mass"]
                if o_inds == inds and o_mass < mass:
                    to_del.append(key)
        for key in set(to_del):
            del self.candidates[key]

    def calculate_vortex_properties(self):
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
                self.fit_gaussians(c)
            except (ValueError, RuntimeError) as e:
                # print("Warning: ValueError encountered in calculating vortex properties:", e)
                pass
            try:
                self.calc_fit_difference_2D(c)
            except KeyError as e:
                # print("Warning: KeyError encountered in calculating fit differences:", e)
                pass

    def fit_gaussians(self, vort):
        inds = vort["vortensity_min_inds"]

        def get_pos(inds):
            r = self.Xc[inds]
            phi = self.Yc[inds]
            return (r, phi)

        top = get_pos(vort["top"])
        left = get_pos(vort["left"])
        right = get_pos(vort["right"])
        bottom = get_pos(vort["bottom"])

        mask = vort["mask"]
        mask_r = mask[:, inds[1]]
        mask_phi = mask[inds[0], :]

        vals = self.Sigma
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
               "sigma_r": p[4], "sigma_phi": p[5], "popt": p, "pcov" : cov}
        save_fit(vort, "sigma", fit)

        # print("Sigma fit parameters")
        # for name, val, guess, low, up in zip(fitter.parameters, popt_rho, fitter.p0.values(), fitter.blow.values(), fitter.bup.values()):
        #     print(f"{name:5s} {val: .2e} ({guess: .2e}) [{low: .2e}, {up: .2e}]")

        vals = self.vortensity
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
               "sigma_r": p[4], "sigma_phi": p[5], "popt": p, "pcov" : cov}
        save_fit(vort, "vortensity", fit)

    def calc_fit_difference_2D(self, c, varname="sigma"):
        """ Calculate the difference of the fit to the data.

        Parameters
        ----------
        c : dict
            Vortex info.
        varname : str
            Name of the fit variable.
        """
        y0 = c[f"{varname}_fit_2D_c"]
        a = c[f"{varname}_fit_2D_a"]
        phi0 = c[f"{varname}_fit_2D_phi0"]
        sigma_phi = c[f"{varname}_fit_2D_sigma_phi"]
        r0 = c[f"{varname}_fit_2D_r0"]
        sigma_r = c[f"{varname}_fit_2D_sigma_r"]

        popt = c[f"{varname}_fit_2D_popt"]

        def f(r, phi):
            return gauss2D((r, phi), *popt)

        if varname == "sigma":
            vals = self.Sigma
        elif varname == "vortensity":
            vals = self.vortensity
        else:
            raise AttributeError(
                f"Can't calculate fit difference in 2D for '{varname}'!")

        R = self.Xc
        PHI = self.Yc
        hr = sigma_r*np.sqrt(2*np.log(2))
        hphi = sigma_phi*np.sqrt(2*np.log(2))

        mc = c["mask"]
        me = ((R-r0)/hr)**2 + ((PHI-phi0)/hphi)**2 <= 1

        Ae = np.sum(self.cell_area[me])
        Ac = c["area"]
        c[f"{varname}_fit_2D_ellipse_area_numerical"] = Ae
        c[f"{varname}_fit_2D_area_ratio_ellipse_to_contour"] = Ae/Ac

        for region, mask, area in zip(["contour", "ellipse"], [mc, me], [Ac, Ae]):
            fitvals = f(R[mask], PHI[mask])
            numvals = vals[mask]
            diff = np.sum(np.abs(fitvals - numvals))
            reldiff = diff/(area*a)
            c[f"{varname}_fit_2D_{region}_diff"] = diff
            c[f"{varname}_fit_2D_{region}_reldiff"] = reldiff
            c[f"{varname}_fit_2D_{region}_mass"] = np.sum(numvals*area)
            c[f"{varname}_fit_2D_{region}_mass_fit"] = np.sum(fitvals*area)


    def calc_vortex_mass(self, c):
        mask = c["mask"]
        c["mass"] = np.sum(self.mass[mask])

    def calc_vortensity(self, c):
        mask = c["mask"]
        c["vortensity_mean"] = np.mean(self.vortensity[mask])
        c["vortensity_median"] = np.median(self.vortensity[mask])
        c["vortensity_min"] = np.min(self.vortensity[mask])
        c["vortensity_max"] = np.max(self.vortensity[mask])

    def calc_sigma(self, c):
        mask = c["mask"]
        c["sigma_mean"] = np.mean(self.Sigma[mask])
        c["sigma_median"] = np.median(self.Sigma[mask])
        c["sigma_min"] = np.min(self.Sigma[mask])
        c["sigma_max"] = np.max(self.Sigma[mask])

    def calc_vortex_extent(self, c):
        mask = c["mask"]
        c["area"] = np.sum(self.cell_area[mask])
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
        A = self.cell_area
        c["vortensity_flux"] = np.sum((A*self.vortensity)[mask])
        c["vortensity_exp_flux"] = np.sum(
            (A*np.exp(-self.vortensity))[mask])

    def find_vortensity_min_position(self, contour):
        # Calculate the position of minimum vortensity
        mask = np.logical_not(contour["mask"])
        ind = np.argmin(np.ma.masked_array(
            self.vortensity, mask=mask), axis=None)
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
            self.Sigma, mask=mask), axis=None)
        inds = np.unravel_index(ind, mask.shape)
        x = self.Xc[inds]
        y = self.Yc[inds]
        contour["sigma_max_pos"] = (x, y)
        contour["sigma_max_inds"] = inds
        if self.verbose:
            print(
                f"Location of maximum surface density (x,y) = ({x}, {y})")

    def sort_vortices_by_mass(self):
        """ Sort the candidates by mass decending. """
        mass_sorted = [key for key in self.candidates]
        mass_sorted = sorted(
            mass_sorted, key=lambda n: self.candidates[n]["mass"])
        self.candidates = {n: self.candidates[n]
                           for n in reversed(mass_sorted)}

    def print_properties(self, n=None):
        if n is None:
            n = len(self.candidates)
        for k, vort in enumerate(self.candidates.values()):
            if k >= n:
                break
            try:
                for v in ["mass", "vortensity_min", "vortensity_median", "vortensity_mean", "vortensity_max"]:
                    print(v, vort[v])
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

        self.calculate_vortex_properties()
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
        self.vortices = []
        for c in self.candidates.values():
            if not keep_internals:
                for key in ["contour", "mask_extended", "bounding_hor",
                            "bounding_vert", "pixel_arcLength", "pixel_area",
                            "top_extended", "left_extended", "bottom_extended", "right_extended",
                            "ancestors", "decendents"]:
                    del c[key]
            if not include_mask:
                del c["mask"]
            self.vortices.append(c)

    def show_fit_overview_1D(self, n, axes=None):
        if axes is None:
            fig, axes = plt.subplots(2, 2, dpi=150, figsize=(
                8, 6), constrained_layout=True, sharex="col", sharey="row")
            axes = axes.flatten()
        else:
            if len(axes) != 4:
                raise ValueError(
                    "You need to pass a 1D array with 4 pyplot axes!")

        ax = axes[1]
        key = "vortensity"
        ref = "contour"
        self.show_radial_fit(ax, key, n, ref=ref)
        ax.set_title(f"rad, ref={ref}")
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("")

        ax = axes[0]
        self.show_azimuthal_fit(ax, key, n, ref=ref)
        ax.set_title(f"phi, ref={ref}")
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("")

        ax = axes[3]
        key = "sigma"
        ref = "sigma"
        center = "sigma"
        self.show_radial_fit(ax, key, n, ref=ref, center=center)

        ax = axes[2]
        self.show_azimuthal_fit(ax, key, n, ref=ref, center=center)

    def show_fit_overview_2D(self, n=0, axes=None, bnd_lines=False, bnd_pnts=False, show_fits=True, fit_contours=True):
        if axes is None:
            fig, axes = plt.subplots(2, 5, figsize=(10, 6), dpi=150,
                                     #  sharex="col",
                                     gridspec_kw={"height_ratios": [1, 4],
                                                  "width_ratios": [3, 1, 1, 3, 1]})
        else:
            if len(axes) != 8:
                raise ValueError(
                    "You need to pass an array with 2 pyplot axes!")
        plt.subplots_adjust(hspace=.001, wspace=0.001)

        self.show_fit_overview_2D_single("vortensity", ax=axes[1, 0],
                                         bnd_lines=bnd_lines, bnd_pnts=bnd_pnts,
                                         show_fits=show_fits, fit_contours=fit_contours,
                                         cbar_axes=[axes[1, 0], axes[1, 1]])

        ax = axes[0, 0]
        self.show_radial_fit(ax, "vortensity", 0, ref="contour")
        ax.set_ylim(-0.5, 1)
        ax.set_xticklabels([])
        ax.set_xlabel("")
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        xticks = ax.get_yticks()
        xticklabels = ax.get_yticklabels()

        ax = axes[1, 1]
        self.show_azimuthal_fit(ax, "vortensity", 0, ref="contour")
        switch_axes_xy(ax)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks], rotation=270)
        ax.set_ylim(-np.pi, np.pi)

        self.show_fit_overview_2D_single("sigma", ax=axes[1, 3],
                                         bnd_lines=bnd_lines, bnd_pnts=bnd_pnts,
                                         show_fits=show_fits, fit_contours=fit_contours,
                                         cbar_axes=[axes[1, 3], axes[1, 4]])
        ax = axes[0, 3]
        self.show_radial_fit(ax, "sigma", 0, ref="contour")
        ax.set_xticklabels([])
        ax.set_xlabel("")
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        xticks = ax.get_yticks()
        ax.set_ylim(bottom=0)

        ax = axes[1, 4]
        self.show_azimuthal_fit(ax, "sigma", 0, ref="contour")
        switch_axes_xy(ax)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1e}" for x in xticks],
                           rotation=270, fontsize=8)
        ax.set_xlim(left=0)
        ax.set_ylim(-np.pi, np.pi)

        for ax in [axes[0, 1], axes[0, 4], axes[0, 2], axes[1, 2]]:
            ax.axis("off")

        # fig.tight_layout()

    def show_fit_overview_2D_old(self, axes=None, bnd_lines=False, bnd_pnts=True, show_fits=True, fit_contours=True):
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150, gridspec_kw={
                                     "height_ratios": [1, 1, 2]})
        else:
            if len(axes) != 2:
                raise ValueError(
                    "You need to pass an array with 2 pyplot axes!")
        self.show_fit_overview_2D_single("vortensity", ax=axes[0],
                                         bnd_lines=bnd_lines, bnd_pnts=bnd_pnts, show_fits=show_fits, fit_contours=fit_contours)

        self.show_fit_overview_2D_single("sigma", ax=axes[1],
                                         bnd_lines=bnd_lines, bnd_pnts=bnd_pnts, show_fits=show_fits, fit_contours=fit_contours)

    def show_fit_overview_2D_single(self, varname, ax, bnd_lines=False,
                                    bnd_pnts=True, show_fits=True, fit_contours=True,
                                    cbar_axes=None):
        import matplotlib.patheffects as pe
        Xc = self.Xc
        Yc = self.Yc

        contour_colors = "darkgray"
        contour_lw = 0.5
        if varname == "vortensity":
            label = r"$\varpi/\varpi_0$"

            levels = self.levels

            Z = self.vortensity
            cmap = "magma"
            norm = colors.Normalize(vmin=levels[0], vmax=levels[-1])
            img = ax.pcolormesh(
                Xc, Yc, Z, cmap=cmap, norm=norm, rasterized=True, shading="auto")

            ax.contour(
                Xc, Yc, Z, levels=levels[::2], colors=contour_colors, linewidths=contour_lw)

        elif varname == "sigma":
            label = r"$\Sigma$"

            Z = self.Sigma
            cmap = "magma"

            try:
                vmax = self.vortices[0]["sigma_fit_2D_c"] + \
                    self.vortices[0]["sigma_fit_2D_a"]
            except (KeyError, IndexError):
                vmax = np.max(Z)

            vmin = min(1e-5*vmax, np.min(Z))
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            img = ax.pcolormesh(
                Xc, Yc, Z, cmap=cmap, norm=norm, rasterized=True, shading="auto")

            ax.contour(Xc, Yc, Z, levels=np.arange(
                0, vmax, vmax/20), colors=contour_colors, linewidths=contour_lw)
        else:
            raise ValueError(
                f"{varname} not supported. Only 'vortensity' and 'sigma'.")

        main_vortex = choose_main_vortex(self.vortices)

        vortices = [main_vortex] + \
            [v for v in self.vortices if v != main_vortex]

        for n, vort in enumerate(vortices):
            ax.contour(Xc, Yc, vort["mask"], levels=[
                0, 1, 2], linewidths=1, colors="white")
            x, y = vort["vortensity_min_pos"]
            if not show_fits:
                ax.plot([x], [y], "x")

            if bnd_lines:
                for key in ["rmin", "rmax"]:
                    ax.axvline(vort[key])
                for key in ["phimin", "phimax"]:
                    ax.axhline(vort[key])

            if bnd_pnts:
                for key in ["top", "bottom", "left", "right"]:
                    x = Xc[vort[key]]
                    y = Yc[vort[key]]
                    ax.plot([x], [y], "x")

            blow = -np.pi
            bup = np.pi
            L = bup - blow
            if show_fits:
                try:
                    lw = 1
                    path_effects = [
                        pe.Stroke(linewidth=2*lw, foreground='w'), pe.Normal()]

                    r0 = vort["vortensity_fit_2D_r0"]
                    sigma_r = vort["vortensity_fit_2D_sigma_r"]
                    w = 2*np.sqrt(2*np.log(2))*sigma_r
                    phi0 = vort["vortensity_fit_2D_phi0"]
                    sigma_phi = vort["vortensity_fit_2D_sigma_phi"]
                    phi0 = (phi0 - blow) % L + blow
                    h = 2*np.sqrt(2*np.log(2))*sigma_phi

                    color_vortensity = "C0"
                    lw = 1
                    if n == 0:
                        plot_ellipse_periodic(
                            ax, r0, phi0, w, h, crosshair=True, color=color_vortensity, ls="-", lw=lw, path_effects=path_effects)
                    else:
                        plot_ellipse_periodic(
                            ax, r0, phi0, w, h, color=color_vortensity, ls="-", lw=lw)

                    r0 = vort["sigma_fit_2D_r0"]
                    sigma_r = vort["sigma_fit_2D_sigma_r"]
                    w = 2*np.sqrt(2*np.log(2))*sigma_r
                    phi0 = vort["sigma_fit_2D_phi0"]
                    phi0 = (phi0 - blow) % L + blow
                    sigma_phi = vort["sigma_fit_2D_sigma_phi"]
                    h = 2*np.sqrt(2*np.log(2))*sigma_phi

                    if n == 0:
                        plot_ellipse_periodic(
                            ax, r0, phi0, w, h, color="C2", ls="-", lw=lw, crosshair=True, path_effects=path_effects)
                    else:
                        plot_ellipse_periodic(
                            ax, r0, phi0, w, h, color="C2", ls="-", lw=lw)
                except KeyError:
                    pass

        ax.set_xlabel(r"$r$ [au]")
        ax.set_ylabel(r"$\phi$")
        ax.set_yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi])
        ax.set_yticklabels(
            [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

        # ax.set_xlim(5.2, 10)

        ax.set_ylim(-np.pi, np.pi)

        if cbar_axes is None:
            cbar = ax.get_figure().colorbar(img, ax=ax, orientation="horizontal")
        else:
            cbar = ax.get_figure().colorbar(img, ax=cbar_axes, orientation="horizontal")
        cbar.set_label(label)

    def clamp_periodic(self, x):
        """ Make sure a periodic quantity is inside the domain.

        Parameters
        ----------
        x : float
            Position.

        Returns
        -------
        float
            Position moved into the boundary values.
        """
        bnd = self.azimuthal_boundaries
        rv = (x - bnd[0]) % (bnd[1]-bnd[0]) + bnd[0]
        return rv

    def select_fit_quantity(self, key):
        """ Return values by name.

        Parameters
        ----------
        key : str
            Name of the quantity.

        Returns
        -------
        array of floats
            Data array.
        """
        if key == "vortensity":
            return self.vortensity
        elif key == "sigma":
            return self.Sigma
        else:
            raise AttributeError(
                f"'{key}' is not a valid choice for a fit quantity.")

    def vortex_mask_r(self, c, hw):
        """ Construct a radial vortex mask from center and width value.

        Parameters
        ----------
        c : float
            Center position.
        hw : float
            Half of the width.

        Returns
        -------
        array of bool
            Mask indicating the vortex region.
        """
        wf = 1.5
        r = self.Xc[:, 0]
        mask = np.zeros(len(r), dtype=bool)
        cind = np.argmin(np.abs(r - c))
        lr = c - wf*hw
        lind = np.argmin(np.abs(r - lr))
        ur = c + wf*hw
        uind = np.argmin(np.abs(r - ur))
        mask = np.zeros(len(r), dtype=bool)
        mask[lind:uind+1] = True
        return mask

    def show_radial_fit(self, ax, key, n, ref="contour", center=None):
        """ Show a plot of a radial gaussian fit for the nth vortex.

        Parameters
        ----------
        ax : plt.axes
            Axes to plot on.
        key : str
            Name of the variable.
        n : int
            Number of the vortex in the sorted vortex array.
        ref : str
            Reference for the vortex region (contour/vortensity/sigma).
        """
        try:
            c = [c for c in self.vortices][n]
        except IndexError:
            print("No vortex found.")
            return

        try:
            center = ref if center is None else center
            inds = self.select_center_inds(c, center)
            mask_r, mask_phi = self.select_fit_region(c, ref)
            vals = self.select_fit_quantity(key)

            mask = mask_r

            y = vals[:, inds[1]]
            x = self.Xc[:, 0]

            ax.plot(x, y, label=f"data slice n={n}")
            ax.plot(x[mask], y[mask], label="vortex region")

            y0 = c[key + "_fit_2D_c"]
            x0 = c[key + "_fit_2D_r0"]
            a = c[key + "_fit_2D_a"]
            sig = c[key + "_fit_2D_sigma_r"]
            popt = [y0, a, x0, sig]

            ax.plot(x[mask], gauss(x[mask], *popt),
                    ls="--", color="C2", lw=2, label=f"fit")
            ax.plot(x, gauss(x, *popt), color="C3", alpha=0.3)
            ax.plot([x0], [y[inds[0]]], "x")
        except KeyError as e:
            print(f"Warning: KeyError encountered in showing r fit: {e}")
            return

        ax.set_xlabel(r"$r$")
        ax.set_ylabel(f"{key}")
        ax.legend()
        ax.grid()

    def show_azimuthal_fit(self, ax, key, n, ref="contour", center=None):
        """ Show a plot of a azimuthal gaussian fit for the nth vortex.

        Parameters
        ----------
        ax : plt.axes
            Axes to plot on.
        key : str
            Name of the variable.
        n : int
            Number of the vortex in the sorted vortex array.
        ref : str
            Reference for the vortex region (contour/vortensity/sigma).
        """
        try:
            c = [c for c in self.vortices][n]
        except IndexError:
            print("No vortex found.")
            return

        try:
            center = ref if center is None else center
            inds = self.select_center_inds(c, center)
            mask_r, mask_phi = self.select_fit_region(c, ref)
            vals = self.select_fit_quantity(key)
            mask = mask_phi

            y = vals[inds[0], :]
            x = self.Yc[0, :]

            ax.plot(x, y, label=f"data slice n={n}")
            plot_periodic(ax, x, y, mask, label="vortex region")
            y0 = c[key + "_fit_2D_c"]
            x0 = c[key + "_fit_2D_phi0"]
            # x0 = self.clamp_periodic(x0)
            a = c[key + "_fit_2D_a"]
            sig = c[key + "_fit_2D_sigma_phi"]

            bnd = self.azimuthal_boundaries
            L = bnd[1] - bnd[0]

            xc, yc = combine_periodic(x, y, mask)

            if x0 < np.min(xc):
                x0 += L
            if x0 > np.max(xc):
                x0 -= L
            popt = [y0, a, x0, sig]

            plot_periodic(ax, xc, gauss(xc, *popt), bnd=bnd,
                          ls="--", lw=2, color="C2", label=f"fit")

            xfull = np.linspace(x0-L/2, x0+L/2, endpoint=True)
            plot_periodic(ax, xfull, gauss(xfull, *popt), bnd=bnd,
                          ls="-", lw=1, color="C3", alpha=0.3)
            ax.plot([self.clamp_periodic(x0)], y[[inds[1]]], "x")
        except KeyError as e:
            print(f"Warning: KeyError encountered in showing phi fit: {e}")
            return

        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(f"{key}")
        ax.set_xticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        ax.legend()
        ax.grid()

    def vortex_mask_phi(self, c, hw):
        """ Construct a azimuthal vortex mask from center and width value.

        This function takes the periodic boundary into account.

        Parameters
        ----------
        c : float
            Center position.
        hw : float
            Half of the width.

        Returns
        -------
        array of bool
            Mask indicating the vortex region.
        """
        wf = 1.5
        phi = self.Yc[0, :]
        mask = np.zeros(len(phi), dtype=bool)
        cind = np.argmin(np.abs(phi - c))
        lphi = self.clamp_periodic(c - wf*hw)
        lind = np.argmin(np.abs(phi - lphi))
        uphi = self.clamp_periodic(c + wf*hw)
        uind = np.argmin(np.abs(phi - uphi))
        mask = np.zeros(len(phi), dtype=bool)
        bnd = self.azimuthal_boundaries
        if c + wf*hw > bnd[1] or c - wf*hw < bnd[0]:
            mask[lind:] = True
            mask[:uind+1] = True
        else:
            mask[lind:uind+1] = True
        return mask

    def select_center_inds(self, c, ref):
        """ Select the indices which indicate the fit region.

        The source which specifies the vortex region is either
        the extracted contour or the fits of vortensity or surface density.
        ref = contour, vortensity, sigma


        Parameters
        ----------
        c : dict
            Vortex candidate.
        ref : str
            Name of the variable to take the mask from.

        Returns
        -------
        inds : tuple of two int
            Radial and azimuthal index of center.
        """
        if ref == "contour":
            r0, phi0 = c["vortensity_min_pos"]
        elif ref == "vortensity":
            r0 = c["vortensity_fit_2D_r0"]
            phi0 = c["vortensity_fit_2D_phi0"]
        elif ref == "sigma":
            r0 = c["sigma_fit_2D_r0"]
            phi0 = c["sigma_fit_2D_phi0"]
        else:
            raise AttributeError(
                f"'{ref}' is not a valid reference for fitting.")
        rind = position_index(self.Xc[:, 0], r0)
        phiind = position_index(self.Yc[0, :], phi0)
        inds = (rind, phiind)
        return inds

    def select_fit_region(self, c, ref):
        """ Select the indices and masks which indicate the fit region.

        The source which specifies the vortex region is either
        the extracted contour or the fits of vortensity or surface density.
        ref = contour, vortensity, sigma


        Parameters
        ----------
        c : dict
            Vortex candidate.
        ref : str
            Name of the variable to take the mask from.

        Returns
        -------
        inds : tuple of two int
            Radial and azimuthal index of center.
        mask_r : array of bool
            Mask indicating vortex region in radial direction.
        mask_phi : array of bool
            Mask indicating vortex region in azimuthal direction.
        """
        inds = self.select_center_inds(c, ref)
        if ref == "contour":
            mask = c["mask"]
            mask_r = mask[:, inds[1]]
            mask_phi = mask[inds[0], :]
        elif ref == "vortensity":
            center = c["vortensity_fit_2D_r0"]
            hw = c["vortensity_fit_2D_sigma_r"]
            mask_r = self.vortex_mask_r(center, hw)
            center = c["vortensity_fit_2D_phi0"]
            hw = c = c["vortensity_fit_2D_sigma_phi"]
            mask_phi = self.vortex_mask_phi(center, hw)
        elif ref == "sigma":
            center = c["sigma_fit_2D_r0"]
            hw = c["sigma_fit_2D_sigma_r"]
            mask_r = self.vortex_mask_r(center, hw)
            center = c["sigma_fit_2D_r0"]
            hw = c["sigma_fit_2D_sigma_phi"]
            mask_phi = self.vortex_mask_phi(center, hw)
        else:
            raise AttributeError(
                f"'{ref}' is not a valid reference for fitting.")
        return mask_r, mask_phi


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


def position_index(x, x0):
    """ Index of position x0 in array x .

    Parameters
    ----------
    x : array
        Position array.
    x0 : float
        Position of interest.

    Returns
    -------
    int
        Index of position x0.
    """
    return int(np.argmin(np.abs(x-x0)))


def choose_main_vortex(vortices):
    """ Choose a vortex from a list of candidates. """
    if len(vortices) == 0:
        return dict({})
    if len(vortices) == 1:
        return vortices[0]

    large_vortices = [vortices[0]]
    ref_mass = vortices[0]["mass"]
    # keep vortices that have 20% of most massive's mass
    for vortex in vortices[1:]:
        if vortex["mass"] > ref_mass/5:
            large_vortices.append(vortex)

    vortices_with_fit = []
    for vortex in large_vortices:
        if "sigma_fit_2D_c" in vortex:
            vortices_with_fit.append(vortex)

    if len(vortices_with_fit) > 0:
        return vortices_with_fit[0]

    sorted_vortices = sorted(large_vortices, key=lambda x: x["vortensity_min"])
    return sorted_vortices[0]


def combine_periodic(x, y, m, bnd=(-np.pi, np.pi)):
    """ Combine an array split at a periodic boundary. 

    This is used for vortices that stretch over the periodic boundary
    in azimuthal directions in disks.
    The x values at the left boundary are appended to the right of the
    values at the right boundary and the periodicity is added.

    The following sketch shows how the arrays are combined.

    |+++++_________xxx|
    to
    |______________xxx|+++++

    Parameters
    ----------
    x: array
        Coordinate values (full domain).
    y: array
        Values (full domain).
    m: array (boolean)
        Mask to select the active values.
    bnd: tuple of two float
        Position of the left and right boundary.

    Returns
    -------
    x : array
        coordinates
    y : array
        values
    """
    lb, rb = bnd
    if m[0] and m[-1] and not all(m):
        b = np.where(m == False)[0][0]
        xl = x[:b]
        yl = y[:b]
        b = np.where(m == False)[0][-1]
        xr = x[b:]
        yr = y[b:]
        xcom = np.append(xr, xl+(rb-lb))
        ycom = np.append(yr, yl)
        return xcom, ycom
    else:
        return x[m], y[m]


def save_fit(c, varname, fit, axis=None, parameters=["y0", "a", "x0", "sigma"]):
    """ Save fit variables to a vortex candidate dict.

    Paramters
    ---------
    c : dict
        Vortex data dict.
    varname : str
        Fit variable name.
    axis : str
        Name of the direction (e.g. r or phi).
    fit : dict
        Dict containing popt vector and aux info.
    """
    for key, val in fit.items():
        c[f"{varname}_fit_2D_{key}"] = val


def plot_periodic_mask(ax, x, y, m, **kwargs):
    #     print(m)
    if m[0] and m[-1] and not all(m):
        bnd = np.where(m == False)[0][0]
        xl = x[:bnd]
        yl = y[:bnd]
        line, = ax.plot(xl, yl, **kwargs)
        bnd = np.where(m == False)[0][-1]
#         print(len(x))
#         print(bnd)
        xr = x[bnd:]
        yr = y[bnd:]
        kwa = kwargs.copy()
        kwa["color"] = line.get_color()
        kwa["ls"] = line.get_linestyle()
        line, = ax.plot(xr, yr, **kwa)
        line.set_label(None)
    else:
        ax.plot(x[m], y[m], **kwargs)


def plot_periodic(ax, x, y, m=None, bnd=(-np.pi, np.pi), **kwargs):
    if m is not None:
        plot_periodic_mask(ax, x, y, m, **kwargs)
        return
    L = bnd[1] - bnd[0]
    m = np.logical_and(x <= bnd[1], x >= bnd[0])
    line, = ax.plot(x[m], y[m], **kwargs)
    kwa = kwargs.copy()
    kwa["color"] = line.get_color()
    kwa["ls"] = line.get_linestyle()

    m = x < bnd[0]
    line, = ax.plot(x[m] + L, y[m], **kwa)
    line.set_label(None)

    m = x > bnd[1]
    line, = ax.plot(x[m] - L, y[m], **kwa)
    line.set_label(None)


def plot_vline_periodic(ax, x, y, dy, **kwargs):
    bup = np.pi
    blow = -np.pi
    L = bup - blow
    y = (y-blow) % L + blow

    at_upper_bnd = y + dy > bup
    at_lower_bnd = y - dy < blow
    if at_upper_bnd:
        line, = ax.plot([x, x], [y-dy, bup], **kwargs)
        c = line.get_color()
        ls = line.get_linestyle()
        lw = line.get_linewidth()
        ax.plot([x, x], [blow, y-L+dy], ls=ls, lw=lw, c=c)
    elif at_lower_bnd:
        line, = ax.plot([x, x], [y+dy, blow], **kwargs)
        c = line.get_color()
        ls = line.get_linestyle()
        lw = line.get_linewidth()
        ax.plot([x, x], [bup, y+L-dy], ls=ls, lw=lw, c=c)
    else:
        ax.plot([x, x], [y-dy, y+dy], **kwargs)


def switch_axes_xy(ax):
    """ Switch the x and y axis of an axes. """
    lines = ax.get_lines()
    for line in lines:
        x = line.get_xdata()
        y = line.get_ydata()
        line.set_xdata(y)
        line.set_ydata(x)

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.set_xticks(yticks)
    ax.set_xticklabels(yticklabels)
    ax.set_yticks(xticks)
    ax.set_yticklabels(xticklabels)
    ax.set_xlim(ylim)
    ax.set_ylim(xlim)

    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    ax.set_xlabel(ylabel)
    ax.set_ylabel(xlabel)


def plot_ellipse_periodic(ax, x, y, w, h, crosshair=False, bnd=(-np.pi, np.pi), **kwargs):
    """ Show an Ellipse in a plot periodic in y direction.

    Parameters
    ----------
    ax : plt.axes
        Axes to plot on.
    x : float
        Center in x.
    y : float
        Center in y.
    w : float
        Width in x.
    h : float
        Height in y.
    crosshair : bool
        Show a cross indicating the center.
    bnd : (float, float)
        Locations of the periodic boundary.
    """
    from matplotlib.patches import Ellipse
    L = bnd[1] - bnd[0]
    C = 0.5*(bnd[1] + bnd[0])

    plot_args = kwargs.copy()
    if "color" in kwargs:
        plot_args["edgecolor"] = kwargs["color"]
        del plot_args["color"]
    else:
        kwargs["edgecolor"] = "C0"
    e = Ellipse(xy=[x, y], width=w, height=h,
                fc="None", **kwargs)
    ax.add_artist(e)
    e.set_zorder(1000)
    e.set_clip_box(ax.bbox)
    y_clone = y+L*(1 if y < C else -1)
    e = Ellipse(xy=[x, y_clone], width=w,
                height=h, angle=0, fc="None", **kwargs)
    ax.add_artist(e)
    e.set_zorder(1000)
    e.set_clip_box(ax.bbox)

    lw = e.get_linewidth()

    if crosshair:
        line_args = kwargs.copy()
        for k in ["lw", "linewidth"]:
            if k in line_args:
                del line_args[k]
        line_args["lw"] = lw/2
        if "path_effects" in line_args:
            del line_args["path_effects"]
        ax.plot([x + w/2, x - w/2],
                [y, y], **line_args)
        plot_vline_periodic(
            ax, x, y, h/2, **line_args)


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
