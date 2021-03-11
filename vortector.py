import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


class Vortector:
    def __init__(self, Xc, Yc, A, vortensity, Sigma, Sigma0,
                 Rlims, levels=[float(x) for x in np.arange(-1, 1.5, 0.05)],
                 med=0.15, mear=np.inf, mvd=0.01, verbose=False, azimuthal_boundaries = [-np.pi, np.pi]):

        self.vortensity = vortensity

        self.Sigma = Sigma
        self.Sigma_background = Sigma0

        self.Rc = Xc
        self.Phic = Yc
        self.cell_area = A

        self.azimuthal_boundaries = azimuthal_boundaries

        # Parameters
        self.Rlims = Rlims
        self.levels = levels
        self.max_ellipse_deviation = med
        self.max_ellipse_aspect_ratio = mear
        self.min_vortensity_drop = mvd

        self.verbose = verbose

    def contour_image_dimensions(self):

        data_view = self.vortensity[self.vmi:self.vma, :]
        self.Nx, self.Ny = data_view.shape
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
        vort = self.vortensity[self.vmi:self.vma, :]
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

        contours_dict = {n: {"cnt": cnt, "n": n}
                         for n, cnt in enumerate(contours)}

        areas = [cv2.contourArea(c) for c in contours]
        for n, d in enumerate(contours_dict.values()):
            d["area"] = areas[n]

        sort_inds = np.argsort(areas)

        # take the up to 100 largest patches
        self.contours_largest = [contours_dict[i]
                                 for i in [n for n in sort_inds[::-1]][:100]]

    def extract_closed_contours(self):
        # Extract closed contours

        self.contours_closed = []
        for n, contour in enumerate(self.contours_largest):
            cnt = contour["cnt"]
            l = cv2.arcLength(cnt, True)
            contour["arcLength"] = l
            a = contour["area"]
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
            print("Area of contours:", [c["area"]
                                        for c in self.contours_closed])

    def extract_ellipse_contours(self):
        # Extract contours that match an ellipse

        self.candidates = {}
        for contour in self.contours_closed:
            cnt = contour["cnt"]
            ellipse = cv2.fitEllipse(cnt)

            im_shape = np.zeros(self.thresh.shape)
            cv2.drawContours(im_shape, [cnt], 0, (255, 255, 255), -1)

            im_ellipse = np.zeros(self.thresh.shape)
            im_ellipse = cv2.ellipse(im_ellipse, ellipse, (255, 255, 255), -1)

            difference = np.abs(im_shape - im_ellipse)
            difference_area = np.sum(difference/255)

            rel_delta = difference_area / contour["area"]

            if rel_delta > self.max_ellipse_deviation:
                continue

            contour["mask_extended"] = im_shape
            self.candidates[contour["n"]] = contour

    def create_vortex_mask_in_view(self):
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
            contour["mask_view"] = mask

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
                contour[key + "_view"] = (x, y)
                contour[key] = (x + self.vmi, y)

        if self.verbose:
            print(
                f"Mapping mask: mask.shape = {mask.shape}, mask_orig.shape {mask_orig.shape}")

    def create_array_views(self):
        """ Create arrays that match the extend defined by Rlims which matches the contour image. """
        self.Xc_view = self.Rc[self.vmi:self.vma, :]
        self.Yc_view = self.Phic[self.vmi:self.vma, :]
        self.cell_area_view = self.cell_area[self.vmi:self.vma, :]
        self.Rho_view = self.Sigma[self.vmi:self.vma, :]
        self.Rho_background_view = self.Sigma_background[self.vmi:self.vma, :]
        self.vortensity_view = self.vortensity[self.vmi:self.vma, :]

    def calc_cell_masses(self):
        self.mass_view = self.cell_area_view*self.Rho_view
        self.mass_background_view = self.cell_area_view*self.Rho_background_view

    def generate_ancestors(self):
        # Generate ancestor list

        # The hierarchy generated by opencv in the contour finder outputs a list with the syntax
        # ```
        # [Next, Previous, First_Child, Parent]
        # ```
        # If any of those is not available its encoded  by -1.

        for c in self.candidates.values():
            ancestors = []
            n_parent = c["n"]
            for n in range(1000):
                n_parent = self.hierarchy[0, n_parent, 3]
                if n_parent == -1 or n_parent not in self.candidates:
                    break
                ancestors.append(n_parent)
            c["ancestors"] = ancestors
            if self.verbose:
                print("Ancestors:", c["n"], ancestors)

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
            if c["n"] in decendents:
                dec = decendents[c["n"]]
            else:
                dec = []
            c["decendents"] = dec
            if self.verbose:
                print("Descendents:", c["n"], dec)

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
            cid = c["n"]
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
            except ValueError as e:
                print(e)

    def fit_gaussians(self, c):
        try:
            fit_vort_phi = self.fit_gaussian_phi(
                c, "vortensity", ref="contour", fix_avg=True)
            save_fit(c, "vortensity", "phi", fit_vort_phi)
            fit_vort_r = self.fit_gaussian_r(c, "vortensity", ref="contour", fixed={
                "y0": fit_vort_phi["popt"][0], "a": fit_vort_phi["popt"][1]})
            save_fit(c, "vortensity", "r", fit_vort_r)

            ref = "vortensity"
            center = "vortensity"
        except RuntimeError:
            ref = "contour"
            center = "contour"

        ymax = c["sigma_max"]
        yavg = c["sigma_mean"]

        inds = self.select_center_inds(c, center)
        mask_r, mask_phi = self.select_fit_region(c, ref)
        y = self.Rho_view[inds[0], :]
        y_outer = y[np.logical_not(mask_phi)]
        ymax_outer = np.max(y_outer)
        yavg_outer = np.average(y_outer)
        ymin_outer = np.min(y_outer)

        dy = ymax - yavg_outer

        fit_phi = self.fit_gaussian_phi(
            c, "sigma", autoweight=True,
            ref=ref, center=center,
            blow={"y0": 0.75*yavg_outer, "a": 0},
            bup={"y0": 1.25*yavg_outer},
            p0={"y0": yavg_outer, "a": dy}
        )
        save_fit(c, "sigma", "phi", fit_phi)

        y0_phi = fit_phi["popt"][0]
        a_phi = fit_phi["popt"][1]
        fit_r = self.fit_gaussian_r(
            c, "sigma", autoweight=True,
            ref=ref, center=center,
            blow={"y0": 0.75*yavg_outer, "a": 0},
            bup={"y0": 1.25*yavg_outer},
            p0={"y0": y0_phi, "a": a_phi},
            fixed={"y0": y0_phi}
        )
        save_fit(c, "sigma", "r", fit_r)

        for n in range(5):
            popt = fit_phi["popt"]
            lf = 0.9
            uf = 1.1
            fit_phi_new = self.fit_gaussian_phi(
                c, "sigma", autoweight=True,
                ref="vortensity", center="sigma",
                blow={"y0": lf*popt[0], "a": lf*popt[1], "sigma": lf*popt[3]},
                bup={"y0": uf*popt[0], "a": uf*popt[1], "sigma": uf*popt[3]},
                p0={"y0": yavg_outer, "a": popt[1],
                    "x0": popt[2], "sigma": popt[3]},
                # fixed={"y0" : popt[0]}
                # fixed = {"y0": popt[0], "a": popt[1],"sigma" : popt[3]},
            )
            if fit_phi_new["diff"] < fit_phi["diff"]:
                if self.verbose:
                    print(
                        f"- fitting sigma to n = {c['n']}: accepted fit {n+2} in phi")
                fit_phi = fit_phi_new
                save_fit(c, "sigma", "phi", fit_phi)
                fit_phi_stopped = False
            else:
                fit_phi_stopped = True

            y0_phi = fit_phi["popt"][0]
            a_phi = fit_phi["popt"][1]
            fit_r_new = self.fit_gaussian_r(
                c, "sigma", autoweight=True,
                ref="sigma", center="sigma",
                blow={"y0": 0.75*yavg_outer, "a": 0},
                bup={"y0": 1.25*yavg_outer},
                p0={"y0": y0_phi, "a": a_phi},
                fixed={"y0": y0_phi, "a": a_phi})

            if fit_r_new["diff"] < fit_r["diff"]:
                if self.verbose:
                    print(f"- fitting sigma: accepted fit {n+2} in r")
                fit_r = fit_r_new
                save_fit(c, "sigma", "r", fit_r)
                fit_r_stopped = False
            else:
                fit_r_stopped = True

            if fit_r_stopped and fit_phi_stopped:
                if self.verbose:
                    print(
                        f"- fitting sigma to n = {c['n']}: finished after {n+2} attempt")
                break
        
        self.calc_fit_difference_2D(c)
    
    def calc_fit_difference_2D(self, c, varname="sigma"):
        """ Calculate the difference of the fit to the data.
        
        Parameters
        ----------
        c : dict
            Vortex info.
        varname : str
            Name of the fit variable.
        """
        pre = f"{varname}_fit"
        y0 = c[f"{pre}_phi_y0"]
        a = c[f"{pre}_phi_a"]
        phi0 = c[f"{pre}_phi_x0"]
        sigma_phi = c[f"{pre}_phi_sigma"]
        r0 = c[f"{pre}_r_x0"]
        sigma_r = c[f"{pre}_r_sigma"]
        def f(r,phi):
            er = np.exp(-(r-r0)**2/(2*sigma_r**2))
            ephi = np.exp(-(phi-phi0)**2/(2*sigma_phi**2))
            return y0 + a*er*ephi
        
        if varname == "sigma":
            vals = self.Rho_view
        elif varname == "vortensity":
            vals = self.vortensity_view
        else:
            raise AttributeError(f"Can't calculate fit difference in 2D for '{varname}'!")
        
        R = self.Xc_view
        PHI = self.Yc_view
        
        mc = c["mask_view"]
        me = ((R-r0)/sigma_r)**2 + ((PHI-phi0)/sigma_phi)**2 <= 1
        
        Ae = np.sum(self.cell_area_view[me])
        Ac = c["area"]
        c[f"{pre}_ellipse_area_numerical"] = Ae
        c[f"{pre}_area_ratio_ellipse_to_contour"] = Ae/Ac
        
        for region, mask, area in zip(["contour", "ellipse"], [mc, me], [Ac, Ae]):
            fitvals = f(R[mask], PHI[mask])
            numvals = vals[mask]
            diff = np.sum(np.abs(fitvals - numvals))
            reldiff = diff/(area*a)
            c[f"{pre}_{region}_diff_2D"] = diff
            c[f"{pre}_{region}_reldiff_2D"] = reldiff
            

    def fit_gaussian_phi(self, c, key, ref="contour", center=None, fixed=None, blow=None, bup=None, p0=None, fix_avg=False, autoweight=True):
        """ Fit a gaussian in phi direction.

        Fit a region in r direction either specified by the mask
        generated from the contour or from a previous fit.
        This is controlled with the ref parameter which can be:
        contour, vortensity, sigma

        Parameters
        ----------
        c : dict
            Vortex candidate.
        key : str
            Name of the variable for fitting.
        ref : str
            Name of the variable to take the mask from.
        """
        fixed = fixed if fixed is not None else {}
        blow = blow if blow is not None else {}
        bup = bup if bup is not None else {}
        p0 = p0 if p0 is not None else {}

        if center is None:
            center = ref
        inds = self.select_center_inds(c, center)
        mask_r, mask_phi = self.select_fit_region(c, ref)
        vals = self.select_fit_quantity(key)

        vals_phi = vals[inds[0], :]

        phi = self.Yc_view[inds[0], :]

        x, y = combine_periodic(phi, vals_phi, mask_phi,
                                bnd=self.azimuthal_boundaries)

        if fix_avg:
            fixed["y0"] = np.average(vals_phi)

        if "sigma" not in bup:
            bup["sigma"] = (np.max(x) - np.min(x))

        if "sigma" not in blow:
            blow["sigma"] = (np.max(x) - np.min(x))/10

        fitter = GaussFitter(x, y, autoweight=autoweight,
                             fixed=fixed, blow=blow, bup=bup, p0=p0,
                             verbose=self.verbose,
                             name=f"{key} azimuthal n={c['n']}")
        popt, pcov = fitter.fit()

        diff = np.sum(np.abs(y - gauss(x, *popt)))
        reldiff = diff / np.sum(y - np.min(y))

        # fig, ax = plt.subplots()
        # ax.plot(x, y)
        # ax.plot(x, gauss(x, *popt), "--")
        # ax.set_title(
        #     fitter.name + f" reldiff={reldiff:.2e}, diff={diff:.2e}, r = {self.Xc_view[inds]:.2e} phi = {self.Yc_view[inds]:.2e}")

        fit = {
            "popt": popt,
            "inds": inds,
            "reldiff": reldiff,
            "diff": diff
        }

        return fit

    def fit_gaussian_r(self, c, key, ref="contour", center=None, fixed=None, blow=None, bup=None, p0=None, autoweight=True):
        """ Fit a gaussian in r direction.

        Fit a region in r direction either specified by the mask
        generated from the contour or from a previous fit.
        This is controlled with the ref parameter which can be:
        contour, vortensity, sigma

        Parameters
        ----------
        c : dict
            Vortex candidate.
        key : str
            Name of the variable for fitting.
        ref : str
            Name of the variable to take the mask from.
        """
        fixed = fixed if fixed is not None else {}
        blow = blow if blow is not None else {}
        bup = bup if bup is not None else {}
        p0 = p0 if p0 is not None else {}

        center = ref if center is None else center
        inds = self.select_center_inds(c, center)
        mask_r, mask_phi = self.select_fit_region(c, ref)
        vals = self.select_fit_quantity(key)

        vals_r = vals[:, inds[1]]
        vals_phi = vals[inds[0], :]

        r = self.Xc_view[:, inds[1]]
        phi = self.Yc_view[inds[0], :]

        x = r[mask_r]
        y = vals_r[mask_r]

        if "sigma" not in bup:
            bup["sigma"] = (np.max(x) - np.min(x))

        if "sigma" not in blow:
            blow["sigma"] = (np.max(x) - np.min(x))/10

        fitter = GaussFitter(x, y, autoweight=autoweight,
                             fixed=fixed, blow=blow, bup=bup, p0=p0,
                             verbose=self.verbose,
                             name=f"{key} radial n={c['n']}")
        popt, pcov = fitter.fit()

        diff = np.average(np.abs(y - gauss(x, *popt)))
        reldiff = diff / (np.max(y) - np.min(y))

        fit = {
            "popt": popt,
            "inds": inds,
            "reldiff": reldiff,
            "diff": diff
        }

        # fig, ax = plt.subplots()
        # ax.plot(x, y)
        # ax.plot(x, gauss(x, *popt), "--")
        # ax.set_title(
        #     fitter.name + f" reldiff={reldiff:.2e}, diff={diff:.2e}, r = {self.Xc_view[inds]:.2e} phi = {self.Yc_view[inds]:.2e}")

        return fit

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
            return self.vortensity_view
        elif key == "sigma":
            return self.Rho_view
        elif key == "vorticity":
            return self.vortensity_view
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
        r = self.Xc_view[:, 0]
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
        except KeyError:
            print("No vortex found.")
            return

        try:
            center = ref if center is None else center
            inds = self.select_center_inds(c, center)
            inds = c[f"{key}_fit_r_inds"]
            mask_r, mask_phi = self.select_fit_region(c, ref)
            vals = self.select_fit_quantity(key)

            mask = mask_r

            y = vals[:, inds[1]]
            x = self.Xc_view[:, 0]

            ax.plot(x, y, label=f"data slice n={c['n']}")
            ax.plot(x[mask], y[mask], label="vortex region")

            y0 = c[key + "_fit_r_y0"]
            x0 = c[key + "_fit_r_x0"]
            a = c[key + "_fit_r_a"]
            sig = c[key + "_fit_r_sigma"]
            popt = [y0, a, x0, sig]
            reldiff = c[key+"_fit_r_reldiff"]

            ax.plot(x[mask], gauss(x[mask], *popt),
                    ls="--", color="C2", lw=2, label=f"fit, reldiff={reldiff:.2e}")
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
        except KeyError:
            print("No vortex found.")
            return

        try:
            center = ref if center is None else center
            inds = self.select_center_inds(c, center)
            inds = c[f"{key}_fit_phi_inds"]
            mask_r, mask_phi = self.select_fit_region(c, ref)
            vals = self.select_fit_quantity(key)
            mask = mask_phi

            y = vals[inds[0], :]
            x = self.Yc_view[0, :]

            ax.plot(x, y, label=f"data slice n={c['n']}")
            plot_periodic(ax, x, y, mask, label="vortex region")
            y0 = c[key + "_fit_phi_y0"]
            x0 = c[key + "_fit_phi_x0"]
            # x0 = self.clamp_periodic(x0)
            a = c[key + "_fit_phi_a"]
            sig = c[key + "_fit_phi_sigma"]
            reldiff = c[key + "_fit_phi_reldiff"]

            bnd = self.azimuthal_boundaries
            L = bnd[1] - bnd[0]

            xc, yc = combine_periodic(x, y, mask)

            if x0 < np.min(xc):
                x0 += L
            if x0 > np.max(xc):
                x0 -= L
            popt = [y0, a, x0, sig]

            plot_periodic(ax, xc, gauss(xc, *popt), bnd=bnd,
                          ls="--", lw=2, color="C2", label=f"fit reldiff={reldiff:.2e} r={self.Xc_view[inds[0],0]:.2e}")

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
        phi = self.Yc_view[0, :]
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
            r0 = c["vortensity_fit_r_x0"]
            phi0 = c["vortensity_fit_phi_x0"]
        elif ref == "sigma":
            r0 = c["sigma_fit_r_x0"]
            phi0 = c["sigma_fit_phi_x0"]
        else:
            raise AttributeError(
                f"'{ref}' is not a valid reference for fitting.")
        rind = position_index(self.Xc_view[:, 0], r0)
        phiind = position_index(self.Yc_view[0, :], phi0)
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
            mask = c["mask_view"]
            mask_r = mask[:, inds[1]]
            mask_phi = mask[inds[0], :]
        elif ref == "vortensity":
            center = c["vortensity_fit_r_x0"]
            hw = c["vortensity_fit_r_sigma"]
            mask_r = self.vortex_mask_r(center, hw)
            center = c["vortensity_fit_phi_x0"]
            hw = c = c["vortensity_fit_phi_sigma"]
            mask_phi = self.vortex_mask_phi(center, hw)
        elif ref == "sigma":
            center = c["sigma_fit_r_x0"]
            hw = c["sigma_fit_r_sigma"]
            mask_r = self.vortex_mask_r(center, hw)
            center = c["sigma_fit_phi_x0"]
            hw = c["sigma_fit_phi_sigma"]
            mask_phi = self.vortex_mask_phi(center, hw)
        else:
            raise AttributeError(
                f"'{ref}' is not a valid reference for fitting.")
        return mask_r, mask_phi

    def calc_vortex_mass(self, c):
        mask = c["mask_view"]
        c["mass"] = np.sum(self.mass_view[mask])
        c["mass_background"] = np.sum(self.mass_background_view[mask])
        c["mass_enhancement"] = c["mass"] - c["mass_background"]

    def calc_vortensity(self, c):
        mask = c["mask_view"]
        c["vortensity_mean"] = np.mean(self.vortensity_view[mask])
        c["vortensity_median"] = np.median(self.vortensity_view[mask])
        c["vortensity_min"] = np.min(self.vortensity_view[mask])
        c["vortensity_max"] = np.max(self.vortensity_view[mask])

    def calc_sigma(self, c):
        mask = c["mask_view"]
        c["sigma_mean"] = np.mean(self.Rho_view[mask])
        c["sigma_median"] = np.median(self.Rho_view[mask])
        c["sigma_min"] = np.min(self.Rho_view[mask])
        c["sigma_max"] = np.max(self.Rho_view[mask])
        c["sigma0_mean"] = np.mean(self.Rho_background_view[mask])
        c["sigma0_median"] = np.median(self.Rho_background_view[mask])
        c["sigma0_min"] = np.min(self.Rho_background_view[mask])
        c["sigma0_max"] = np.max(self.Rho_background_view[mask])

    def calc_vortex_extent(self, c):
        mask = c["mask_view"]
        c["area"] = np.sum(self.cell_area_view[mask])
        c["rmax"] = self.Xc_view[c["left_view"]]
        c["rmin"] = self.Xc_view[c["right_view"]]
        c["phimin"] = self.Yc_view[c["top_view"]]
        c["phimax"] = self.Yc_view[c["bottom_view"]]
        if c["phimax"] < c["phimin"]:
            c["height"] = c["phimax"] + 2*np.pi - c["phimin"]
        else:
            c["height"] = c["phimax"] - c["phimin"]

    def calc_vortensity_flux(self, c):
        mask = c["mask_view"]
        A = self.cell_area_view
        c["vortensity_flux"] = np.sum((A*self.vortensity_view)[mask])
        c["vortensity_exp_flux"] = np.sum(
            (A*np.exp(-self.vortensity_view))[mask])

    def find_vortensity_min_position(self, contour):
        # Calculate the position of minimum vortensity
        mask = np.logical_not(contour["mask_view"])
        ind = np.argmin(np.ma.masked_array(
            self.vortensity_view, mask=mask), axis=None)
        inds = np.unravel_index(ind, mask.shape)
        x = self.Xc_view[inds]
        y = self.Yc_view[inds]
        contour["vortensity_min_pos"] = (x, y)
        contour["vortensity_min_inds"] = inds
        if self.verbose:
            print(f"Location of minimum vortensity: (x,y) = ({x}, {y})")

    def find_density_max_position(self, contour):
        # Calculate the position of maximum density
        mask = np.logical_not(contour["mask_view"])
        ind = np.argmax(np.ma.masked_array(
            self.Rho_view, mask=mask), axis=None)
        inds = np.unravel_index(ind, mask.shape)
        x = self.Xc_view[inds]
        y = self.Yc_view[inds]
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
                for v in ["mass_enhancement", "mass_background", "mass", "vortensity_min", "vortensity_median", "vortensity_mean", "vortensity_max"]:
                    print(v, vort[v])
                strength = np.exp(-vort["vortensity_median"])*vort["mass"]
                print("strength", strength)
            except KeyError:
                pass
            print()

    def construct_global_mask(self, v):
        """ Construct a mask array indicating the vortex with True.

        This mask fits the initial dimensions of the data arrays.
        """
        mask = v["mask_view"]
        gmask = np.concatenate(
            [np.zeros((self.vmi, mask.shape[1]), dtype=bool),
             mask,
             np.zeros((self.Rc.shape[0]-self.vma, mask.shape[1]), dtype=bool)],
            axis=0)

        return gmask

    def map_indices_to_global(self):
        """ Update the indices of min/max location for use with original array size. """
        for c in self.candidates.values():
            for key in ["vortensity_min_inds", "sigma_max_inds"]:
                ix = c[key][0] + self.vmi
                iy = c[key][0]
                c[key] = (ix, iy)

    def detect_vortex(self, include_mask=False, keep_internals=False):

        # Range of indices for view
        self.vmi = np.argmin(self.Rc[:, 0] < self.Rlims[0])
        self.vma = np.argmin(self.Rc[:, 0] < self.Rlims[1])

        if self.verbose:
            print(
                f"Radial index boundary for xlims: vmi = {self.vmi}, vma = {self.vma}")

        self.contour_image_dimensions()
        self.contour_image()
        self.find_contours()
        self.extract_closed_contours()
        self.extract_ellipse_contours()
        self.create_vortex_mask_in_view()

        self.create_array_views()
        self.calc_cell_masses()

        self.generate_ancestors()
        self.generate_decendents()
        self.prune_candidates_by_hierarchy()

        self.calculate_vortex_properties()
        self.remove_non_vortex_candidates()
        self.remove_duplicates_by_min_vort_pos()

        self.map_indices_to_global()

        self.sort_vortices_by_mass()

        if self.verbose:
            self.print_properties()

        self.remove_intermediate_data(include_mask, keep_internals)

        # self.vortices = [c.copy() for c in self.candidates.values()]

        # Return vortices
        return self.vortices

    def show_contours(self):
        """ Plot the contour image. """
        _, ax = plt.subplots()
        ax.imshow(self.thresh)

    def remove_intermediate_data(self, include_mask=False, keep_internals=False):
        self.vortices = []
        for c in self.candidates.values():
            if not keep_internals:
                for key in ["cnt", "mask_extended", "bounding_hor",
                            "bounding_vert", "area", "arcLength",
                            "ancestors", "decendents"]:
                    del c[key]
            if include_mask:
                c["mask"] = self.construct_global_mask(c)
            else:
                del c["mask_view"]
            self.vortices.append(c)


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


def gauss(x, y0, a, x0, sigma):
    """ A gaussian bell function.

    Parameters
    ----------
    x: array
        Coordinates
    y0: float
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
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + y0


def gauss_jac(x, y0, a, x0, sigma):
    d1 = np.ones(len(x))
    d2 = np.exp(-(x - x0)**2 / (2 * sigma**2))
    d3 = (x - x0) / (sigma**2) * a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    d4 = (x - x0)**2 / (sigma**3) * a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    return np.array([d1, d2, d3, d4]).transpose()


def gauss_jac_fix_y0(x, y0, a, x0, sigma):
    d2 = np.exp(-(x - x0)**2 / (2 * sigma**2))
    d3 = (x - x0) / (sigma**2) * a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    d4 = (x - x0)**2 / (sigma**3) * a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    return np.array([d2, d3, d4]).transpose()


def gauss_jac_fix_y0_a(x, y0, a, x0, sigma):
    d3 = (x - x0) / (sigma**2) * a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    d4 = (x - x0)**2 / (sigma**3) * a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    return np.array([d3, d4]).transpose()


def fit_gauss(x, y, y0=None, a=None, w=None, autoweight=True,
              blow=[None]*4, bup=[None]*4):
    """ Fit a gaussian to the data.

    Parameters
    ----------
    x : array
        Positions
    y : array
        Values
    autoweight : bool
        Perform two fits using deviations from first fit as weights.
    y0 : float
        Fixed value for offset.
    a : float
        Fixed value for amplitude.
    w : array
        Weights for the fit.

    Returns
    -------
    popt : array
        Optimal parameter
    pcov : 2-D array
        Estimated covariance of popt

    """

    blow_def = [-10*np.max(np.abs(y)), -10 *
                np.max(np.abs(y)), x[0], (x[-1]-x[0])/10]
    bup_def = [10*np.max(np.abs(y)), 10*np.max(np.abs(y)),
               x[-1], 10*(x[-1]-x[0])]

    blow_def = [-np.inf]*4
    bup_def = [np.inf]*4

    blow = [v if v is not None else d for v, d in zip(blow, blow_def)]
    bup = [v if v is not None else d for v, d in zip(bup, bup_def)]

    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

    if y0 is not None and a is not None:
        def f(x, x0, sig): return gauss(x, y0, a, x0, sig)
        p0 = [mean, sigma]
        blow = blow[2:]
        bup = bup[2:]
        popt, pcov = curve_fit(f, x, y, p0=p0, bounds=(blow, bup), sigma=w)
        popt = [y0, a, popt[0], popt[1]]
    elif y0 is not None:
        def f(x, a, x0, sig): return gauss(x, y0, a, x0, sig)
        p0 = [np.average(y), mean, sigma]
        blow = blow[1:]
        bup = bup[1:]
        popt, pcov = curve_fit(f, x, y, p0=p0, bounds=(blow, bup), sigma=w)
        popt = [y0, popt[0], popt[1], popt[2]]
    else:
        f = gauss
        p0 = [np.average(y), y[int(len(y)/2)], mean, sigma]
        popt, pcov = curve_fit(f, x, y, p0=p0, bounds=(blow, bup), sigma=w)

    if autoweight:
        return fit_gauss(x, y, y0=y0, a=a, w=w, autoweight=False)

    print("r0 min/max/fit =", blow[-2], bup[-2], popt[-2])
    print("sigma min/max/fit =", blow[-1], bup[-1], popt[-1])
    return popt, pcov


class GaussFitter:
    def __init__(self, x, y, name="", weights=None, autoweight=True, blow=None, bup=None, p0=None, fixed=None, verbose=False):
        self.x = x
        self.y = y
        self.autoweight = autoweight
        self.weights = weights
        self.name = name
        self.verbose = verbose

        self.parameters = ["y0", "a", "x0", "sigma"]

        self.blow = {key: -np.inf for key in self.parameters}
        self.bup = {key: np.inf for key in self.parameters}

        self.blow["sigma"] = 0
        dx = np.max(x) - np.min(x)
        self.blow["x0"] = np.min(x)
        self.bup["x0"] = np.max(x)

        if blow is not None:
            for key, val in blow.items():
                self.set_lower_bound(key, val)
        if bup is not None:
            for key, val in bup.items():
                self.set_upper_bound(key, val)

        self.fixed = fixed if fixed is not None else {}
        self.p0 = p0 if p0 is not None else {}

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

    def fit(self):
        popt, pcov = self.fit_single()

        if self.weights is None and self.autoweight:
            peak_value = popt[0] + popt[1]  # y0 + a
            self.calc_weights(peak_value)
            self.p0 = {k: v for k, v in zip(self.parameters, popt)}
            popt, pcov = self.fit_single()

        return popt, pcov

    def calc_weights(self, peak_value):
        difference = np.abs(self.y - peak_value)
        self.weights = np.exp(-difference/np.max(difference))

    def fit_single(self):
        x = self.x
        y = self.y
        fixed = self.fixed
        weights = self.weights
        lower = [self.blow[key] for key in self.parameters]
        upper = [self.bup[key] for key in self.parameters]

        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

        p0 = {
            "y0": np.average(y),
            "a": y[int(len(y)/2)] - np.average(y),
            "x0": mean,
            "sigma": sigma
        }
        for k, v in self.p0.items():
            p0[k] = v

        for key, v in p0.items():
            if v <= self.blow[key] or v >= self.bup[key]:
                p0[key] = 0.5*(self.blow[key] + self.bup[key])

        if "y0" in fixed and "a" in fixed:
            def f(x, x0, sig):
                return gauss(x, fixed["y0"], fixed["a"], x0, sig)

            def jac(x, x0, sig):
                return gauss_jac_fix_y0_a(x, fixed["y0"], fixed["a"], x0, sig)
            p0_vec = [p0[k] for k in ["x0", "sigma"]]
            bounds = (lower[2:], upper[2:])
            popt, pcov = curve_fit(
                f, x, y, p0=p0_vec, bounds=bounds, sigma=weights, jac=jac)
            popt = [fixed["y0"], fixed["a"], popt[0], popt[1]]
        elif "y0" in fixed:
            def f(x, a, x0, sig):
                return gauss(x, fixed["y0"], a, x0, sig)

            def jac(x, a, x0, sig):
                return gauss_jac_fix_y0(x, fixed["y0"], a, x0, sig)
            p0_vec = [p0[k] for k in ["a", "x0", "sigma"]]
            bounds = (lower[1:], upper[1:])
            popt, pcov = curve_fit(
                f, x, y, p0=p0_vec, bounds=bounds, sigma=weights, jac=jac)
            popt = [fixed["y0"], popt[0], popt[1], popt[2]]
        else:
            f = gauss
            jac = gauss_jac
            p0_vec = [p0[k] for k in ["y0", "a", "x0", "sigma"]]
            bounds = (lower, upper)
            popt, pcov = curve_fit(
                f, x, y, p0=p0_vec, bounds=bounds, sigma=weights, jac=jac)

        if self.verbose:
            print("-------------------")
            print(f"- fitting gaussian to {self.name}")
            for n, key in enumerate(self.parameters):
                if key in fixed:
                    print(f"- {key:6s} = {self.fixed[key]:+.3e} fixed")
                else:
                    print(
                        f"- {key:6s} = {popt[n]:+.3e} <- {p0[key]:+.3e} ,[{self.blow[key]:+.3e},{self.bup[key]:+.3e}]")
            print("-------------------")

        return popt, pcov


def save_fit(c, varname, axis, fit, parameters=["y0", "a", "x0", "sigma"]):
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
    pre = f"{varname}_fit_{axis}"
    for n, param in enumerate(parameters):
        c[f"{pre}_{param}"] = fit["popt"][n]
    for key, val in fit.items():
        c[f"{pre}_{key}"] = fit[key]


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
