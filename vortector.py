import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


class Vortector:
    def __init__(self, Xc, Yc, A, vortensity, Sigma, Sigma0,
                 Rlims, levels=[float(x) for x in np.arange(-1, 1.5, 0.05)],
                 med=0.15, mear=np.inf, mvd=0.01, verbose=False):

        self.vortensity = vortensity

        self.Sigma = Sigma
        self.Sigma_background = Sigma0

        self.Rc = Xc
        self.Phic = Yc
        self.cell_area = A

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
            except ValueError:
                pass

    def fit_gaussians(self, c):

        vals = self.Rho_view

        inds = c["vortensity_min_inds"]
        mask = c["mask"]
        mask_r = mask[:, inds[1]]
        mask_phi = mask[inds[0], :]

        names = ["sigma", "vortensity", "vorticity"]
        arrays = [self.Rho_view, self.vortensity_view]

        r = self.Xc_view[:, inds[1]]
        phi = self.Yc_view[inds[0], :]

        for name, vals in zip(names, arrays):
            vals_r = vals[:, inds[1]]
            vals_phi = vals[inds[0], :]

            x = r[mask_r]
            y = vals_r[mask_r]
            popt, pcov = fit_gauss(x, y)

            c[name + "_fit_r_y0"] = popt[0]
            c[name + "_fit_r_a"] = popt[1]
            c[name + "_fit_r_x0"] = popt[2]
            c[name + "_fit_r_sigma"] = popt[3]

            x, y = combine_periodic(phi, vals_phi, mask_phi)
            popt, pcov = fit_gauss(x, y)

            c[name + "_fit_phi_y0"] = popt[0]
            c[name + "_fit_phi_a"] = popt[1]
            c[name + "_fit_phi_x0"] = popt[2]
            c[name + "_fit_phi_sigma"] = popt[3]

    def calc_vortex_mass(self, c):
        mask = c["mask"]
        c["mass"] = np.sum(self.mass_view[mask])
        c["mass_background"] = np.sum(self.mass_background_view[mask])
        c["mass_enhancement"] = c["mass"] - c["mass_background"]

    def calc_vortensity(self, c):
        mask = c["mask"]
        c["vortensity_mean"] = np.mean(self.vortensity_view[mask])
        c["vortensity_median"] = np.median(self.vortensity_view[mask])
        c["vortensity_min"] = np.min(self.vortensity_view[mask])
        c["vortensity_max"] = np.max(self.vortensity_view[mask])

    def calc_sigma(self, c):
        mask = c["mask"]
        c["sigma_mean"] = np.mean(self.Rho_view[mask])
        c["sigma_median"] = np.median(self.Rho_view[mask])
        c["sigma_min"] = np.min(self.Rho_view[mask])
        c["sigma_max"] = np.max(self.Rho_view[mask])
        c["sigma0_mean"] = np.mean(self.Rho_background_view[mask])
        c["sigma0_median"] = np.median(self.Rho_background_view[mask])
        c["sigma0_min"] = np.min(self.Rho_background_view[mask])
        c["sigma0_max"] = np.max(self.Rho_background_view[mask])

    def calc_vortex_extent(self, c):
        mask = c["mask"]
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
        mask = c["mask"]
        A = self.cell_area_view
        c["vortensity_flux"] = np.sum((A*self.vortensity_view)[mask])
        c["vortensity_exp_flux"] = np.sum(
            (A*np.exp(-self.vortensity_view))[mask])

    def find_vortensity_min_position(self, contour):
        # Calculate the position of minimum vortensity
        mask = np.logical_not(contour["mask"])
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
        mask = np.logical_not(contour["mask"])
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
        mask = v["mask"]
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
                del c["mask"]
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


def combine_periodic(x, y, m, lb=-np.pi, rb=np.pi):
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
    lb: float
        Position of the left boundary.
    rb: float
        Position of the right boundary.

    Returns
    -------
    x : array
        coordinates
    y : array
        values
    """
    if m[0] and m[-1] and not all(m):
        bnd = np.where(m == False)[0][0]
        xl = x[:bnd]
        yl = y[:bnd]
        bnd = np.where(m == False)[0][-1]
        xr = x[bnd:]
        yr = y[bnd:]
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


def fit_gauss(x, y, double_fit=False):
    """ Fit a gaussian to the data.

    Parameters
    ----------
    x : array
        Positions
    y : array
        Values
    double_fit : bool
        Perform two fits using deviations from first fit as weights.

    Returns
    -------
    popt : array
        Optimal parameter
    pcov : 2-D array
        Estimated covariance of popt

    """
    blow = [-10*np.max(np.abs(y)), -10*np.max(np.abs(y)), -
            np.inf, (x[-1]-x[0])/20]
    bup = [10*np.max(np.abs(y)), 10*np.max(np.abs(y)), np.inf, (x[-1]-x[0])/2]
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[np.average(y), y[int(len(y)/2)], mean, sigma],
                           bounds=(blow, bup))
    if double_fit:
        peak_value = popt[0] + popt[1]  # y0 + a
        print("popt", popt)
        print("peak_value", peak_value)
        difference = np.abs(y - peak_value)
        inv_weights = np.maximum(difference, np.max(difference)*1e-5)
        popt, pcov = curve_fit(gauss, x, y, p0=[np.average(y), y[int(len(y)/2)], mean, sigma],
                               sigma=inv_weights, bounds=(blow, bup))
    return popt, pcov
