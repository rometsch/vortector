import numpy as np
import cv2
import matplotlib.pyplot as plt


def detect_elliptic_contours(data, levels, max_ellipse_aspect_ratio, max_ellipse_deviation, periodic=True, verbose=False):
    """ Detect closed equivalue lines in a 2D contour plot of the data.

    1. create a contour line image of the data
    2. find closed contours
    3. fit ellipses to contours
    4. remove contours that are enclosed in another one

    Parameters
    ----------
    data : numpy.array 2D
        2D array of values to be analyzed.
    levels : array of floats
        Value levels for which to analyze equivalue lines.
    max_ellipse_aspect_ratio : float
        Maximum aspect ratio for ellipses.
    max_ellipse_deviation : float
        Maximum threshold for the deviation of a contour area to the fitted ellipse.
    verbose : bool
        Verbose output. Default False.

    Returns
    -------
    dict
        Dictionary of the candidate contours with information stored in dictionaries. 
    """
    Nx, Ny, SNx, SNy, int_aspect, supersample = contour_image_dimensions(
        data.shape, verbose=verbose)
    thresh = contour_image(SNx, SNy, data, levels, periodic=periodic)

    _, ax = plt.subplots(dpi=150)
    ax.imshow(thresh)

    contours_largest, hierarchy = find_contours(thresh, verbose=verbose)
    contours_closed = extract_closed_contours(
        thresh, contours_largest, max_ellipse_aspect_ratio, verbose=verbose)
    if periodic:
        remove_periodic_duplicates(contours_closed, SNy)
    contours = extract_ellipse_contours(
        thresh, contours_closed, max_ellipse_deviation)

    generate_ancestors(contours, hierarchy)
    generate_descendants(contours)
    prune_candidates_by_hierarchy(contours)

    parameters = {
        "Nx": Nx,
        "Ny": Ny,
        "SNx": SNx,
        "SNy": SNy,
        "int_aspect": int_aspect,
        "supersample": supersample,
        "max_ellipse_aspect_ratio": max_ellipse_aspect_ratio,
        "max_ellipse_deviation": max_ellipse_deviation,
        "levels": [float(x) for x in levels]
    }

    for contour in contours.values():
        contour["parameters"] = parameters

    candidates = [{"detection": contour} for contour in contours.values()]
    create_vortex_mask(candidates, supersample, Nx,
                       Ny, int_aspect, verbose=verbose, mirror=periodic)

    remove_empty_contours(candidates)

    return candidates


def remove_empty_contours(candidates):
    """ Remove artifact contours that don't include any data cell. """
    to_del = []
    for n, candidate in enumerate(candidates):
        if np.sum(candidate["mask"]) == 0:
            to_del.append(n)
    for k, n in enumerate(to_del):
        del candidates[n-k]


def contour_image_dimensions(img_shape, verbose=False):

    Nx, Ny = img_shape
    int_aspect = int(np.max([Nx/Ny, Ny/Nx]))

    if int_aspect >= 2:
        if Nx < Ny:
            SNx = int_aspect*Nx
            SNy = Ny
        else:
            SNx = Nx
            SNy = int_aspect*Ny
    else:
        SNx = Nx
        SNy = Ny

    if min(SNx, SNy) < 1000:
        supersample = int(np.ceil(1000/min(SNx, SNy)))
    else:
        supersample = 1

    SNx *= supersample
    SNy *= supersample

    if verbose:
        print(
            f"Contour image dimensions: Nx {Nx}, Ny {Ny}, int_aspect {int_aspect}, supersample {supersample}, CNx {SNx}, CNy {SNy}")

    return Nx, Ny, SNx, SNy, int_aspect, supersample


def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)


def contour_image(SNx, SNy, vortensity, levels, periodic=False):
    size = (SNx, 2*SNy if periodic else SNy)
    fig = plt.figure(frameon=False, figsize=size, dpi=1)
    # fig.set_size_inches(w,h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    if periodic:
        # periodically extend vortensity
        Ny = vortensity.shape[1]
        pad_top = int(Ny/2)
        pad_bottom = Ny - pad_top
        vort_pe = np.pad(
            vortensity, ((0, 0), (pad_bottom, pad_top)), mode="wrap")
    else:
        vort_pe = vortensity

    linewidth = SNx/1000  # configvalue
    ax.contour(vort_pe.transpose(),
               levels=levels, linewidths=linewidth)

    img_data = fig2rgb_array(fig)
    plt.close(fig)

    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    # Threshold contour image for full contrast
    _, thresh = cv2.threshold(img_data, 250, 255, 0)

    return thresh


def find_contours(thresh, verbose=False):
    # Extract contours and construct hierarchy

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if verbose:
        print("Number of found contours:", len(contours))

    contours_dict = {n: {"boundary": cnt, "opencv_contour_number": n}
                     for n, cnt in enumerate(contours)}

    areas = [cv2.contourArea(c) for c in contours]
    for n, d in enumerate(contours_dict.values()):
        d["pixel_area"] = areas[n]

    sort_inds = np.argsort(areas)

    # take the up to 100 largest patches
    contours_largest = [contours_dict[i]
                        for i in [n for n in sort_inds[::-1]][:100]]

    return contours_largest, hierarchy


def extract_closed_contours(thresh, contours_largest, max_ellipse_aspect_ratio, verbose=False):
    # Extract closed contours

    contours_closed = []
    for n, contour in enumerate(contours_largest):
        cnt = contour["boundary"]
        l = cv2.arcLength(cnt, True)
        contour["pixel_arcLength"] = l
        a = contour["pixel_area"]
        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

        dx = rightmost[0] - leftmost[0]
        dy = bottommost[1] - topmost[1]

        bounding_hor = np.array([rightmost[0], leftmost[0]])
        bounding_vert = np.array([topmost[1], bottommost[1]])
        contour["bounding_hor_img"] = bounding_hor
        contour["bounding_vert_img"] = bounding_vert
        # save the bounding points
        contour["bottom_img"] = bottommost
        contour["top_img"] = topmost
        contour["left_img"] = leftmost
        contour["right_img"] = rightmost
        contour["dx_img"] = dx
        contour["dy_img"] = dy

        Nh = int(thresh.shape[0]/2)
        Nq = int(thresh.shape[0]/4)

        # sort out mirrors of contours fully contained in original area
        if bottommost[1] < Nq or topmost[1] > 3*Nq:
            continue

        is_not_too_elongated = dx > 0 and dy > 0 and max(
            dx/dy, dy/dx) < max_ellipse_aspect_ratio
        is_area_larget_delimiter = l > 0 and a > l
        is_not_spanning_whole_height = dy < 0.5*0.95*thresh.shape[0]

        if not(is_not_too_elongated and is_area_larget_delimiter and is_not_spanning_whole_height):
            continue

        contours_closed.append(contour)

    if verbose:
        print("Number of closed contours:", len(contours_closed))
        print("Area of contours:", [c["pixel_area"]
                                    for c in contours_closed])

    return contours_closed


def remove_periodic_duplicates(closed_contours, SNy):
    Nh = SNy // 2
    indices_to_delete = []
    for n, contour in enumerate(closed_contours):
        # sort out the lower of mirror images
        bounding_hor = contour["bounding_hor_img"]
        bounding_vert = contour["bounding_vert_img"]

        found_mirror = False
        for k, other in enumerate(closed_contours[n:]):
            if k in indices_to_delete:
                continue
            same_hor = (bounding_hor == other["bounding_hor_img"]).all()
            same_vert = (np.abs(bounding_vert %
                                Nh - other["bounding_vert_img"] % Nh) < 20).all()
            if same_hor and same_vert:
                if other["bounding_vert_img"][1] > bounding_vert[1]:
                    indices_to_delete.append(k)
                    found_mirror = True
            if found_mirror == True:
                break

    for k, n in enumerate(indices_to_delete):
        del closed_contours[n-k]


def extract_ellipse_contours(thresh, contours_closed, max_ellipse_deviation):
    # Extract contours that match an ellipse

    candidates = dict()
    for contour in contours_closed:
        cnt = contour["boundary"]
        ellipse = cv2.fitEllipse(cnt)

        im_shape = np.zeros(thresh.shape)
        cv2.drawContours(im_shape, [cnt], 0, (255, 255, 255), -1)

        im_ellipse = np.zeros(thresh.shape)
        im_ellipse = cv2.ellipse(im_ellipse, ellipse, (255, 255, 255), -1)

        difference = np.abs(im_shape - im_ellipse)
        difference_area = np.sum(difference/255)

        rel_delta = difference_area / contour["pixel_area"]

        if rel_delta > max_ellipse_deviation:
            continue

        contour["mask_img"] = im_shape
        candidates[contour["opencv_contour_number"]] = contour

    return candidates


def create_vortex_mask(candidates, supersample, Nx, Ny, int_aspect, mirror=False, verbose=False):
    # Transform the image from ellipse fitting images back to match the grid
    print(f"creating masks with mirror={mirror}")
    for candidate in candidates:

        if not mirror:
            contour = candidate["detection"]
            mask = contour["mask_img"]

            # fit back to original data shape
            mask = mask.transpose()[:, ::-1]
            mask = mask[::supersample, ::supersample]
            if int_aspect >= 2:
                if Nx < Ny:
                    mask = mask[::int_aspect, :]
                else:
                    mask = mask[:, ::int_aspect]
            mask = np.array(mask, dtype=bool)
            candidate["mask"] = mask

            for key in ["bottom", "top", "left", "right"]:
                pnt = contour[key + "_img"]
                x, y = map_ext_pnt_to_orig(pnt, Ny/2)
                y = Ny - y
                x /= supersample
                y /= supersample
                if Nx < Ny:
                    x /= int_aspect
                else:
                    y /= int_aspect
                x = int(x)
                y = int(y)
                candidate[key] = (x, y)
        else:
            contour = candidate["detection"]
            mask_img = contour["mask_img"]
            # reduce back to normal image size
            Nq = int(mask_img.shape[0]/4)
            mask_lower = mask_img[3*Nq:, :]
            mask_upper = mask_img[:Nq, :]
            mask_repeated = np.concatenate([mask_lower, mask_upper])
            mask_orig = mask_img[Nq:3*Nq, :]
            mask_reduced = np.logical_or(mask_orig, mask_repeated)

            # fit back to original data shape
            mask = mask_reduced.transpose()[:, ::-1]
            mask = mask[::supersample, ::supersample]
            if int_aspect >= 2:
                if Nx < Ny:
                    mask = mask[::int_aspect, :]
                else:
                    mask = mask[:, ::int_aspect]
            mask = np.array(mask, dtype=bool)
            candidate["mask"] = mask

            # map the bounding points to view and data shape
            for key in ["bottom", "top", "left", "right"]:
                pnt = contour[key + "_img"]
                x, y = map_ext_pnt_to_orig(pnt, Nq)
                y = 2*Nq - y
                x /= supersample
                y /= supersample
                if Nx < Ny:
                    x /= int_aspect
                else:
                    y /= int_aspect
                x = int(x)
                y = int(y)
                candidate[key] = (x, y)


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


def generate_ancestors(candidates, hierarchy):
    # Generate ancestor list

    # The hierarchy generated by opencv in the contour finder outputs a list with the syntax
    # ```
    # [Next, Previous, First_Child, Parent]
    # ```
    # If any of those is not available its encoded  by -1.

    for c in candidates.values():
        ancestors = []
        n_parent = c["opencv_contour_number"]
        for n in range(1000):
            n_parent = hierarchy[0, n_parent, 3]
            if n_parent == -1 or n_parent not in candidates:
                break
            ancestors.append(n_parent)
        c["ancestors"] = ancestors


def generate_descendants(candidates):
    # Construct descendants from ancestor list
    # This is done to avoid causing trouble when an intermediate contour is missing.

    descendants = {}
    for c in candidates.values():
        ancestors = c["ancestors"]
        for k, n in enumerate(ancestors):
            if not n in descendants or len(descendants[n]) < k:
                descendants[n] = [i for i in reversed(ancestors[:k])]

    for c in candidates.values():
        if c["opencv_contour_number"] in descendants:
            dec = descendants[c["opencv_contour_number"]]
        else:
            dec = []
        c["descendants"] = dec


def prune_candidates_by_hierarchy(candidates):
    # Remove children from candidates

    descendants = []
    for c in candidates.values():
        descendants += c["descendants"].copy()
    descendants = set(descendants)
    for n in descendants:
        del candidates[n]
