import numpy as np
import cv2
import matplotlib.pyplot as plt

from uuid import uuid4 as generate_uuid


def detect_elliptic_contours(data, levels, max_ellipse_aspect_ratio, max_ellipse_deviation, periodic=True, blur=None, verbose=False):
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
    blur : int
        Width of the gaussian filter. Default None
    verbose : bool
        Verbose output. Default False.

    Returns
    -------
    dict
        Dictionary of the candidate contours with information stored in dictionaries. 
    """
    if blur is not None:
        data = gaussian_blur(data, blur)

    Nx, Ny = data.shape
    data, pad = extend_array(data)

    contours_largest = find_contours(data, levels, verbose=verbose)
    contours_closed = extract_closed_contours(
        pad, Nx, Ny, contours_largest, max_ellipse_aspect_ratio,
        periodic=periodic, verbose=verbose)
    remove_duplicates_by_geometry(contours_closed, verbose=verbose)

    deviation_method = "semianalytic"  # configvalue
    contours = extract_ellipse_contours(
        data.shape, contours_closed, max_ellipse_deviation,
        deviation_method=deviation_method)

    # generate_ancestors(contours, hierarchy)
    # generate_descendants(contours)
    # prune_candidates_by_hierarchy(contours)

    parameters = {
        "Nx": Nx,
        "Ny": Ny,
        "max_ellipse_aspect_ratio": max_ellipse_aspect_ratio,
        "max_ellipse_deviation": max_ellipse_deviation,
        "levels": [float(x) for x in levels]
    }

    for contour in contours.values():
        contour["parameters"] = parameters

    candidates = [{"detection": contour} for contour in contours.values()]
    create_vortex_mask(candidates, Ny, pad, periodic=periodic)
    remove_empty_contours(candidates)

    return candidates


def gaussian_blur(data, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(data, sigma)


def extend_array(data):
    """ Periodically extend the array in y direction. """
    Nx, Ny = data.shape
    pad_low = Ny//2
    pad_high = Ny - pad_low
    extended = np.pad(data, ((0, 0), (pad_low, pad_high)), mode="wrap")
    return extended, (pad_low, pad_high)


def remove_duplicates_by_geometry(contours, verbose=False, Ny=None, periodic=False):
    """ Remove duplicate contours.

    Sort out one of multiple contours that have the same area and bounding box.
    """
    to_del = []

    for n_can, c in enumerate(contours):
        if n_can in to_del:
            continue
        bbox = c["bbox_inds"]
        area = c["pixel_area"]

        for n_other, o in enumerate(contours):
            if n_other == n_can:
                continue
            o_bbox = o["bbox_inds"]
            o_area = o["pixel_area"]

            if area == o_area and (bbox == o_bbox).all():
                to_del.append(n_other)

    to_del = set(to_del)
    for k, n in enumerate(to_del):
        del contours[n-k]

    if verbose:
        print(
            f"Removed {len(to_del)} contours which were duplicates. {len(contours)} remaining.")


def remove_boundary_cases(candidates, shape):
    """ Remove contours which overlap with the boundary.

    These are likely artifacts of the contour detection process.

    Check wether any of the bounding points lies on the image boundary.
    """
    to_del = []
    for n, contour in enumerate(candidates):
        pnt_xhigh = contour["pnt_xhigh_img"]
        pnt_xlow = contour["pnt_xlow_img"]
        pnt_ylow = contour["pnt_ylow_img"]
        pnt_yhigh = contour["pnt_yhigh_img"]

        xs = np.array([p[0]
                       for p in (pnt_xhigh, pnt_xlow, pnt_ylow, pnt_yhigh)])
        ys = np.array([p[1]
                       for p in (pnt_xhigh, pnt_xlow, pnt_ylow, pnt_yhigh)])

        if any(np.abs(xs - shape[0]) < 2) or any(xs < 2):
            to_del.append(n)
            continue
        if any(np.abs(ys - shape[1]) < 2) or any(ys < 2):
            to_del.append(n)

    for k, n in enumerate(to_del):
        del candidates[n-k]


def remove_empty_contours(candidates):
    """ Remove artifact contours that don't include any data cell. """
    to_del = []
    for n, candidate in enumerate(candidates):
        if np.sum(candidate["mask"]) == 0:
            to_del.append(n)
    for k, n in enumerate(to_del):
        del candidates[n-k]


def contour_image_dimensions(img_shape, min_image_size=1000, verbose=False):

    Nx, Ny = img_shape
    if Nx == 0 or Ny == 0:
        raise ValueError(
            f"Shape of data is zero in one direction: (Nx, Ny) = ({Nx}, {Ny})")
    int_aspect = 1

    supersample = [1, 1]
    SNx = Nx * supersample[0]
    SNy = Ny * supersample[1]

    return Nx, Ny, SNx, SNy, int_aspect, supersample


def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)


def contour_image(SNx, SNy, vortensity, levels, min_image_size=1000, periodic=False):
    size = (SNx, 2*SNy if periodic else SNy)

    if periodic:
        # periodically extend vortensity
        Ny = vortensity.shape[1]
        pad_pnt_xlow = int(Ny/2)
        pad_pnt_xhigh = Ny - pad_pnt_xlow
        pad = (pad_pnt_xhigh, pad_pnt_xlow)
        vort_pe = np.pad(
            vortensity, ((0, 0), pad), mode="wrap")
    else:
        vort_pe = vortensity
        pad = (0, 0)

    linewidth = SNx/min_image_size  # configvalue

    mpl_inter = plt.matplotlib.is_interactive()
    if mpl_inter:
        plt.ioff()

    fig = plt.figure(frameon=False, figsize=size, dpi=1)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.contour(vort_pe.transpose(),
               levels=levels, linewidths=linewidth)

    img_data = fig2rgb_array(fig)
    plt.close(fig)
    if mpl_inter:
        plt.ion()

    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    # Threshold contour image for full contrast
    _, thresh = cv2.threshold(img_data, 250, 255, 0)

    return thresh, pad


def contour_threshold(data, threshold):
    image = data < threshold
    image = image / np.max(image) * 255
    image = image.astype(np.uint8)
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def find_contours(data, levels, verbose=False):
    # Extract contours and construct hierarchy

    contours = []
    for n, th in enumerate(levels):
        rv = contour_threshold(data, th)
        for k, bnd in enumerate(rv):
            d = {
                "boundary": bnd,
                "contour_value": th,
                "opencv_contour_number": 1000*n + k,
                "uuid": f"{generate_uuid()}"}
            contours.append(d)

    if verbose:
        print("Number of found contours:", len(contours))

    for d in contours:
        d["pixel_area"] = cv2.contourArea(d["boundary"])

    # take the up to 100 largest patches
    contours_largest = sorted(contours, key=lambda x: -x["pixel_area"])[:1000]

    return contours_largest


def extract_closed_contours(pad, Nx, Ny, contours_largest, max_ellipse_aspect_ratio, periodic=False, verbose=False):
    # Extract closed contours

    contours_closed = []
    for contour in contours_largest:
        cnt = contour["boundary"]
        l = cv2.arcLength(cnt, True)
        contour["pixel_arcLength"] = l
        a = contour["pixel_area"]
        pnt_ylow = cnt[cnt[:, :, 0].argmin()][0][::-1]
        pnt_yhigh = cnt[cnt[:, :, 0].argmax()][0][::-1]
        pnt_xlow = cnt[cnt[:, :, 1].argmin()][0][::-1]
        pnt_xhigh = cnt[cnt[:, :, 1].argmax()][0][::-1]

        dx = pnt_xhigh[0] - pnt_xlow[0]
        dy = pnt_yhigh[1] - pnt_ylow[1]

        # sort out mirrors of contours fully contained in original area
        # if pnt_xhighmost[1] < pad[0] or pnt_xlowmost[1] > Ny+pad[1]:
        #     continue

        is_not_too_elongated = dx > 0 and dy > 0 and max(
            dx/dy, dy/dx) < max_ellipse_aspect_ratio
        is_area_larger_delimiter = l > 0 and a > l
        is_not_spanning_whole_height = dy < 0.95*Ny
        if not is_not_too_elongated:
            # print("discarding because is_not_too_elongated", dx, dy)
            continue
        if not is_area_larger_delimiter:
            # print("discarding because is_area_larget_delimiter")
            continue
        if not is_not_spanning_whole_height:
            # print("discarding because is_not_spanning_whole_height", dy, Ny)
            continue

        bounding_x = np.array([pnt_xlow[0], pnt_xhigh[0]])
        bounding_y = np.array([pnt_ylow[1], pnt_yhigh[1]])
        contour["bounding_x_img"] = bounding_x
        contour["bounding_y_img"] = bounding_y
        # save the bounding points
        contour["pnt_xlow_img"] = pnt_xlow
        contour["pnt_xhigh_img"] = pnt_xhigh
        contour["pnt_ylow_img"] = pnt_ylow
        contour["pnt_yhigh_img"] = pnt_yhigh
        contour["dx_img"] = dx
        contour["dy_img"] = dy

        for key in ["pnt_xlow", "pnt_xhigh", "pnt_ylow", "pnt_yhigh"]:
            x, y = contour[key + "_img"]
            if periodic:
                y = (y - pad[0]) % Ny
            x = int(np.round(x))
            y = int(np.round(y))
            contour[key] = (x, y)

        bbox_inds = np.array([contour[key] for key in
                              ["pnt_xlow", "pnt_xhigh", "pnt_ylow", "pnt_yhigh"]])
        contour["bbox_inds"] = bbox_inds

        contours_closed.append(contour)

    if verbose:
        print("Number of closed contours:", len(contours_closed))
        print("Area of contours:", [c["pixel_area"]
                                    for c in contours_closed])

    return contours_closed


def remove_periodic_duplicates(closed_contours, Ny, pad):
    indices_to_delete = []
    for n, contour in enumerate(closed_contours):
        # sort out the lower of mirror images
        bounding_y = contour["bounding_y_img"]
        bounding_x = contour["bounding_x_img"]

        found_mirror = False
        for k, other in enumerate(closed_contours[n:]):
            if k in indices_to_delete:
                continue
            same_hor = (bounding_y == other["bounding_y_img"]).all()
            same_vert = (np.abs(bounding_x %
                                Ny - other["bounding_x_img"] % Ny) < 20).all()
            if same_hor and same_vert:
                if other["bounding_x_img"][1] > bounding_x[1]:
                    indices_to_delete.append(k)
                    found_mirror = True
            if found_mirror == True:
                break

    for k, n in enumerate(indices_to_delete):
        del closed_contours[n-k]


def extract_ellipse_contours(img_shape, contours_closed, max_ellipse_deviation, deviation_method="semianalytic"):
    # Extract contours that match an ellipse

    candidates = dict()
    for contour in contours_closed:
        cnt = contour["boundary"]
        ellipse = cv2.fitEllipse(cnt)
        if deviation_method == "semianalytic":
            rel_delta, delta = ellipse_deviation_semianalytic(
                cnt, ellipse, contour["pixel_area"])
        elif deviation_method == "draw":
            rel_delta, delta = ellipse_deviation_draw(
                cnt, ellipse, contour["pixel_area"])
        else:
            ValueError(
                f"Method for estimating ellipse deviation does not exist: '{deviation_method}'")

        if rel_delta > max_ellipse_deviation:
            continue

        contour["ellipse"] = {
            "center_img": ellipse[0],
            "axesLengths_img": ellipse[1],
            "angle_img": ellipse[2]
        }
        contour["mask_img"] = contour_mask(img_shape, cnt)
        contour["ellipse_area_delta_relative"] = rel_delta
        contour["ellipse_area_delta"] = delta
        candidates[contour["opencv_contour_number"]] = contour

    return candidates


def ellipse_pnt(x, y, ellipse):
    angle = ellipse[2]/180*np.pi
    cx = ellipse[0][0]
    cy = ellipse[0][1]
    e1 = ellipse[1][0]/2
    e2 = ellipse[1][1]/2

    l = np.sqrt((x-cx)**2 + (y-cy)**2)
    nx = (x-cx) / l
    ny = (y-cy) / l

    na = np.arctan2(ny, nx)
    alpha = na - angle

    tansq = np.tan(alpha)**2
    length = e1 * np.sqrt((1+tansq)/(1+tansq*e1**2/e2**2))

    px = cx + length * nx
    py = cy + length * ny

    return px, py


def tetragon_area(p1, p2, p3, p4):
    return triangle_area(p1, p2, p3) + triangle_area(p1, p3, p4)


def triangle_area(p1, p2, p3):
    signed_area = (
        +p3[0] * (p1[1] - p2[1])
        + p1[0] * (p2[1] - p3[1])
        + p2[0] * (p3[1] - p1[1])
    ) / 2
    return np.abs(signed_area)


def calc_diff_area(bx, by, ellipse):
    px, py = ellipse_pnt(bx, by, ellipse)
    area = 0
    for n in range(len(px)-1):
        p1 = [px[n], py[n]]
        p4 = [px[n+1], py[n+1]]
        p2 = [bx[n], by[n]]
        p3 = [bx[n+1], by[n+1]]
        area += tetragon_area(p1, p2, p3, p4)
    return area


def ellipse_deviation_semianalytic(boundary_pnts, ellipse, area):
    bx = boundary_pnts[:, 0, 0]
    by = boundary_pnts[:, 0, 1]
    if len(bx) > 100:
        Nstride = int(np.ceil(len(bx)/100))
    else:
        Nstride = 1
    area_diff = calc_diff_area(bx[::Nstride], by[::Nstride], ellipse)
    return area_diff/area, area_diff


def ellipse_deviation_draw(boundary_pnts, ellipse, area):
    # detemine minimum and maximum extend of the ellipse
    xmin = np.min(boundary_pnts[:, 0, 0])
    xmax = np.max(boundary_pnts[:, 0, 0])
    ymin = np.min(boundary_pnts[:, 0, 1])
    ymax = np.max(boundary_pnts[:, 0, 1])

    # make sure the ellipse also fits
    ec = ellipse[0]
    L = np.ceil(np.max(ellipse[1])/2)
    xmin = int(min(ec[0]-L, xmin))
    xmax = int(max(ec[0]+L, xmin))
    ymin = int(min(ec[1]-L, ymin))
    ymax = int(max(ec[1]+L, ymin))

    Nx = xmax - xmin
    Ny = ymax - ymin

    img_shape = (Ny, Nx)

    # adjust a copy of the boundary points
    boundary_pnts = np.copy(boundary_pnts)
    boundary_pnts[:, 0, 0] -= xmin
    boundary_pnts[:, 0, 1] -= ymin

    # adjust a copy of the ellipse
    ellipse = ((ellipse[0][0] - xmin, ellipse[0][1] - ymin),
               ellipse[1], ellipse[2])

    im_shape = np.zeros(img_shape)
    cv2.drawContours(im_shape, [boundary_pnts], 0, (255, 255, 255), -1)

    im_ellipse = np.zeros(img_shape)
    im_ellipse = cv2.ellipse(im_ellipse, ellipse, (255, 255, 255), -1)

    difference = np.abs(im_shape - im_ellipse)
    difference_area = np.sum(difference/255)

    rel_delta = difference_area / area
    return rel_delta, difference_area


def ellipse_deviation_draw_full(img_shape, boundary_pnts, ellipse, area):
    im_shape = np.zeros(img_shape)
    cv2.drawContours(im_shape, [boundary_pnts], 0, (255, 255, 255), -1)

    im_ellipse = np.zeros(img_shape)
    im_ellipse = cv2.ellipse(im_ellipse, ellipse, (255, 255, 255), -1)

    difference = np.abs(im_shape - im_ellipse)
    difference_area = np.sum(difference/255)

    rel_delta = difference_area / area
    return rel_delta


def contour_mask(img_shape, boundary_pnts):
    """Create a boolean mask indicating the interior of the image.

    Parameters
    ----------
    img_shape : tuple of ints
        Number of pixels in x and y direction.
    boundary_pnts : np.array of pairs of ints
        List of boundary points.
    """
    mask = np.zeros(img_shape, dtype=np.int8)
    cv2.drawContours(mask, [boundary_pnts], 0, (255, 255, 255), -1)
    return mask


def create_vortex_mask(candidates, Ny, pad, periodic=False):
    # Transform the image from ellipse fitting images back to match the grid

    for candidate in candidates:
        contour = candidate["detection"]
        mask = contour["mask_img"]
        mask = np.array(mask, dtype=bool)
        if periodic:
            pad_low = pad[0]
            mask_high = mask[:, Ny+pad_low:]
            mask_low = mask[:, :pad_low]
            mask_pad = np.concatenate([mask_high, mask_low], axis=1)
            mask_orig = mask[:, pad_low:Ny+pad_low]

            mask_reduced = np.logical_or(mask_orig, mask_pad)
            mask = mask_reduced

        candidate["mask"] = mask

        for key in ["pnt_xhigh", "pnt_xlow", "pnt_ylow", "pnt_yhigh"]:
            x, y = contour[key + "_img"]
            if periodic:
                y = (y - pad_low) % Ny
            x = int(np.round(x))
            y = int(np.round(y))
            candidate[key] = (x, y)


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
