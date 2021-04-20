import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_elliptic_contours(data, levels, max_ellipse_aspect_ratio, max_ellipse_deviation, verbose=False):
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
    Nx, Ny, CNx, CNy, int_aspect, supersample = contour_image_dimensions(data.shape, verbose=verbose)
    thresh = contour_image(CNx, CNy, data, levels)

    contours_largest, hierarchy = find_contours(thresh, verbose=verbose)
    contours_closed = extract_closed_contours(thresh, contours_largest, max_ellipse_aspect_ratio, verbose=verbose)

    candidates = extract_ellipse_contours(thresh, contours_closed, max_ellipse_deviation)
    create_vortex_mask(candidates, supersample, Nx, Ny, int_aspect, verbose=verbose)

    generate_ancestors(candidates, hierarchy, verbose=verbose)
    generate_decendents(candidates, verbose=verbose)
    prune_candidates_by_hierarchy(candidates)
    
    return candidates

def contour_image_dimensions(img_shape, verbose=False):

    Nx, Ny = img_shape
    int_aspect = int(np.max([Nx/Ny, Ny/Nx]))

    if int_aspect >= 2:
        if Nx < Ny:
            CNx = int_aspect*Nx
            CNy = Ny
        else:
            CNx = Nx
            CNy = int_aspect*Ny
    else:
        CNx = Nx
        CNy = Ny

    if min(CNx, CNy) < 1000:
        supersample = int(np.ceil(1000/min(CNx, CNy)))
    else:
        supersample = 1

    CNx *= supersample
    CNy *= supersample

    if verbose:
        print(
            f"Contour image dimensions: Nx {Nx}, Ny {Ny}, int_aspect {int_aspect}, supersample {supersample}, CNx {CNx}, CNy {CNy}")

    return Nx, Ny, CNx, CNy, int_aspect, supersample

def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)



def contour_image(CNx, CNy, vortensity, levels):
    fig = plt.figure(frameon=False, figsize=(CNx, 2*CNy), dpi=1)
    # fig.set_size_inches(w,h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # periodically extend vortensity
    Hhalf = int(vortensity.shape[1]/2)
    vort_pe = np.concatenate(
        [vortensity[:, Hhalf:],
            vortensity,
            vortensity[:, :Hhalf]],
        axis=1
    )

    ax.contour(vort_pe.transpose(),
                levels=levels, linewidths=CNx/1000)

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

    contours_dict = {n: {"contour": cnt, "opencv_contour_number": n}
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
        for k, c in enumerate(contours_closed):
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
                del contours_closed[to_del]
                contours_closed.append(contour)
        else:
            contours_closed.append(contour)

    if verbose:
        print("Number of closed contours:", len(contours_closed))
        print("Area of contours:", [c["pixel_area"]
                                    for c in contours_closed])

    return contours_closed


def extract_ellipse_contours(thresh, contours_closed, max_ellipse_deviation):
    # Extract contours that match an ellipse

    candidates = dict()
    for contour in contours_closed:
        cnt = contour["contour"]
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

        contour["mask_extended"] = im_shape
        candidates[contour["opencv_contour_number"]] = contour
        
    return candidates

def create_vortex_mask(candidates, supersample, Nx, Ny, int_aspect, verbose=False):
    # Transform the image from ellipse fitting images back to match the grid

    for contour in candidates.values():
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
        mask = mask[::supersample, ::supersample]
        if int_aspect >= 2:
            if Nx < Ny:
                mask = mask[::int_aspect, :]
            else:
                mask = mask[:, ::int_aspect]
        mask = np.array(mask, dtype=bool)
        contour["mask"] = mask

        # map the bounding points to view and data shape
        for key in ["bottom", "top", "left", "right"]:
            pnt = contour[key + "_extended"]
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
            contour[key] = (x, y)

    if verbose:
        print(
            f"Mapping mask: mask.shape = {mask.shape}, mask_orig.shape {mask_orig.shape}")

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


def generate_ancestors(candidates, hierarchy, verbose=False):
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
        if verbose:
            print("Ancestors:", c["opencv_contour_number"], ancestors)
            
def generate_decendents(candidates, verbose=False):
    # Construct decendents from ancestor list
    # This is done to avoid causing trouble when an intermediate contour is missing.

    decendents = {}
    for c in candidates.values():
        ancestors = c["ancestors"]
        for k, n in enumerate(ancestors):
            if not n in decendents or len(decendents[n]) < k:
                decendents[n] = [i for i in reversed(ancestors[:k])]

    for c in candidates.values():
        if c["opencv_contour_number"] in decendents:
            dec = decendents[c["opencv_contour_number"]]
        else:
            dec = []
        c["decendents"] = dec
        if verbose:
            print("Descendents:", c["opencv_contour_number"], dec)
            
def prune_candidates_by_hierarchy(candidates):
    # Remove children from candidates

    decendents = []
    for c in candidates.values():
        decendents += c["decendents"].copy()
    decendents = set(decendents)
    for n in decendents:
        del candidates[n]