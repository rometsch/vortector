import numpy as np

def interpolate_radial(data, trafo):
    data_interp = np.zeros_like(data)
    indices, weights = trafo
    for n, inds, ws in zip(range(len(indices)), indices, weights):
        data_interp[n] = data[inds[0]]*ws[0] + data[inds[1]]*ws[1]
    return data_interp

def interpolate_to_linear(data, radius_mg):
    rs = radius_mg[:,0]
    rlins = np.linspace(rs[0], rs[-1], len(rs))
    
    trafo = interpolation_indices_weights(rs, rlins)
    data_interp = interpolate_radial(data, trafo)
    
    trafo_rev = interpolation_indices_weights(rlins, rs)
    
    return data_interp, trafo_rev, trafo

def transform_indices(pnt, trafo):
    inds, ws = trafo
    xind, yind = pnt
    
    x = inds[xind][0]*ws[xind][0] + inds[xind][1]*ws[xind][1]
    x = int(np.round(x))
    return (x, yind)


def interpolation_indices_weights(x, xto):
    """ Calculate indices and weights to interpolate from coordinates x to y.
    
    x and y must be arrays of monotonically increasing series with the same start and end.
    
    Parameters
    ----------
    x : array of floats
        Coordinates to interpolate from.
    xto : array of floats
        Coordinates to interpolate to.
    """
    indices = np.zeros((len(xto), 2), dtype=np.int32)
    weights = np.zeros((len(xto), 2), dtype=np.float64)
    indices[0] = [0, 0]
    weights[0] = [0.5, 0.5]
    for n in range(1, len(xto)):
        for k in range(indices[0, 0], len(xto)):
            nup = k
            if x[nup] >= xto[n]:
                break
        for k in range(nup, -1, -1):
            nlow = k
            if x[nlow] <= xto[n]:
                break
        if nlow >= nup:
            weights[n, :] = [0.5, 0.5]
            indices[n, :] = [n, n]
        else:
            wup = (xto[n] - x[nlow]) / (x[nup] - x[nlow])
            wlow = (x[nup] - xto[n]) / (x[nup] - x[nlow])
            weights[n, :] = [wlow, wup]
            indices[n, :] = [nlow, nup]
    return indices, weights