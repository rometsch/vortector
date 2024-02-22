""" Fast 2D arrays with opensimplex noise.

This is an adaptation of the OpenSimplex Noise python package by github user lmas:
https://github.com/lmas/opensimplex

I extracted the 2d noise function and wrapped into a 
convenience function to obtain 2d arrays of simplex noise.
The code is sped up with numba just in time compilation.

Author: Thomas Rometsch
Email: thomas.rometsch@uni-tuebingen.de
License: MIT License

"""
from ctypes import c_int64
from math import floor
import numpy as np
from numba import njit

STRETCH_CONSTANT_2D = -0.211324865405187    # (1/Math.sqrt(2+1)-1)/2
SQUISH_CONSTANT_2D = 0.366025403784439      # (Math.sqrt(2+1)-1)/2
NORM_CONSTANT_2D = 47

DEFAULT_SEED = 0

# Gradients for 2D. They approximate the directions to the
# vertices of an octagon from the center.
GRADIENTS_2D = np.array([
    5,  2,    2,  5,
    -5,  2,   -2,  5,
    5, -2,    2, -5,
    -5, -2,   -2, -5,
])


def simplexnoise(Nx, Ny, featuresize, seed=DEFAULT_SEED):
    """
    Initiate the class using a permutation array generated from a 64-bit seed number.
    """
    # Generates a proper permutation (i.e. doesn't merely perform N
    # successive pair swaps on a base array)
    # Have to zero fill so we can properly loop over it later
    perm = [0 for i in range(0, 256)]
    source = [i for i in range(0, 256)]
    seed = overflow(seed * 6364136223846793005 + 1442695040888963407)
    seed = overflow(seed * 6364136223846793005 + 1442695040888963407)
    seed = overflow(seed * 6364136223846793005 + 1442695040888963407)
    for i in range(255, -1, -1):
        seed = overflow(seed * 6364136223846793005 + 1442695040888963407)
        r = int((seed + 31) % (i + 1))
        if r < 0:
            r += i + 1
        perm[i] = source[r]
        source[r] = source[i]

    perm = np.array(perm, dtype=np.int32)

    return noise2d_array(Nx, Ny, featuresize, perm)


def overflow(x):
    # Since normal python ints and longs can be quite humongous we have to use
    # this hack to make them be able to overflow
    return c_int64(x).value


@njit(cache=True)
def extrapolate2d(xsb, ysb, dx, dy, perm):
    index = perm[(perm[xsb & 0xFF] + ysb) & 0xFF] & 0x0E
    g1 = GRADIENTS_2D[index]
    g2 = GRADIENTS_2D[index+1]
    return g1 * dx + g2 * dy


@njit(cache=True)
def noise2d_array(Nx, Ny, featuresize, perm):
    rv = np.empty((Nx, Ny))
    for nx in range(Nx):
        for ny in range(Ny):
            rv[nx, ny] = noise2d(nx/featuresize, ny/featuresize, perm)
    return rv


@njit(cache=True)
def noise2d(x, y, perm):
    """
    Generate 2D OpenSimplex noise from X,Y coordinates.
    """
    # Place input coordinates onto grid.
    stretch_offset = (x + y) * STRETCH_CONSTANT_2D
    xs = x + stretch_offset
    ys = y + stretch_offset

    # Floor to get grid coordinates of rhombus (stretched square) super-cell origin.
    xsb = floor(xs)
    ysb = floor(ys)

    # Skew out to get actual coordinates of rhombus origin. We'll need these later.
    squish_offset = (xsb + ysb) * SQUISH_CONSTANT_2D
    xb = xsb + squish_offset
    yb = ysb + squish_offset

    # Compute grid coordinates relative to rhombus origin.
    xins = xs - xsb
    yins = ys - ysb

    # Sum those together to get a value that determines which region we're in.
    in_sum = xins + yins

    # Positions relative to origin point.
    dx0 = x - xb
    dy0 = y - yb

    value = 0

    # Contribution (1,0)
    dx1 = dx0 - 1 - SQUISH_CONSTANT_2D
    dy1 = dy0 - 0 - SQUISH_CONSTANT_2D
    attn1 = 2 - dx1 * dx1 - dy1 * dy1
    if attn1 > 0:
        attn1 *= attn1
        extra = extrapolate2d(xsb + 1, ysb + 0, dx1, dy1, perm)
        value += attn1 * attn1 * extra

    # Contribution (0,1)
    dx2 = dx0 - 0 - SQUISH_CONSTANT_2D
    dy2 = dy0 - 1 - SQUISH_CONSTANT_2D
    attn2 = 2 - dx2 * dx2 - dy2 * dy2
    if attn2 > 0:
        attn2 *= attn2
        value += attn2 * attn2 * \
            extrapolate2d(xsb + 0, ysb + 1, dx2, dy2, perm)

    if in_sum <= 1:  # We're inside the triangle (2-Simplex) at (0,0)
        zins = 1 - in_sum
        # (0,0) is one of the closest two triangular vertices
        if zins > xins or zins > yins:
            if xins > yins:
                xsv_ext = xsb + 1
                ysv_ext = ysb - 1
                dx_ext = dx0 - 1
                dy_ext = dy0 + 1
            else:
                xsv_ext = xsb - 1
                ysv_ext = ysb + 1
                dx_ext = dx0 + 1
                dy_ext = dy0 - 1
        else:  # (1,0) and (0,1) are the closest two vertices.
            xsv_ext = xsb + 1
            ysv_ext = ysb + 1
            dx_ext = dx0 - 1 - 2 * SQUISH_CONSTANT_2D
            dy_ext = dy0 - 1 - 2 * SQUISH_CONSTANT_2D
    else:  # We're inside the triangle (2-Simplex) at (1,1)
        zins = 2 - in_sum
        # (0,0) is one of the closest two triangular vertices
        if zins < xins or zins < yins:
            if xins > yins:
                xsv_ext = xsb + 2
                ysv_ext = ysb + 0
                dx_ext = dx0 - 2 - 2 * SQUISH_CONSTANT_2D
                dy_ext = dy0 + 0 - 2 * SQUISH_CONSTANT_2D
            else:
                xsv_ext = xsb + 0
                ysv_ext = ysb + 2
                dx_ext = dx0 + 0 - 2 * SQUISH_CONSTANT_2D
                dy_ext = dy0 - 2 - 2 * SQUISH_CONSTANT_2D
        else:  # (1,0) and (0,1) are the closest two vertices.
            dx_ext = dx0
            dy_ext = dy0
            xsv_ext = xsb
            ysv_ext = ysb
        xsb += 1
        ysb += 1
        dx0 = dx0 - 1 - 2 * SQUISH_CONSTANT_2D
        dy0 = dy0 - 1 - 2 * SQUISH_CONSTANT_2D

    # Contribution (0,0) or (1,1)
    attn0 = 2 - dx0 * dx0 - dy0 * dy0
    if attn0 > 0:
        attn0 *= attn0
        extra = extrapolate2d(xsb, ysb, dx0, dy0, perm)
        value += attn0 * attn0 * extra

    # Extra Vertex
    attn_ext = 2 - dx_ext * dx_ext - dy_ext * dy_ext
    if attn_ext > 0:
        attn_ext *= attn_ext
        extra = extrapolate2d(xsv_ext, ysv_ext, dx_ext, dy_ext, perm)
        value += attn_ext * attn_ext * extra
    return value / NORM_CONSTANT_2D
