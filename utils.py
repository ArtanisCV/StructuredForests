__author__ = 'artanis'

import cv2
import numpy as N

import pyximport
pyximport.install(build_dir=".pyxbld",
                  setup_args={"include_dirs": N.get_include()})
from _utils import histogram_core, pdist_core


def resize(src, size):
    assert len(size) == 2
    size = (int(size[0]), int(size[1]))

    if size == src.shape[:2]:
        return src
    elif size[0] < src.shape[0] and size[1] < src.shape[1]:
        return cv2.resize(src, size[::-1], interpolation=cv2.INTER_AREA)
    else:
        return cv2.resize(src, size[::-1], interpolation=cv2.INTER_LINEAR)


def conv_tri(src, radius):
    """
    Image convolution with a triangle filter.

    :param src: input image
    :param radius: gradient normalization radius
    :return: convolution result
    """

    if radius == 0:
        return src
    elif radius <= 1:
        p = 12.0 / radius / (radius + 2) - 2
        kernel = N.asarray([1, p, 1], dtype=N.float64) / (p + 2)
        return cv2.sepFilter2D(src, ddepth=-1, kernelX=kernel, kernelY=kernel,
                               borderType=cv2.BORDER_REFLECT)
    else:
        radius = int(radius)
        kernel = range(1, radius + 1) + [radius + 1] + range(radius, 0, -1)
        kernel = N.asarray(kernel, dtype=N.float64) / (radius + 1) ** 2
        return cv2.sepFilter2D(src, ddepth=-1, kernelX=kernel, kernelY=kernel,
                               borderType=cv2.BORDER_REFLECT)


def rgb2luv(src):
    """
    This function implements rgb to luv conversion in a way similar to UCSD
    computer vision toolbox.
    """

    assert src.dtype == N.float64 or src.dtype == N.float32
    assert src.ndim == 3 and src.shape[-1] == 3

    a = 29.0 ** 3 / 27
    y0 = 8.0 / a
    maxi = 1.0 / 270

    table = [i / 1024.0 for i in xrange(1025)]
    table = [116 * y ** (1.0 / 3.0) - 16 if y > y0 else y * a for y in table]
    table = [l * maxi for l in table]
    table += [table[-1]] * 39

    rgb2xyz_mat = N.asarray([[0.430574, 0.222015, 0.020183],
                             [0.341550, 0.706655, 0.129553],
                             [0.178325, 0.071330, 0.939180]])
    xyz = N.dot(src, rgb2xyz_mat)
    nz = 1.0 / (xyz[:, :, 0] + 15 * xyz[:, :, 1] + 3 * xyz[:, :, 2] + 1e-35)

    L = [table[int(1024 * item)] for item in xyz[:, :, 1].flatten()]
    L = N.asarray(L).reshape(xyz.shape[:2])
    u = L * (13 * 4 * xyz[:, :, 0] * nz - 13 * 0.197833) + 88 * maxi
    v = L * (13 * 9 * xyz[:, :, 1] * nz - 13 * 0.468331) + 134 * maxi

    luv = N.concatenate((L[:, :, None], u[:, :, None], v[:, :, None]), axis=2)
    return luv.astype(src.dtype, copy=False)


def gradient(src, norm_radius=0, norm_const=0.01):
    """
    Compute gradient magnitude and orientation at each image location.

    :param src: input image
    :param norm_radius: normalization radius (no normalization if 0)
    :param norm_const: normalization constant
    :return: gradient magnitude and orientation (0 ~ pi)
    """

    if src.ndim == 2:
        src = src[:, :, None]

    dx = N.zeros(src.shape, dtype=src.dtype)
    dy = N.zeros(src.shape, dtype=src.dtype)
    for i in xrange(src.shape[2]):
        dy[:, :, i], dx[:, :, i] = N.gradient(src[:, :, i])

    magnitude = N.sqrt(dx ** 2 + dy ** 2)
    idx_2 = N.argmax(magnitude, axis=2)
    idx_0, idx_1 = N.indices(magnitude.shape[:2])
    magnitude = magnitude[idx_0, idx_1, idx_2]
    if norm_radius != 0:
        magnitude /= conv_tri(magnitude, norm_radius) + norm_const
    magnitude = magnitude.astype(src.dtype, copy=False)

    dx = dx[idx_0, idx_1, idx_2]
    dy = dy[idx_0, idx_1, idx_2]
    orientation = N.arctan2(dy, dx)
    orientation[orientation < 0] += N.pi
    orientation[N.abs(dx) + N.abs(dy) < 1e-5] = 0.5 * N.pi
    orientation = orientation.astype(src.dtype, copy=False)

    return magnitude, orientation


def histogram(magnitude, orientation, downscale, n_orient, interp=False):
    """
    Compute oriented gradient histograms.

    :param magnitude: gradient magnitude
    :param orientation: gradient orientation
    :param downscale: spatially downscaling factor
    :param n_orient: number of orientation bins
    :param interp: true for interpolation over orientations
    :return: oriented gradient histogram
    """

    dtype = magnitude.dtype
    magnitude = magnitude.astype(N.float64, copy=False)
    orientation = orientation.astype(N.float64, copy=False)

    hist = histogram_core(magnitude, orientation, downscale, n_orient, interp)
    return hist.astype(dtype, copy=False)


def pdist(points):
    """
    Compute the pairwise differences between n-dimensional points in a way
    specified in the paper "Structured Forests for Fast Edge Detection".
    Note: Indeed this is not a valid distance measurement (asymmetry).

    :param points: n-dimensional points
    :return: pairwise differences
    """

    dtype = points.dtype
    prefix_shape, (n_pt, n_dim) = points.shape[:-2], points.shape[-2:]
    src = points.reshape((-1, n_pt, n_dim)).astype(N.float64, copy=False)

    dst = pdist_core(src).astype(dtype, copy=False)
    return dst.reshape(prefix_shape + (n_pt * (n_pt - 1) / 2, n_dim))