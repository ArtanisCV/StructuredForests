__author__ = 'artanis'

import cv2
import numpy as N
from skimage.util import view_as_windows
from utils import resize, conv_tri, rgb2luv, gradient, histogram, pdist


class BaseStructuredForests(object):
    def __init__(self, options):
        """
        :param options:
            rgbd: 0 for RGB, 1 for RGB + depth
            shrink: amount to shrink channels
            n_orient: number of orientations per gradient scale
            grd_smooth_rad: radius for image gradient smoothing
            grd_norm_rad: radius for gradient normalization
            reg_smooth_rad: radius for reg channel smoothing
            ss_smooth_rad: radius for sim channel smoothing
            p_size: size of image patches
            n_cell: number of self similarity cells
        """

        self.options = options
        assert self.options["p_size"] % 2 == 0

    def get_ftr_dim(self):
        shrink = self.options["shrink"]
        p_size = self.options["p_size"]
        n_cell = self.options["n_cell"]

        n_color_ch = 3 if self.options["rgbd"] == 0 else 4
        n_grad_ch = 2 * (1 + self.options["n_orient"])
        n_ch = n_color_ch + n_grad_ch

        reg_ftr_dim = (p_size / shrink) ** 2 * n_ch
        ss_ftr_dim = n_cell ** 2 * (n_cell ** 2 - 1) / 2 * n_ch

        return reg_ftr_dim, ss_ftr_dim

    def get_shrunk_channels(self, src):
        shrink = self.options["shrink"]
        n_orient = self.options["n_orient"]
        grd_smooth_rad = self.options["grd_smooth_rad"]
        grd_norm_rad = self.options["grd_norm_rad"]

        luv = rgb2luv(src)
        size = (luv.shape[0] / shrink, luv.shape[1] / shrink)
        channels = [resize(luv, size)]

        for scale in [1.0, 0.5]:
            img = resize(luv, (luv.shape[0] * scale, luv.shape[1] * scale))
            img = conv_tri(img, grd_smooth_rad)

            magnitude, orientation = gradient(img, grd_norm_rad)

            downscale = max(1, int(shrink * scale))
            hist = histogram(magnitude, orientation, downscale, n_orient)

            channels.append(resize(magnitude, size)[:, :, None])
            channels.append(resize(hist, size))

        channels = N.concatenate(channels, axis=2)

        reg_smooth_rad = self.options["reg_smooth_rad"] / float(shrink)
        ss_smooth_rad = self.options["ss_smooth_rad"] / float(shrink)

        if reg_smooth_rad > 1.0:
            reg_ch = conv_tri(channels, int(round(reg_smooth_rad)))
        else:
            reg_ch = conv_tri(channels, reg_smooth_rad)

        if ss_smooth_rad > 1.0:
            ss_ch = conv_tri(channels, int(round(ss_smooth_rad)))
        else:
            ss_ch = conv_tri(channels, ss_smooth_rad)

        return reg_ch, ss_ch

    def get_shrunk_loc(self, pos):
        shrink = self.options["shrink"]

        return [(r / shrink, c / shrink) for r, c in pos]

    def get_reg_ftr(self, channels, smp_loc=None):
        """
        Compute regular features.

        :param channels: shrunk channels for regular features
        :param smp_loc: shrunk sample locations (None for all)
        :return: regular features
        """

        shrink = self.options["shrink"]
        p_size = self.options["p_size"] / shrink
        n_r, n_c, n_ch = channels.shape

        reg_ftr = view_as_windows(channels, (p_size, p_size, n_ch))
        reg_ftr = reg_ftr.reshape((n_r - p_size + 1, n_c - p_size + 1,
                                   p_size ** 2 * n_ch))

        if smp_loc is not None:
            r_pos = [r - p_size / 2 for r, _ in smp_loc]
            c_pos = [c - p_size / 2 for _, c in smp_loc]
            reg_ftr = reg_ftr[r_pos, c_pos]

        return reg_ftr

    def get_ss_ftr(self, channels, smp_loc=None):
        """
        Compute self-similarity features

        :param channels: shrunk channels for self-similarity features
        :param smp_loc: shrunk sample locations (None for all)
        :return: self-similarity features
        """

        shrink = self.options["shrink"]
        p_size = self.options["p_size"] / shrink
        n_r, n_c, n_ch = channels.shape

        ss_ftr = view_as_windows(channels, (p_size, p_size, n_ch))

        if smp_loc is not None:
            ss_ftr = ss_ftr.reshape((n_r - p_size + 1, n_c - p_size + 1,
                                     p_size ** 2, n_ch))
            r_pos = [r - p_size / 2 for r, _ in smp_loc]
            c_pos = [c - p_size / 2 for _, c in smp_loc]
            ss_ftr = ss_ftr[r_pos, c_pos]
        else:
            ss_ftr = ss_ftr.reshape((-1, p_size ** 2, n_ch))

        n_cell = self.options["n_cell"]
        half_cell_size = int(round(p_size / (2.0 * n_cell)))
        grid_pos = [int(round((i + 1) * (p_size + 2 * half_cell_size - 1) / \
                              (n_cell + 1.0) - half_cell_size))
                    for i in xrange(n_cell)]
        grid_pos = [r * p_size + c for r in grid_pos for c in grid_pos]
        ss_ftr = ss_ftr[:, grid_pos]

        ss_ftr = pdist(ss_ftr)
        return ss_ftr.reshape((ss_ftr.shape[0], -1))

    def get_features(self, src, smp_loc):
        bottom, right = (4 - src.shape[0] % 4) % 4, (4 - src.shape[1] % 4) % 4
        src = cv2.copyMakeBorder(src, 0, bottom, 0, right,
                                 borderType=cv2.BORDER_REFLECT)

        reg_ch, ss_ch = self.get_shrunk_channels(src)
        smp_loc = self.get_shrunk_loc(smp_loc)

        reg_ftr = self.get_reg_ftr(reg_ch, smp_loc)
        ss_ftr = self.get_ss_ftr(ss_ch, smp_loc)

        return reg_ftr, ss_ftr