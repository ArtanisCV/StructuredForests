__author__ = 'artanis'

import math
import numpy as N

cimport numpy as N


ctypedef N.int32_t C_INT32
ctypedef N.float64_t C_FLOAT64


def build_feature_table(shrink, p_size, n_cell, n_ch):
    p_size /= shrink

    reg_tb = []
    for i in xrange(p_size):
        for j in xrange(p_size):
            for k in xrange(n_ch):
                reg_tb.append([i, j, k])

    half_cell_size = int(round(p_size / (2.0 * n_cell)))
    grid_pos = [int(round((i + 1) * (p_size + 2 * half_cell_size - 1) / \
                          (n_cell + 1.0) - half_cell_size))
                for i in xrange(n_cell)]
    grid_pos = [(r, c) for r in grid_pos for c in grid_pos]

    ss_tb = []
    for i in xrange(n_cell ** 2):
        for j in xrange(i + 1, n_cell ** 2):
            for z in xrange(n_ch):
                x1, y1 = grid_pos[i]
                x2, y2 = grid_pos[j]
                ss_tb.append([x1, y1, x2, y2, z])

    return N.asarray(reg_tb, dtype=N.int32), \
           N.asarray(ss_tb, dtype=N.int32)


def find_leaves(double[:, :, :] src, double[:, :, :] reg_ch,
                double[:, :, :] ss_ch,
                int shrink, int p_size, int g_size, int n_cell, int stride,
                int n_tree_eval,
                double[:, :] thrs, int[:, :] fids, int[:, :] cids):
    cdef int n_ftr_ch = reg_ch.shape[2]
    cdef int height = src.shape[0] - p_size, width = src.shape[1] - p_size
    cdef int n_tree = cids.shape[0], n_node_per_tree = cids.shape[1]
    cdef int n_reg_dim = (p_size / shrink) ** 2 * n_ftr_ch
    cdef int i, j, k, x1, x2, y1, y2, z, tree_idx, node_idx, ftr_idx
    cdef double ftr
    cdef int[:, :] reg_tb, ss_tb
    cdef N.ndarray[C_INT32, ndim=3] lids_arr

    reg_tb, ss_tb = build_feature_table(shrink, p_size, n_cell, n_ftr_ch)

    lids_arr = N.zeros((src.shape[0], src.shape[1], n_tree_eval), dtype=N.int32)
    cdef int[:, :, :] lids = lids_arr

    with nogil:
        for i from 0 <= i < height by stride:
            for j from 0 <= j < width by stride:
                for k from 0 <= k < n_tree_eval:
                    tree_idx = ((i + j) / stride % 2 * n_tree_eval + k) % n_tree
                    node_idx = 0

                    while cids[tree_idx, node_idx] != 0:
                        ftr_idx = fids[tree_idx, node_idx]

                        if ftr_idx >= n_reg_dim:
                            x1 = ss_tb[ftr_idx - n_reg_dim, 0] + i / shrink
                            y1 = ss_tb[ftr_idx - n_reg_dim, 1] + j / shrink
                            x2 = ss_tb[ftr_idx - n_reg_dim, 2] + i / shrink
                            y2 = ss_tb[ftr_idx - n_reg_dim, 3] + j / shrink
                            z = ss_tb[ftr_idx - n_reg_dim, 4]

                            ftr = ss_ch[x1, y1, z] - ss_ch[x2, y2, z]
                        else:
                            x1 = reg_tb[ftr_idx, 0] + i / shrink
                            y1 = reg_tb[ftr_idx, 1] + j / shrink
                            z = reg_tb[ftr_idx, 2]

                            ftr = reg_ch[x1, y1, z]

                        if ftr < thrs[tree_idx, node_idx]:
                            node_idx = cids[tree_idx, node_idx] - 1
                        else:
                            node_idx = cids[tree_idx, node_idx]

                    lids[i, j, k] = tree_idx * n_node_per_tree + node_idx

    return lids_arr


def build_neigh_table(g_size):
    tb = N.zeros((g_size, g_size, 4, 2), dtype=N.int32)
    dir_x = N.asarray([1, 1, -1, -1], dtype=N.int32)
    dir_y = N.asarray([1, -1, 1, -1], dtype=N.int32)

    for i in xrange(g_size):
        for j in xrange(g_size):
            for k in xrange(4):
                r = min(max(dir_x[k] + i, 0), g_size - 1)
                c = min(max(dir_y[k] + j, 0), g_size - 1)
                tb[i, j, k] = [r, c]

    return tb


def compose(double[:, :, :] src, int[:, :, :] lids,
            int p_size, int g_size, int stride, int sharpen, int n_tree_eval,
            int[:, :] cids, int[:] n_seg, int[:, :, :] segs, int[:] edge_bnds,
            int[:] edge_pts):
    cdef int height = src.shape[0] - p_size, width = src.shape[1] - p_size
    cdef int depth = src.shape[2], border = (p_size - g_size) / 2
    cdef int n_bnd = edge_bnds.shape[0] / cids.shape[0] / cids.shape[1]
    cdef int n_s, max_n_s = N.max(n_seg)
    cdef int i, j, k, m, n, p, begin, end
    cdef int leaf_idx, x1, x2, y1, y2, best_seg
    cdef double err, min_err
    cdef N.ndarray[C_FLOAT64, ndim=2] dst_arr

    cdef int[:, :] patch = N.zeros((g_size, g_size), dtype=N.int32)
    cdef double[:] count = N.zeros((max_n_s,), dtype=N.float64),
    cdef double[:, :] mean = N.zeros((max_n_s, depth), dtype=N.float64)
    cdef int[:, :, :, :] neigh_tb = build_neigh_table(g_size)

    dst_arr = N.zeros((src.shape[0], src.shape[1]), dtype=N.float64)
    cdef double[:, :] dst = dst_arr

    with nogil:
        for i from 0 <= i < height by stride:
            for j from 0 <= j < width by stride:
                for k from 0 <= k < n_tree_eval:
                    leaf_idx = lids[i, j, k]

                    begin = edge_bnds[leaf_idx * n_bnd]
                    end = edge_bnds[leaf_idx * n_bnd + sharpen + 1]
                    if begin == end:
                        continue

                    n_s = n_seg[leaf_idx]
                    if n_s == 1:
                        continue

                    patch[:, :] = segs[leaf_idx]
                    count[:] = 0.0
                    mean[:] = 0.0

                    # compute color model for each segment using every other pixel
                    for m from 0 <= m < g_size by 2:
                        for n from 0 <= n < g_size by 2:
                            count[patch[m, n]] += 1.0

                            for p from 0 <= p < depth:
                                mean[patch[m, n], p] += \
                                    src[i + m + border, j + n + border, p]

                    for m from 0 <= m < n_s:
                        for n from 0 <= n < depth:
                            mean[m, n] /= count[m]

                    # update segment according to local color values
                    end = edge_bnds[leaf_idx * n_bnd + sharpen]
                    for m from begin <= m < end:
                        min_err = 1e10
                        best_seg = -1

                        x1 = edge_pts[m] / g_size
                        y1 = edge_pts[m] % g_size

                        for n from 0 <= n < 4:
                            x2 = neigh_tb[x1, y1, n, 0]
                            y2 = neigh_tb[x1, y1, n, 1]

                            if patch[x2, y2] == best_seg:
                                continue

                            err = 0.0
                            for p from 0 <= p < depth:
                                err += (src[x1 + i + border, y1 + j + border, p] -
                                        mean[patch[x2, y2], p]) ** 2

                            if err < min_err:
                                min_err = err
                                best_seg = patch[x2, y2]

                        patch[x1, y1] = best_seg

                    # convert mask to edge maps (examining expanded set of pixels)
                    end = edge_bnds[leaf_idx * n_bnd + sharpen + 1]
                    for m from begin <= m < end:
                        x1 = edge_pts[m] / g_size
                        y1 = edge_pts[m] % g_size

                        for n from 0 <= n < 4:
                            x2 = neigh_tb[x1, y1, n, 0]
                            y2 = neigh_tb[x1, y1, n, 1]

                            if patch[x1, y1] != patch[x2, y2]:
                                dst[x1 + i, y1 + j] += 1.0
                                break

    return dst_arr


def predict_core(N.ndarray[C_FLOAT64, ndim=3] src,
                 N.ndarray[C_FLOAT64, ndim=3] reg_ch,
                 N.ndarray[C_FLOAT64, ndim=3] ss_ch,
                 int shrink, int p_size, int g_size, int n_cell,
                 int stride, int sharpen, int n_tree_eval,
                 N.ndarray[C_FLOAT64, ndim=2] thrs,
                 N.ndarray[C_INT32, ndim=2] fids,
                 N.ndarray[C_INT32, ndim=2] cids,
                 N.ndarray[C_INT32, ndim=1] n_seg,
                 N.ndarray[C_INT32, ndim=3] segs,
                 N.ndarray[C_INT32, ndim=1] edge_bnds,
                 N.ndarray[C_INT32, ndim=1] edge_pts):
    cdef int n_tree = cids.shape[0], n_node_per_tree = cids.shape[1]
    cdef int n_bnd = edge_bnds.shape[0] / n_tree / n_node_per_tree
    cdef int i, j, k, m, begin, end
    cdef int leaf_idx, loc, x1, y1
    cdef N.ndarray[C_INT32, ndim=3] lids
    cdef N.ndarray[C_FLOAT64, ndim=2] dst

    lids = find_leaves(src, reg_ch, ss_ch, shrink, p_size, g_size, n_cell,
                       stride, n_tree_eval, thrs, fids, cids)

    if sharpen == 0:
        dst = N.zeros((src.shape[0], src.shape[1]), dtype=N.float64)

        for i in xrange(0, src.shape[0] - p_size, stride):
            for j in xrange(0, src.shape[1] - p_size, stride):
                for k in xrange(n_tree_eval):
                    leaf_idx = lids[i, j, k]

                    begin = edge_bnds[leaf_idx * n_bnd]
                    end = edge_bnds[leaf_idx * n_bnd + 1]
                    if begin == end:
                        continue

                    for m in xrange(begin, end):
                        loc = edge_pts[m]
                        x1 = loc / g_size + i
                        y1 = loc % g_size + j

                        dst[x1, y1] += 1.0
    else:
        dst = compose(src, lids, p_size, g_size, stride, sharpen, n_tree_eval,
                      cids, n_seg, segs, edge_bnds, edge_pts)

    return dst


cdef inline float bilinear_interp(double[:, :] img, float x, float y) nogil:
    """
    Return img[y, x] via bilinear interpolation
    """

    cdef int h = img.shape[0], w = img.shape[1]

    if x < 0:
        x = 0
    elif x > w - 1.001:
        x = w - 1.001

    if y < 0:
        y = 0
    elif y > h - 1.001:
        y = h - 1.001

    cdef int x0 = int(x), y0 = int(y), x1 = x0 + 1, y1 = y0 + 1
    cdef double dx0 = x - x0, dy0 = y - y0, dx1 = 1 - dx0, dy1 = 1 - dy0

    return img[y0, x0] * dx1 * dy1 + img[y0, x1] * dx0 * dy1 + \
           img[y1, x0] * dx1 * dy0 + img[y1, x1] * dx0 * dy0


def non_maximum_supr(double[:, :] E0, double[:, :] O, int r, int s, double m):
    """
    Non-Maximum Suppression

    :param E0: original edge map
    :param O: orientation map
    :param r: radius for nms suppression
    :param s: radius for suppress boundaries
    :param m: multiplier for conservative suppression
    :return: suppressed edge map
    """

    cdef int h = E0.shape[0], w = E0.shape[1], x, y, d
    cdef double e, e0, co, si
    cdef N.ndarray[C_FLOAT64, ndim=2] E_arr = N.zeros((h, w), dtype=N.float64)
    cdef double[:, :] E = E_arr
    cdef double[:, :] C = N.cos(O), S = N.sin(O)

    with nogil:
        # suppress edges where edge is stronger in orthogonal direction
        for y from 0 <= y < h:
            for x from 0 <= x < w:
                e = E[y, x] = E0[y, x]
                if e == 0:
                    continue

                e *= m
                co = C[y, x]
                si = S[y, x]

                for d from -r <= d <= r:
                    if d != 0:
                        e0 = bilinear_interp(E0, x + d * co, y + d * si)
                        if e < e0:
                            E[y, x] = 0
                            break

        # suppress noisy edge estimates near boundaries
        s = w / 2 if s > w / 2 else s
        s = h / 2 if s > h / 2 else s

        for x from 0 <= x < s:
            for y from 0 <= y < h:
                E[y, x] *= x / <double>s
                E[y, w - 1 - x] *= x / <double>s

        for x from 0 <= x < w:
            for y from 0 <= y < s:
                E[y, x] *= y / <double>s
                E[h - 1 - y, x] *= y / <double>s

    return E_arr