__author__ = 'artanis'

import os
import sys
import tables
import cv2
import numpy as N
from math import floor, ceil, log
from scipy.ndimage.morphology import distance_transform_edt
from BaseStructuredForests import BaseStructuredForest
from RandomForests import RandomForest
from RobustPCA import robust_pca
from utils import conv_tri, gradient

import pyximport
pyximport.install(setup_args={'include_dirs': N.get_include()})
from _StructuredForests import predict_core


class StructuredForest(BaseStructuredForest):
    def __init__(self, options, model_dir='model/',
                 rand=N.random.RandomState(123)):
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
            g_size: size of ground truth patches
            n_cell: number of self similarity cells

            n_pos: number of positive patches per tree
            n_neg: number of negative patches per tree
            fraction: fraction of features to use to train each tree
            n_tree: number of trees in forest to train
            n_class: number of classes (clusters) for binary splits
            min_count: minimum number of data points to allow split
            min_child: minimum number of data points allowed at child nodes
            max_depth: maximum depth of tree
            split: options include 'gini', 'entropy' and 'twoing'
            discretize: optional function mapping structured to class labels

            stride: stride at which to compute edges
            sharpen: sharpening amount (can only decrease after training)
            n_tree_eval: number of trees to evaluate per location

        :param model_dir: directory for model
            A trained model will contain
            thrs: threshold corresponding to each feature index
            fids: feature indices for each node
            cids: indices of children for each node
            edge_bnds: begin / end of edge points for each node
            edge_pts: edge points for each node
            n_seg: number of segmentations for each node
            segs: segmentation map for each node

        :param rand: random number generator
        """

        BaseStructuredForest.__init__(self, options)
        assert self.options["stride"] % self.options["shrink"] == 0

        self.model_dir = model_dir
        self.data_dir = os.path.join(self.model_dir, "data")
        self.tree_dir = os.path.join(self.model_dir, "trees")
        self.forest_dir = os.path.join(self.model_dir, "forests")
        self.data_prefix = "data_"
        self.tree_prefix = "tree_"
        self.forest_name = "forest.h5"

        try:
            model_file = os.path.join(self.forest_dir, self.forest_name)
            self.model = self.load_model(model_file)
        except:
            self.model = {}
            print >> sys.stderr, "No model file found. Training is required."

        self.rand = rand

    def load_model(self, model_file):
        with tables.open_file(model_file) as mfile:
            model = {
                "thrs": mfile.get_node("/thrs")[:],
                "fids": mfile.get_node("/fids")[:],
                "cids": mfile.get_node("/cids")[:],
                "edge_bnds": mfile.get_node("/edge_bnds")[:].flatten(),
                "edge_pts": mfile.get_node("/edge_pts")[:].flatten(),
                "n_seg": mfile.get_node("/n_seg")[:].flatten(),
                "segs": mfile.get_node("/segs")[:],
            }

        return model

    def predict(self, src):
        stride = self.options["stride"]
        sharpen = self.options["sharpen"]
        shrink = self.options["shrink"]
        p_size = self.options["p_size"]
        g_size = self.options["g_size"]
        n_cell = self.options["n_cell"]
        n_tree_eval = self.options["n_tree_eval"]
        thrs = self.model["thrs"]
        fids = self.model["fids"]
        cids = self.model["cids"]
        edge_bnds = self.model["edge_bnds"]
        edge_pts = self.model["edge_pts"]
        n_seg = self.model["n_seg"]
        segs = self.model["segs"]
        p_rad = p_size / 2
        g_rad = g_size / 2

        pad = cv2.copyMakeBorder(src, p_rad, p_rad, p_rad, p_rad,
                                 borderType=cv2.BORDER_REFLECT)

        reg_ftr, ss_ftr = self.get_shrunk_channels(pad)

        if sharpen != 0:
            pad = conv_tri(pad, 1)

        dst = predict_core(pad, reg_ftr, ss_ftr, shrink, p_size, g_size,
                           n_cell, stride, sharpen, n_tree_eval, thrs, fids,
                           cids, n_seg, segs, edge_bnds, edge_pts)

        if sharpen == 0:
            alpha = 2.6 * stride ** 2 / g_size ** 2 / n_tree_eval
        elif sharpen == 1:
            alpha = 1.8 * stride ** 2 / g_size ** 2 / n_tree_eval
        else:
            alpha = 1.4 * stride ** 2 / g_size ** 2 / n_tree_eval

        dst = N.minimum(dst * alpha, 1.0)

        return conv_tri(dst, 1)[g_rad: src.shape[0] + g_rad,
                                g_rad: src.shape[1] + g_rad]

    def train(self, input_data):
        self.prepare_data(input_data)
        self.train_tree()
        self.merge_trees()

    def prepare_data(self, input_data):
        """
        Prepare data for model training
        """

        n_img = len(input_data)

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        n_tree = self.options["n_tree"]
        n_pos = self.options["n_pos"]
        n_neg = self.options["n_neg"]
        fraction = self.options["fraction"]
        p_size = self.options["p_size"]
        g_size = self.options["g_size"]
        shrink = self.options["shrink"]
        p_rad, g_rad = p_size / 2, g_size / 2
        n_ftr_dim = N.sum(self.get_ftr_dim())
        n_smp_ftr_dim = int(n_ftr_dim * fraction)
        rand = self.rand

        for i in xrange(n_tree):
            data_file = self.data_prefix + str(i + 1) + ".h5"
            data_path = os.path.join(self.data_dir, data_file)
            if os.path.exists(data_path):
                print "Found Data %d '%s', reusing..." % ((i + 1), data_file)
                continue

            ftrs = N.zeros((n_pos + n_neg, n_smp_ftr_dim), dtype=N.float32)
            lbls = N.zeros((n_pos + n_neg, g_size, g_size), dtype=N.int32)
            sids = rand.permutation(n_ftr_dim)[:n_smp_ftr_dim]
            total = 0

            for j, (img, bnds, segs) in enumerate(input_data):
                mask = N.zeros(bnds[0].shape, dtype=bnds[0].dtype)
                mask[::shrink, ::shrink] = 1
                mask[:p_rad] = mask[-p_rad:] = 0
                mask[:, :p_rad] = mask[:, -p_rad:] = 0

                n_pos_per_gt = int(ceil(float(n_pos) / n_img / len(bnds)))
                n_neg_per_gt = int(ceil(float(n_neg) / n_img / len(bnds)))

                for k, boundary in enumerate(bnds):
                    dis = distance_transform_edt(boundary == 0)

                    pos_loc = ((dis < g_rad) * mask).nonzero()
                    pos_loc = zip(pos_loc[0].tolist(), pos_loc[1].tolist())
                    pos_loc = [pos_loc[item] for item in
                               rand.permutation(len(pos_loc))[:n_pos_per_gt]]

                    neg_loc = ((dis >= g_rad) * mask).nonzero()
                    neg_loc = zip(neg_loc[0].tolist(), neg_loc[1].tolist())
                    neg_loc = [neg_loc[item] for item in
                               rand.permutation(len(neg_loc))[:n_neg_per_gt]]

                    loc = pos_loc + neg_loc
                    n_loc = min(len(loc), ftrs.shape[0] - total)
                    loc = [loc[item] for item in rand.permutation(len(loc))[:n_loc]]
                    if n_loc == 0:
                        continue

                    ftr = N.concatenate(self.get_features(img, loc), axis=1)
                    assert ftr.shape[1] == n_ftr_dim
                    ftr = ftr[:, sids]

                    lbl = N.zeros((ftr.shape[0], g_size, g_size), dtype=N.int8)
                    for m, (x, y) in enumerate(loc):
                        sub = segs[k][x - g_rad: x + g_rad, y - g_rad: y + g_rad]
                        sub = N.unique(sub, return_inverse=True)[1]
                        lbl[m] = sub.reshape((g_size, g_size))

                    ftrs[total: total + n_loc] = ftr
                    lbls[total: total + n_loc] = lbl
                    total += n_loc

                sys.stdout.write("Processing Data %d: %d/%d\r" % (i + 1, j + 1, n_img))
                sys.stdout.flush()
            print

            with tables.open_file(data_path, "w") as dfile:
                dfile.create_array('/', 'ftrs', ftrs[:total])
                dfile.create_array('/', 'lbls', lbls[:total])
                dfile.create_array('/', 'sids', sids.astype(N.int32))
            print "Saving %d samples to '%s'..." % (total, data_file)

    def train_tree(self):
        """
        Train a single tree
        """

        n_tree = self.options["n_tree"]

        if not os.path.exists(self.tree_dir):
            os.makedirs(self.tree_dir)

        rf = RandomForest(n_class=self.options["n_class"],
                          min_count=self.options["min_count"],
                          min_child=self.options["min_child"],
                          max_depth=self.options["max_depth"],
                          split=self.options["split"],
                          discretize=self.options["discretize"],
                          rand=self.rand)

        for i in xrange(n_tree):
            data_file = self.data_prefix + str(i + 1) + ".h5"
            data_path = os.path.join(self.data_dir, data_file)
            tree_file = self.tree_prefix + str(i + 1) + ".h5"
            tree_path = os.path.join(self.tree_dir, tree_file)
            if os.path.exists(tree_path):
                print "Found Tree %d '%s', reusing..." % ((i + 1), tree_file)
                continue

            with tables.open_file(data_path) as dfile:
                ftrs = dfile.get_node('/ftrs')[:]
                lbls = dfile.get_node('/lbls')[:]
                sids = dfile.get_node('/sids')[:]

                forest = rf.train(ftrs, lbls)
                thrs, probs, preds, fids, cids, counts, depths = forest[0]
                fids[cids > 0] = sids[fids[cids > 0]]

                with tables.open_file(tree_path, 'w') as tfile:
                    tfile.create_array('/', 'fids', fids)
                    tfile.create_array('/', 'thrs', thrs)
                    tfile.create_array('/', 'cids', cids)
                    tfile.create_array('/', 'probs', probs)
                    tfile.create_array('/', 'segs', preds)
                    tfile.create_array('/', 'counts', counts)
                    tfile.create_array('/', 'depths', depths)
                    tfile.close()

                sys.stdout.write("Processing Tree %d/%d\r" % (i + 1, n_tree))
                sys.stdout.flush()
            print

    def merge_trees(self):
        """
        Accumulate trees and merge into final model
        """

        n_tree = self.options["n_tree"]
        g_size = self.options["g_size"]

        if not os.path.exists(self.forest_dir):
            os.makedirs(self.forest_dir)

        forest_path = os.path.join(self.forest_dir, self.forest_name)
        if os.path.exists(forest_path):
            print "Found model, reusing..."
            self.model = self.load_model(forest_path)
            return

        trees = []
        for i in xrange(n_tree):
            tree_file = self.tree_prefix + str(i + 1) + ".h5"
            tree_path = os.path.join(self.tree_dir, tree_file)

            with tables.open_file(tree_path) as mfile:
                tree = {'fids': mfile.get_node('/fids')[:],
                        'thrs': mfile.get_node('/thrs')[:],
                        'cids': mfile.get_node('/cids')[:],
                        'segs': mfile.get_node('/segs')[:]}
            trees.append(tree)

        max_n_node = 0
        for i in xrange(n_tree):
            max_n_node = max(max_n_node, trees[i]["fids"].shape[0])

        # merge all fields of all trees
        thrs = N.zeros((n_tree, max_n_node), dtype=N.float64)
        fids = N.zeros((n_tree, max_n_node), dtype=N.int32)
        cids = N.zeros((n_tree, max_n_node), dtype=N.int32)
        segs = N.zeros((n_tree, max_n_node, g_size, g_size), dtype=N.int32)
        for i in xrange(n_tree):
            tree = trees[i]
            n_node = tree["fids"].shape[0]
            thrs[i, :n_node] = tree["thrs"].flatten()
            fids[i, :n_node] = tree["fids"].flatten()
            cids[i, :n_node] = tree["cids"].flatten()
            segs[i, :n_node] = tree["segs"]

        # remove very small segments (<=5 pixels)
        n_seg = N.max(segs.reshape((n_tree, max_n_node, g_size ** 2)), axis=2) + 1
        for i in xrange(n_tree):
            for j in xrange(max_n_node):
                m = n_seg[i, j]
                if m <= 1:
                    continue

                S = segs[i, j]
                remove = False

                for k in xrange(m):
                    Sk = (S == k)
                    if N.count_nonzero(Sk) > 5:
                        continue

                    S[Sk] = N.median(S[conv_tri(Sk.astype(N.float64), 1) > 0])
                    remove = True

                if remove:
                    S = N.unique(S, return_inverse=True)[1]
                    segs[i, j] = S.reshape((g_size, g_size))
                    n_seg[i, j] = N.max(S) + 1

        # store compact representations of sparse binary edge patches
        n_bnd = self.options["sharpen"] + 1
        edge_pts = []
        edge_bnds = N.zeros((n_tree, max_n_node, n_bnd), dtype=N.int32)
        for i in xrange(n_tree):
            for j in xrange(max_n_node):
                if cids[i, j] != 0 or n_seg[i, j] <= 1:
                    continue

                E = gradient(segs[i, j].astype(N.float64), norm_radius=0,
                             norm_const=0.005)[0] > .01
                E0 = 0

                for k in xrange(n_bnd):
                    r, c = N.nonzero(E & (~ E0))
                    edge_pts += [r[m] * g_size + c[m] for m in xrange(len(r))]
                    edge_bnds[i, j, k] = len(r)

                    E0 = E
                    E = conv_tri(E.astype(N.float64), 1) > 0.01

        segs = segs.reshape((-1, segs.shape[-2], segs.shape[-1]))
        edge_pts = N.asarray(edge_pts, dtype=N.int32)
        edge_bnds = N.hstack(([0], N.cumsum(edge_bnds.flatten()))).astype(N.int32)

        with tables.open_file(forest_path, 'w') as mfile:
            mfile.create_array('/', 'thrs', thrs)
            mfile.create_array('/', 'fids', fids)
            mfile.create_array('/', 'cids', cids)
            mfile.create_array('/', 'edge_bnds', edge_bnds)
            mfile.create_array('/', 'edge_pts', edge_pts)
            mfile.create_array('/', 'n_seg', n_seg)
            mfile.create_array('/', 'segs', segs)
            mfile.close()

        self.model = self.load_model(forest_path)


def discretize(segs, n_class, n_sample, rand):
    """
    Convert a set of segmentations into a set of labels in [0, n_class - 1]

    :param segs: segmentations
    :param n_class: number of classes (clusters) for binary splits
    :param n_sample: number of samples for clustering structured labels
    :param rand: random number generator
    """

    w = segs[0].shape[0]
    segs = segs.reshape((segs.shape[0], w ** 2))

    # compute all possible lookup inds for w x w patches
    ids = N.arange(w ** 4, dtype=N.float64)
    ids1 = N.floor(ids / w / w)
    ids2 = ids - ids1 * w * w
    kp = ids2 > ids1
    ids1 = ids1[kp]
    ids2 = ids2[kp]

    # compute n binary codes zs of length nSamples
    n_sample = min(n_sample, ids1.shape[0])
    kp = rand.permutation(ids1.shape[0])[:n_sample]
    n = segs.shape[0]
    ids1 = ids1[kp].astype(N.int32)
    ids2 = ids2[kp].astype(N.int32)

    zs = N.zeros((n, n_sample), dtype=N.float64)
    for i in xrange(n):
        zs[i] = (segs[i][ids1] == segs[i][ids2])
    zs -= N.mean(zs, axis=0)
    zs = zs[:, N.any(zs, axis=0)]

    if N.count_nonzero(zs) == 0:
        lbls = N.ones(n, dtype=N.int32)
        segs = segs[0]
    else:
        # find most representative segs (closest to mean)
        ind = N.argmin(N.sum(zs * zs, axis=1))
        segs = segs[ind]

        # discretize zs by discretizing pca dimensions
        d = min(5, n_sample, int(floor(log(n_class, 2))))
        zs = robust_pca(zs, d, rand=rand)[0]
        lbls = N.zeros(n, dtype=N.int32)
        for i in xrange(d):
            lbls += (zs[:, i] < 0).astype(N.int32) * 2 ** i
        lbls = N.unique(lbls, return_inverse=True)[1].astype(N.int32)

    return lbls, segs.reshape((-1, w, w))


def bsds500_train(input_root):
    import scipy.io as SIO
    from skimage import img_as_float
    from skimage.io import imread

    dataset_dir = os.path.join(input_root, "BSDS500", "data")
    image_dir = os.path.join(dataset_dir, "images", "train")
    label_dir = os.path.join(dataset_dir, "groundTruth", "train")
    data = []

    for file_name in os.listdir(label_dir):
        gts = SIO.loadmat(os.path.join(label_dir, file_name))
        gts = gts["groundTruth"].flatten()
        bnds = [gt["Boundaries"][0, 0] for gt in gts]
        segs = [gt["Segmentation"][0, 0] for gt in gts]

        img = imread(os.path.join(image_dir, file_name[:-3] + "jpg"))
        img = img_as_float(img)

        data.append((img, bnds, segs))

    return data


def bsds500_test(input_root, output_root):
    from skimage import img_as_float
    from skimage.io import imread, imsave

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    image_dir = os.path.join(input_root, "BSDS500", "data", "images", "test")
    file_names = filter(lambda name: name[-3:] == 'jpg', os.listdir(image_dir))
    n_image = len(file_names)

    for i, file_name in enumerate(file_names):
        img = img_as_float(imread(os.path.join(image_dir, file_name)))

        edge = model.predict(img)

        imsave(os.path.join(output_root, file_name[:-3] + "png"), edge)

        sys.stdout.write("Processing Image %d/%d\r" % (i + 1, n_image))
        sys.stdout.flush()
    print


if __name__ == "__main__":
    rand = N.random.RandomState(1)

    options = {
        "rgbd": 0,
        "shrink": 2,
        "n_orient": 4,
        "grd_smooth_rad": 0,
        "grd_norm_rad": 4,
        "reg_smooth_rad": 2,
        "ss_smooth_rad": 8,
        "p_size": 32,
        "g_size": 16,
        "n_cell": 5,

        "n_pos": 10000,
        "n_neg": 10000,
        "fraction": 0.25,
        "n_tree": 8,
        "n_class": 2,
        "min_count": 1,
        "min_child": 8,
        "max_depth": 64,
        "split": 'gini',
        "discretize": lambda lbls, n_class:
            discretize(lbls, n_class, n_sample=256, rand=rand),

        "stride": 2,
        "sharpen": 2,
        "n_tree_eval": 4,
    }

    model = StructuredForest(options, rand=rand)
    model.train(bsds500_train("toy"))
    bsds500_test("toy", "edges")
