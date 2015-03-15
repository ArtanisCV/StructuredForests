__author__ = 'artanis'

import math
import numpy as np

import pyximport
pyximport.install(build_dir=".pyxbld")
from _RandomForests import find_threshold


class RandomForests(object):
    def __init__(self, n_tree=1, n_class=None, sub_n=None, sub_f=None,
                 split='gini', min_count=1, min_child=1, max_depth=64,
                 discretize=None, rand=np.random.RandomState(123)):
        """
        m - number of trees, i.e., n_tree
        n - number of inputs
        f - number of features

        :param n_tree: [1] number of trees to train
        :param n_class: [max(labels)] number of classes
        :param sub_n: [5*n/m] number of data points for training each tree
        :param sub_f: [sqrt(f)] number features to sample for each node split
        :param split: ['gini'] options include 'gini', 'entropy' and 'twoing'
        :param min_count: [1] minimum number of data points to allow split
        :param min_child: [1] minimum number of data points allowed at child nodes
        :param max_depth: [64] maximum depth of tree
        :param discretize: optional function mapping structured to class labels
                           format: [class, best] = discretize(structured, n_class)
        :param rand: [RandomState(123)] random number generator
        """

        self.n_tree = n_tree
        self.n_class = n_class
        self.sub_n = sub_n
        self.sub_f = sub_f
        self.split = split
        self.min_count = min_count
        self.min_child = min_child
        self.max_depth = max_depth
        self.discretize = discretize
        self.rand = rand

    def train(self, ftrs, lbls):
        """
        :param ftrs: features
        :param lbls: labels
        :return: a list of trees
        """

        assert ftrs.shape[0] == lbls.shape[0]
        assert lbls.ndim == 1 or (lbls.dtype == np.int32 and
               self.discretize is not None and self.n_class is not None)
        ftrs = ftrs.astype(np.float32, copy=False)

        m = self.n_tree
        n, f = ftrs.shape
        min_child = max(1, self.min_child)
        min_count = max(1, self.min_count, self.min_child)
        n_class = np.max(lbls) + 1 if self.n_class is None else self.n_class
        sub_n = min(n, int(round(5.0 * n / m)) if self.sub_n is None else self.sub_n)
        sub_f = min(f, int(round(math.sqrt(f))) if self.sub_f is None else self.sub_f)
        split = ['gini', 'entropy', 'twoing'].index(self.split)
        forest = []

        # train M random trees on different subsets of data
        for i in xrange(m):
            if n == sub_n:
                data, hs = ftrs, lbls
            else:
                idx = self.rand.permutation(n)[:sub_n]
                data, hs = ftrs[idx], lbls[idx]

            tree = self._train_tree(data, hs, n_class, sub_f, min_count, min_child,
                                    self.max_depth, split, self.discretize)
            forest.append(tree)

        return forest

    def _train_tree(self, ftrs, lbls, n_class, sub_f, min_count, min_child,
                    max_depth, split, discretize):
        n, f = ftrs.shape
        max_n_node = 2 * n - 1

        thrs = np.zeros(max_n_node, dtype=ftrs.dtype)
        preds = np.zeros((max_n_node,) + lbls.shape[1:], dtype=lbls.dtype)
        probs = np.zeros((max_n_node, n_class), dtype=np.float64)
        fids = np.zeros(max_n_node, dtype=np.int32)
        cids = np.zeros(max_n_node, dtype=np.int32)
        counts = np.zeros(max_n_node, dtype=np.int32)
        depths = np.zeros(max_n_node, dtype=np.int32)
        dwts = np.ones(n, dtype=np.float32) / n
        dids = [None] * max_n_node

        dids[0] = np.arange(n)
        cid, max_cid = 0, 1
        while cid < max_cid:
            # get node data and store distribution
            sub_dids = dids[cid]
            sub_ftrs = ftrs[sub_dids]
            sub_lbls = lbls[sub_dids]
            sub_dwts = dwts[sub_dids]
            sub_n = sub_ftrs.shape[0]
            counts[cid] = sub_n
            dids[cid] = None

            if discretize is not None:
                sub_lbls, preds[cid] = discretize(sub_lbls, n_class)
                sub_lbls = sub_lbls.astype(np.int32, copy=False)

            assert np.all(0 <= sub_lbls) and np.all(sub_lbls < n_class)

            pure = np.all(sub_lbls[0] == sub_lbls)

            if discretize is None:
                if pure:
                    probs[cid, sub_lbls[0]] = 1
                    preds[cid] = sub_lbls[0]
                else:
                    probs[cid] = np.histogram(sub_lbls, np.arange(n_class + 1),
                                              density=True)[0]
                    preds[cid] = np.argmax(probs[cid])

            # if pure node or insufficient data don't train split
            if pure or sub_n <= min_count or depths[cid] > max_depth:
                cid += 1
                continue

            # train split and continue
            sub_fids = np.arange(f) if f <= sub_f else self.rand.permutation(f)[:sub_f]
            split_fid, thr, gain = find_threshold(
                n_class, split, sub_ftrs[:, sub_fids], sub_lbls, sub_dwts)
            split_fid = sub_fids[split_fid]
            left = sub_ftrs[:, split_fid].flatten() < thr
            n_left = np.count_nonzero(left)
            if gain > 1e-10 and n_left >= min_child and (sub_n - n_left) >= min_child:
                thrs[cid] = thr
                fids[cid] = split_fid
                cids[cid] = max_cid + 1
                depths[max_cid: max_cid + 2] = depths[cid] + 1
                dids[max_cid: max_cid + 2] = sub_dids[left], sub_dids[~left]
                max_cid += 2
            cid += 1

        ids = np.arange(max_cid)
        return thrs[ids], probs[ids], preds[ids], fids[ids], cids[ids], \
               counts[ids], depths[ids]