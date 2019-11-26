# -*- coding: utf-8 -*-

import csv
import itertools
from operator import itemgetter

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Renumber NBER category. 0 is reserved as PAD.
NBER_CATEGORY = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}

# Renumber NBER subcategory. 0 is reserved as PAD.
NBER_SUBCATEGORY = {11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 19: 6, 21: 7, 22: 8,
                    23: 9, 24: 10, 25: 11, 31: 12, 32: 13, 33: 14, 39: 15,
                    41: 16, 42: 17, 43: 18, 44: 19, 45: 20, 46: 21, 49: 22,
                    51: 23, 52: 24, 53: 25, 54: 26, 55: 27, 59: 28, 61: 29,
                    62: 30, 63: 31, 64: 32, 65: 33, 66: 34, 67: 35, 68: 36,
                    69: 37, 70: 38}

# Path configuration
data_config = {'pat_ts': ['dataset/patent_time.csv', float],
               'pat_cat': ['dataset/patent_nber_category.csv', int],
               'pat_subcat': ['dataset/patent_nber_subcategory.csv', int],
               'assignee_ts': ['dataset/assignee_time.csv', float],
               'inventor_ts': ['dataset/inventor_time.csv', float]}
keys = ['pat_ts', 'pat_cat', 'pat_subcat', 'assignee_ts', 'inventor_ts']
ts_keys = ['pat_ts', 'assignee_ts', 'inventor_ts']
assert(set(ts_keys) <= set(keys) <= set(data_config.keys()))
eps = torch.tensor(1e-12)  # Replace zero with this small number


# =============================================================================
# ============================== Define dataset ===============================
# =============================================================================

def load_csv(max_len):
    """Load raw csv file. Read 5 csv files configured in data_config.

    Returns
    -------
    dict
        {'time series name': a list of time series}. Each time series is
        a list.

    """

    rets = {}
    for key, config in data_config.items():
        if key in ['pat_ts', 'pat_cat', 'pat_subcat']:
            rets[key] = [[config[1](e) for e in row[:max_len]]
                         for row in csv.reader(open(config[0], 'r'))]
        else:
            rets[key] = [[config[1](e) for e in row[-max_len:]]
                         for row in csv.reader(open(config[0], 'r'))]
    return rets


def timestamp_normalization(streams):
    """Normalize timestamp in the time series.

    Parameters
    ----------
    streams : dict
        Five data streams.

    Returns
    -------
        Normalized streams.

    """

    ts = list(itertools.chain(*itertools.chain.from_iterable(
        [streams[key] for key in ts_keys])))
    maxt, mint = np.max(ts), np.min(ts)
    maxr = maxt - mint
    for key in ts_keys:
        streams[key] = [[(y - mint) / maxr for y in s] for s in streams[key]]
    return streams


def split_train_test(total_num, tr_ratio, keep_same=True):
    """Get train idx and test idx.

    Parameters
    ----------
    total_num : int
        Total number of points in the dataset.
    tr_ratio : float
        Ratio of the training set vs test set.
    keep_same : bool
        True to keep same split every time, False otherwise.

    """

    if keep_same:  # Keep shuffle same every time
        np.random.seed(42)
    idx = np.arange(total_num)
    np.random.shuffle(idx)
    num_train = int(total_num * tr_ratio)
    return idx[:num_train].tolist(), idx[num_train:].tolist()


class PatentDataset(Dataset):
    """Load patent datset including patent granted time (pts), patent category
    (pcat), patent subcategory (psubcat), assignee citation time (ats),
    inventor citation time (its)."""

    def __init__(self, dataset, ob_ratio, use_category=False):
        super(PatentDataset, self).__init__()
        self.pts, self.pcat, self.psubcat, self.ats, self.its = dataset
        self.ob_ratio = ob_ratio
        self.use_category = use_category

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, ix):
        """Patent sequences are splited to encoder and decoder. All assignee
        and inventor sequences are used as encoder.

        Returns
        -------
        patent time series source side, patent time series target side, patent
        category info source side, patent category info target side, assignee
        time series, inventor time series.

        """

        pts, pcat, psubcat = self.pts[ix], self.pcat[ix], self.psubcat[ix]
        len1, len2, len3 = len(pts), len(pcat), len(psubcat)
        assert (len1 == len2 == len3)
        ats, its = self.ats[ix], self.its[ix]
        # mark = NBER category or NBER subcategory
        category = NBER_CATEGORY if self.use_category else NBER_SUBCATEGORY
        pcat = pcat if self.use_category else psubcat
        # Split data for encoder and decoder
        elen = max(2, int(self.ob_ratio * len1))  # ob window > 2, has 1 tgt
        src_pts, tgt_pts = pts[:elen], pts[elen:]
        src_pcat = [category[c] for c in pcat[:elen]]  # Renumbering
        tgt_pcat = [category[c] for c in pcat[elen:]]  # Renumbering
        return src_pts, tgt_pts, src_pcat, tgt_pcat, ats, its


def load_dataset(args, tr_ratio=.8, norm=True, max_len=200):
    """Load dataset given path.

    Workflow: load_scv -> timestamp_normalization -> split_train_test ->
    PatentDataset

    Parameters
    ----------
    args : dict
        Arguments.
    tr_ratio : float
        Ratio of dataset used as training set.
    norm : bool
        True to normalize time, False otherwise.

    """

    streams = load_csv(max_len)
    if norm:  # Normalization timestamp
        streams = timestamp_normalization(streams)
    # Get training and test set
    train_idx, test_idx = split_train_test(len(streams[keys[0]]), tr_ratio)
    train = [itemgetter(*train_idx)(streams[key]) for key in keys]
    test = [itemgetter(*test_idx)(streams[key]) for key in keys]
    # Build Dataset object
    train_set = PatentDataset(train, args.ob_ratio,
                              use_category=args.use_category)
    test_set = PatentDataset(test, args.ob_ratio,
                             use_category=args.use_category)
    ncats = len(NBER_CATEGORY) if args.use_category else len(NBER_SUBCATEGORY)
    return train_set, test_set, ncats

# =============================================================================
# ============================= Handle minibatch ==============================
# =============================================================================


def collate_fn(insts):
    """Handle six streams in the minibatch.

    Six streams included: src_pts, tgt_pts, src_pcat, tgt_pcat, ats, its. Note
    that the first element is the target patent.

    Pytorch LSTM needs ordered source side data. longest -> shortest. Then we
    need to sort patent series, assignee series, and inventor series,
    respectively.

    Parameters
    ----------
    A minibatch in which patent, assignee, and inventor series are sorted
    respectively.

    """

    # Sorted patent stream
    insts = sorted(insts, key=lambda k: len(k[0]), reverse=True)
    src_pts, tgt_pts, src_pcat, tgt_pcat, ats, its = list(zip(*insts))
    # Category. Drop target point.
    src_pcat = pad_sequence([torch.tensor(seq[1:], dtype=torch.long)
                             for seq in src_pcat])
    tgt_pcat = pad_sequence([torch.tensor(seq, dtype=torch.long)
                             for seq in tgt_pcat])
    mask = (tgt_pcat > 0)  # 0 is for pad
    # Interval. Target point of sequence is dropped
    src_pts_delta = [
            torch.max((torch.tensor(seq)[1:] - torch.tensor(seq)[:-1]), eps)
            for seq in src_pts]
    length = torch.tensor([s.size(0) for s in src_pts_delta], dtype=torch.long)
    src_pts_delta = pad_sequence(src_pts_delta)
    tgt_pts_delta = pad_sequence([
        torch.tensor(seq) - torch.tensor([src_pts[ix][-1]] + seq[:-1])
        for ix, seq in enumerate(tgt_pts)])
    # Sorted assignee time series
    sort_ats = sorted(enumerate(ats), key=lambda x: len(x[1]), reverse=True)
    aorg_idx, sort_ats = list(zip(*sort_ats))
    sort_ats = [  # Delta
            torch.max(torch.tensor(seq[1:]) - torch.tensor(seq[:-1]), eps)
            for seq in sort_ats]
    alength = torch.tensor([len(e) for e in sort_ats])
    sort_ats = pad_sequence(sort_ats)
    # Sorted inventor time series
    sort_its = sorted(enumerate(its), key=lambda x: len(x[1]), reverse=True)
    iorg_idx, sort_its = list(zip(*sort_its))
    sort_its = [  # Delta
            torch.max(torch.tensor(seq[1:]) - torch.tensor(seq[:-1]), eps)
            for seq in sort_its]
    ilength = torch.tensor([len(e) for e in sort_its])
    sort_its = pad_sequence(sort_its)
    return src_pts_delta, tgt_pts_delta, src_pcat, tgt_pcat, length, mask, \
        sort_ats, aorg_idx, alength, sort_its, iorg_idx, ilength
