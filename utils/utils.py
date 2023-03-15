from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

from scipy.optimize import fsolve


def split_dates(split: float = None, start_date: datetime = None, end_date: datetime = None):
    start_train_date = start_date
    end_test_date = end_date
    end_train_date = (start_train_date + (end_test_date - start_train_date) * split).replace(hour=0, minute=0, second=0,
                                                                                             microsecond=0,
                                                                                             nanosecond=0)
    start_test_date = end_train_date + timedelta(hours=24)
    return Set(idx=1, start=start_train_date, end=end_train_date), Set(idx=1, start=start_test_date, end=end_test_date)


@dataclass
class Set:
    idx: int
    start: datetime
    end: datetime


@dataclass
class Walk:
    train: Set
    valid: Set
    test: Set


class WalkForward:
    def __init__(self,
                 start_date: datetime,
                 end_date: datetime,
                 n_walks: int = 5,
                 val_size: float = 0,
                 test_size: float = 0
                 ):

        self.start_date = start_date
        self.end_date = end_date
        self.n_walks = n_walks
        total_days = (self.end_date - self.start_date).days
        days_by_walk = int(fsolve(self.to_solve, 1, args=(val_size, self.n_walks, total_days)))
        self.train_days = int(days_by_walk * (1 - val_size - test_size))
        self.valid_days = int(days_by_walk * val_size)
        self.test_days = int(days_by_walk * test_size)

    @staticmethod
    def to_solve(x, test_size, n_walks, total_days):
        return x + (x * test_size) * (n_walks - 1) - total_days

    def get_walks(self, verbose: bool = False):
        start_train = self.start_date
        idx = 0
        start_valid = start_train + timedelta(days=self.train_days)
        start_test = start_valid + timedelta(days=self.valid_days)
        end_test = start_test + timedelta(days=self.test_days) + timedelta(days=-1)
        while (end_test < self.end_date) and (self.n_walks is None or idx < self.n_walks):
            idx = idx + 1
            walk = Walk(train=Set(idx=idx, start=start_train, end=start_valid + timedelta(days=-1)),
                        valid=Set(idx=idx, start=start_valid, end=start_test + timedelta(days=-1)),
                        test=Set(idx=idx, start=start_test, end=np.min([end_test, self.end_date])))
            if verbose:
                print('*' * 20, f'{idx}th walking forward', '*' * 20)
                print(f'Training: {walk.train.start} to {walk.train.end}')
                print(f'Validation: {walk.valid.start} to {walk.valid.end}')
                if self.test_days != 0: print(f'Testing: {walk.test.start} to {walk.test.end}')
            yield idx, walk
            start_train = start_train + timedelta(days=self.valid_days)
            start_valid = start_train + timedelta(days=self.train_days)
            start_test = start_valid + timedelta(days=self.valid_days)
            end_test = start_test + timedelta(days=self.test_days) + timedelta(days=-1)


def print_features_corr(X):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(X).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=X.columns.tolist(), ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    fig.tight_layout()
    plt.show()
