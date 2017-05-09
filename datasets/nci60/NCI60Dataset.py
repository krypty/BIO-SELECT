import glob
import os
import re

import pandas as pd

from datasets.Dataset import Dataset


class NCI60Dataset(Dataset):
    def __init__(self):
        # super() call is voluntarily omitted

        filename = r'./data/nci60/nci60data.csv'
        df = pd.read_csv(filename, sep="\t")
        df = df.dropna()  # ignore NaN values

        self._X = df.ix[:, :-1].values
        self._y = df.ix[:, -1].values

        self._parse_features(None)

    def _parse_labels(self, filenames):
        pass  # processing done in __init__()

    def _parse_features(self, filename):
        # idx -> name
        self._features_names = ["data.{}".format(i + 1) for i in range(self._X.shape[0])]
        # name -> idx
        self._features_indices = {feature: idx for (idx, feature) in enumerate(self._features_names)}


if __name__ == '__main__':
    os.chdir("../..")

    ds = NCI60Dataset()
    best_features_idx = [4, 10, 2]
    best_features_names = ds.get_features_names(best_features_idx)
    print(best_features_names)

    print("----")
    y = list(set(ds.get_y()))
    print(y)
    print(len(y))
