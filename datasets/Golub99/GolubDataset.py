import glob
import os
import random
import re

import pandas as pd

from datasets.Dataset import Dataset
from datasets.Golub99.GolubDatasetSampleParser import GolubDatasetSampleParser


class GolubDataset(Dataset):
    def __init__(self):
        self._dataset_folder = "data" + os.sep + "golub99"
        self._dataset_type = "processed"
        self._dataset_suffix = ".csv"
        filenames = []
        self._sample_and_data = None

        # Train and test sets are scrambled into one unique dataset. Splitting is done later
        # train set
        path = "%s*%s" % (
            self._dataset_folder + os.sep + self._dataset_type + os.sep + "train" + os.sep, self._dataset_suffix)
        filenames += glob.glob(path)

        # test set
        path = "%s*%s" % (
            self._dataset_folder + os.sep + self._dataset_type + os.sep + "test" + os.sep, self._dataset_suffix)
        filenames += glob.glob(path)

        super(GolubDataset, self).__init__(filenames, sample_parser=GolubDatasetSampleParser)

    def _parse_labels(self, filenames):
        for f in filenames:
            # get sample label name
            label = self._get_label_from_filename(f)
            label = self._binarize_label(label)
            self._y.append(label)

    @staticmethod
    def _binarize_label(label):
        return 0 if label == "ALL" else 1

    @staticmethod
    def _get_label_from_filename(f):
        pattern = r'\/sample_[\d]+_([\w]+)\.csv'
        result = re.search(pattern, f, re.IGNORECASE)
        return result.group(1)


if __name__ == '__main__':
    os.chdir("../..")

    ds = GolubDataset()
    best_features_idx = [4, 10, 2]
    best_features_names = ds.get_features_names(best_features_idx)

    print(best_features_names)

    assert best_features_names[0] == "AFFX-BioC-3_at"
    assert best_features_names[1] == "AFFX-BioB-M_st"
    assert best_features_names[2] == "AFFX-BioB-3_at"

    y = list(set(ds.get_y()))
    print(y)
    print(len(y))
