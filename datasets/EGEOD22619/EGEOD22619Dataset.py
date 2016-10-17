import glob
import re

import pandas as pd

import os

from datasets.Dataset import Dataset
from datasets.EGEOD22619.EGEOD22619DatasetSampleParser import EGEOD22619DatasetSampleParser


class EGEOD22619Dataset(Dataset):
    def __init__(self):
        self._dataset_folder = "data" + os.sep + "E-GEOD-22619"
        self._dataset_type = "processed"
        self._dataset_suffix = "_sample_table.txt"
        self._sample_and_data = self._dataset_folder + os.sep + "E-GEOD-22619.sdrf.csv"

        # get all sample files in the folder
        filenames = glob.glob(
            "%s*%s" % (self._dataset_folder + os.sep + self._dataset_type + os.sep, self._dataset_suffix))

        super(EGEOD22619Dataset, self).__init__(filenames, sample_parser=EGEOD22619DatasetSampleParser)

    def _parse_labels(self, filenames):
        col_sample_name = "Source Name"
        col_label = "Characteristics[individual]"

        df = pd.read_csv(self._sample_and_data, sep="\t", usecols=[col_sample_name, col_label])
        df = df.dropna()  # ignore NaN values

        for f in filenames:
            sample_name = self._parse_sample_name(f)
            # get sample label name
            label = df[df[col_sample_name] == sample_name][col_label].iloc[0]
            # label = self._convert_label_to_int(label)
            self._y.append(label)

    @staticmethod
    def _parse_sample_name(f):
        result = re.search("(GSM[\d]+)", f, re.IGNORECASE)
        # small hack : all sample name are suffixed with " 1" in the sample data csv file
        return result.group(1) + " 1"

    @staticmethod
    def _convert_label_to_int(label):
        return 1 if label == "diseased" else 0


if __name__ == '__main__':
    # execute this function at the root project folder
    import os
    os.chdir("..")

    ds = EGEOD22619Dataset()
    best_features_idx = [4, 10, 2]
    best_features_names = ds.get_features_names(best_features_idx)
    print(best_features_names)
