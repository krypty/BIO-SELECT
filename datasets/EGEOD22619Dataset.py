import glob
import os
import re

import pandas as pd

from datasets.Dataset import Dataset
from datasets.EGEOD22619DatasetSampleParser import EGEOD22619DatasetSampleParser


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
            self._y.append(label)

    @staticmethod
    def _parse_sample_name(f):
        result = re.search("(GSM[\d]+)", f, re.IGNORECASE)
        # small hack : all sample name are suffixed with " 1" in the sample data csv file
        return result.group(1) + " 1"
