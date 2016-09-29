import glob
import os

import re

from Dataset import Dataset
from MileDatasetSampleParser import MileDatasetSampleParser

import pandas as pd


class MileDataset(Dataset):
    def __init__(self, full_dataset=False):
        self._dataset_folder = "datasets" + os.sep + "MILE"
        self._dataset_type = "processed"
        self._dataset_suffix = "_sample_table.txt"
        filenames = []
        self._sample_and_data = self._dataset_folder + os.sep + "mile-sample-and-data.csv"

        if full_dataset is True:
            # get all sample files in the folder
            filenames = glob.glob(
                "%s*%s" % (self._dataset_folder + os.sep + self._dataset_type + os.sep, self._dataset_suffix))

            # debug only
            filenames = filenames[:100]
        else:
            self._datasets = [
                "GSM331662 1",
                "GSM331663 1",
                "GSM331664 1",
                "GSM329407 1",
                "GSM329408 1",
                "GSM329409 1",
                "GSM329410 1",
                "GSM329411 1",
                "GSM329412 1"
            ]

            filenames = self._get_full_filenames()

        super(MileDataset, self).__init__(filenames, sample_parser=MileDatasetSampleParser)

    def _get_full_filenames(self):
        # "GSM331662 1" -> /<full>/<path>/<to>/<dataset>/GSM331662<suffix>
        filenames = [self._dataset_folder + os.sep + self._dataset_type + os.sep + ds_name[:-2] + self._dataset_suffix
                     for
                     ds_name in
                     self._datasets]
        return filenames

    def _parse_labels(self, filenames):
        df = pd.read_csv(self._sample_and_data, sep="\t", usecols=["Source Name", "Leukemia description"])
        df = df.dropna()  # ignore NaN values

        for f in filenames:
            sample_name = self._parse_sample_name(f)
            # get sample label name
            label = df[df["Source Name"] == sample_name]["Leukemia description"].iloc[0]
            self._y.append(label)

    @staticmethod
    def _parse_sample_name(f):
        result = re.search("(GSM[\d]+)", f, re.IGNORECASE)
        # small hack : all sample name are suffixed with " 1" in the sample data csv file
        return result.group(1) + " 1"
