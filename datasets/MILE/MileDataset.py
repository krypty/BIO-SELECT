import glob
import os
import random
import re

import pandas as pd

from datasets.Dataset import Dataset
from datasets.MILE.MileDatasetSampleParser import MileDatasetSampleParser


class MileDataset(Dataset):
    def __init__(self, samples_limit=None):
        self._dataset_folder = "data" + os.sep + "MILE"
        self._dataset_type = "processed"
        self._dataset_suffix = "_sample_table.txt"
        self._sample_and_data = self._dataset_folder + os.sep + "mile-sample-and-data.csv"

        # get all sample files in the folder
        filenames = glob.glob(
            "%s*%s" % (self._dataset_folder + os.sep + self._dataset_type + os.sep, self._dataset_suffix))

        assert len(filenames) > 0, "Samples not found !"

        if isinstance(samples_limit, int):
            filenames = random.sample(filenames, samples_limit)

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

            # Regroup similar leukemias (18 subtypes into 4 main types of leukemia)
            self._y = self._regroup_similar_leukemias(self._y)

    @staticmethod
    def _parse_sample_name(f):
        result = re.search("(GSM[\d]+)", f, re.IGNORECASE)
        # small hack : all sample name are suffixed with " 1" in the sample data csv file
        return result.group(1) + " 1"

    @staticmethod
    def _regroup_similar_leukemias(_y):
        """
        Regroup similars leukemias together. There is about 18 sub types of leukemias in this dataset. There are
        derived from the 4 main types of leukemia which are AML, CML, ALL and CLL.
        Example :

        :param _y: original classes from the dataset
        :return: regrouped classes with the 4 main types of leukemia
        """
        return map(translate_subtype_into_maintype_class, _y)


leukemia_types_lookup_table = {
    "AML": [
        "AML complex aberrant karyotype",
        "AML with inv(16)/t(16;16)",
        "AML with normal karyotype + other abnormalities",
        "AML with t(11q23)/MLL",
        "AML with t(15;17)",
        "AML with t(8;21)"
    ],

    "CML": [
        "CML"
    ],

    "ALL": [
        "ALL with hyperdiploid karyotype",
        "ALL with t(12;21)",
        "ALL with t(1;19)",
        "Pro-B-ALL with t(11q23)/MLL",
        "T-ALL",
        "c-ALL/Pre-B-ALL with t(9;22)",
        "c-ALL/Pre-B-ALL without t(9;22)",
        "mature B-ALL with t(8;14)"
    ],
    "CLL": [
        "CLL"
    ],

    "MDS": [
        "MDS"
    ],

    "Healthy": [
        "Non-leukemia and healthy bone marrow"
    ]
}


def translate_subtype_into_maintype_class(subtype):
    for k, v in leukemia_types_lookup_table.iteritems():
        if subtype in leukemia_types_lookup_table[k]:
            return k

    # in the case we do not find a matching type, we keep the original subtype
    return subtype


if __name__ == '__main__':
    os.chdir("../..")

    ds = MileDataset()
    best_features_idx = [4, 10, 2]
    best_features_names = ds.get_features_names(best_features_idx)
    print(best_features_names)

    y = list(set(ds.get_y()))
    print(y)
    print(len(y))

    t = translate_subtype_into_maintype_class("Pro-B-ALL with t(11q23)/MLL")
    print(t)
