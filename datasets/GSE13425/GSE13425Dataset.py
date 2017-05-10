import glob
import os
import re

import pandas as pd

from datasets.Dataset import Dataset


class GSE13425Dataset(Dataset):
    def __init__(self):
        # super() call is voluntarily omitted

        filename = r = './data/GSE13425/E-GEOD-13425-processed-data-1721486034.txt'
        df = pd.read_csv(filename, sep="\t", header=None)
        df = df.dropna()  # ignore NaN values
        df = df.transpose()

        self._X = df.ix[1:, 2:].astype(float).values
        self._y = []

        self._features_names = df.ix[0, 2:].values
        self._parse_features(None)

        filenames = df[0][1:].values
        self._sample_and_data = r'./data/GSE13425/samples-and-data.csv'
        self._parse_labels(filenames)

    def _parse_labels(self, filenames):
        col_sample_name, col_sample_class = "Scan Name", "Comment [Sample_characteristics]"
        df = pd.read_csv(self._sample_and_data, sep="\t", usecols=[col_sample_name, col_sample_class])
        df = df.dropna()  # ignore NaN values

        for f in filenames:
            sample_name = f
            # get sample label name
            label = df[df[col_sample_name] == sample_name][col_sample_class].iloc[0]
            self._y.append(label)

            # Regroup similar leukemias (11 subtypes into 7 main types of leukemia)
            self._y = self._regroup_similar_leukemias(self._y)

    def _parse_features(self, filename):
        # idx -> name : nothing to do, done in __init__()
        # name -> idx
        self._features_indices = {feature: idx for (idx, feature) in enumerate(self._features_names)}

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
    "ALL-Hyperdiploid": [
        "Precursor-B  ALL, subtype: Hyperdiploid"
    ],

    "ALL-ABL": [
        "Precursor-B ALL, subtype: BCR-ABL",
        "Precursor-B ALL, subtype: BCR-ABL (+hyperdiploidy)"
    ],

    "ALL-E2A-rearranged": [
        "Precursor-B ALL, subtype: E2A-rearranged (E-sub)",
        "Precursor-B ALL, subtype: E2A-rearranged (E)",
        "Precursor-B ALL, subtype: E2A-rearranged (EP)"
    ],
    "ALL-MLL": [
        "Precursor-B ALL, subtype: MLL"
    ],

    "ALL-other": [
        "Precursor-B ALL, subtype: other"
    ],

    "ALL-TEL-AML1": [
        "Precursor-B ALL, subtype: TEL-AML1",
        "Precursor-B ALL, subtype: TEL-AML1 (+hyperdiploidy)"
    ],

    "ALL": [
        "T-ALL"
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

    ds = GSE13425Dataset()
    best_features_idx = [4, 10, 2]
    best_features_names = ds.get_features_names(best_features_idx)
    print(best_features_names)

    print("----")
    y = list(set(ds.get_y()))
    print(y)
    print(len(y))
