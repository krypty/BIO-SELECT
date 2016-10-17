import pandas as pd

from datasets.DatasetSampleParser import DatasetSampleParser


class GolubDatasetSampleParser(DatasetSampleParser):
    def __init__(self, filename):
        self._df = pd.read_csv(filename, sep="\t", usecols=["ID_REF", "VALUE"])
        self._df = self._df.dropna()  # ignore NaN values

        super(GolubDatasetSampleParser, self).__init__(filename)

    def _parse(self):
        self._X = self._df["VALUE"].values

    def parse_features_names(self):
        self._features_names = self._df["ID_REF"].values
