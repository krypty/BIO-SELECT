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


if __name__ == '__main__':
    import os

    os.chdir("../..")

    sample_parser = GolubDatasetSampleParser(r"./data/golub99/processed/test/sample_0_ALL.csv")
    X = sample_parser.get_X()
    print(len(X))

    sample_parser.parse_features_names()
    features_names = sample_parser.get_features_names()
    print(features_names)

    # little pseudo unit test
    assert features_names[4] == "AFFX-BioC-3_at"
