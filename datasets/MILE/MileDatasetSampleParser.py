import pandas as pd

from datasets.DatasetSampleParser import DatasetSampleParser


class MileDatasetSampleParser(DatasetSampleParser):
    def __init__(self, filename):
        self._df = pd.read_csv(filename, sep="\t", usecols=["ID_REF", "VALUE"])
        self._df = self._df.dropna()  # ignore NaN values

        super(MileDatasetSampleParser, self).__init__(filename)

    def _parse(self):
        self._X = self._df["VALUE"].values

    def parse_features_names(self):
        self._features_names = self._df["ID_REF"].values


if __name__ == '__main__':
    import os
    os.chdir("../..")

    sample_parser = MileDatasetSampleParser(r"./data/MILE/processed/GSM329407_sample_table.txt")
    X = sample_parser.get_X()
    print(len(X))

    sample_parser.parse_features_names()
    features_names = sample_parser.get_features_names()
    print(features_names)

    # little pseudo unit test
    assert features_names[4] == "AFFX-HUMRGE/M10098_5_at"
