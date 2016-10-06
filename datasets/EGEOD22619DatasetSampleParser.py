import pandas as pd

from datasets.DatasetSampleParser import DatasetSampleParser


class EGEOD22619DatasetSampleParser(DatasetSampleParser):
    def __init__(self, filename):
        self._df = pd.read_csv(filename, sep="\t", usecols=["ID_REF", "VALUE"])
        self._df = self._df.dropna()  # ignore NaN values

        super(EGEOD22619DatasetSampleParser, self).__init__(filename)

    def _parse(self):
        self._X = self._df["VALUE"].values

    def parse_features_names(self):
        self._features_names = self._df["ID_REF"].values


if __name__ == '__main__':
    # execute this function at the root project folder
    import os

    os.chdir("..")

    sample_parser = EGEOD22619DatasetSampleParser(r"data/E-GEOD-22619/processed/GSM560961_sample_table.txt")
    X = sample_parser.get_X()
    print(len(X))

    sample_parser.parse_features_names()
    features_names = sample_parser.get_features_names()
    print(features_names)

    # little pseudo unit test
    assert features_names[4] == "1255_g_at"
