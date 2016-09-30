import pandas as pd

from datasets.DatasetSampleParser import DatasetSampleParser


class MileDatasetSampleParser(DatasetSampleParser):
    def _parse(self, filename):
        df = pd.read_csv(filename, sep="\t", usecols=["VALUE"])
        df = df.dropna()  # ignore NaN values

        self._X = df["VALUE"].values


if __name__ == '__main__':
    sample_parser = MileDatasetSampleParser(r"./data/MILE/processed/GSM329407_sample_table.txt")
    X = sample_parser.get_X()
    print(len(X))
