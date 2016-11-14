import os

from datasets.DatasetLoader import DatasetLoader
from datasets.Golub99.GolubDataset import GolubDataset


class TestDatasetLoader:
    def test_load_from_env_var(self):
        os.environ["BIOSELECT_DATASET"] = "Golub"
        ds_class = DatasetLoader.load_from_env_var(default_dataset="MILE")

        assert ds_class.__name__ == GolubDataset.__name__

    def test_load_from_env_var_no_var(self):
        ds_class = DatasetLoader.load_from_env_var(default_dataset="Golub")

        assert ds_class.__name__ == GolubDataset.__name__

    def test_load_from_env_var_no_default_dataset(self):
        try:
            ds_class = DatasetLoader.load_from_env_var()
            assert False
        except TypeError:
            assert True
