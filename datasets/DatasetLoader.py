import os

from datasets.EGEOD22619.EGEOD22619Dataset import EGEOD22619Dataset
from datasets.Golub99.GolubDataset import GolubDataset
from datasets.MILE.MileDataset import MileDataset


class DatasetLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_from_env_var(default_dataset):
        # Load dataset from environment variable. This is used by automated scripts
        ENV_KEY = "BIOSELECT_DATASET"
        dataset_to_execute = {
            "MILE": MileDataset,
            "Golub": GolubDataset,
            "EGEOD22619": EGEOD22619Dataset
        }

        # Default dataset to load if env variable not set.
        ds_class = dataset_to_execute[default_dataset]

        try:
            ds_class = dataset_to_execute[os.environ[ENV_KEY]]
        except KeyError:
            pass

        return ds_class
