from datasets.Dataset import Dataset


class DummyDataset(Dataset):
    """
    Just a dummy dataset for the tests
    """
    def __init__(self):
        self._X = [
            [60, 190],
            [80, 180],
            [40, 150],
            [70, 170]
        ]
        self._y = ["M", "M", "M", "F"]

    def _parse_labels(self, filenames):
        pass
