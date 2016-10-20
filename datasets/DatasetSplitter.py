from sklearn.model_selection import train_test_split

from datasets.Dataset import Dataset


class DatasetSplitter:
    def __init__(self, dataset, test_size):
        self._dataset = dataset  # type: Dataset
        self._test_size = test_size  # type: float

        self._split()

    def _split(self):
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._dataset.get_X(),
                                                                                    self._dataset.get_y(),
                                                                                    test_size=self._test_size)

    def get_X(self):
        return self._dataset.get_X()

    def get_y(self):
        return self._dataset.get_y()

    def get_X_train(self):
        return self._X_train

    def get_y_train(self):
        return self._y_train

    def get_X_test(self):
        return self._X_test

    def get_y_test(self):
        return self._y_test

    def get_features_names(self, features_idx):
        return self._dataset.get_features_names(features_idx)
