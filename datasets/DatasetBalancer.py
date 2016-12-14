from imblearn.over_sampling import RandomOverSampler

from datasets.Dataset import Dataset


class DatasetBalancer:
    def __init__(self, dataset):
        self._ds = dataset  # type: Dataset or DatasetSplitter
        self._balancer = RandomOverSampler()

    def balance(self):
        """
        Balance the classes of the dataset using a random oversampling.
        If the dataset has been split, then the balancer will only balance the train set
        leaving the whole dataset get_X() and get_y() unchanged
        Otherwise the balancer will balance the whole dataset.
        :return: the dataset oversampled
        """
        try:
            self._ds._X_train, self._ds._y_train = self._balancer.fit_sample(self._ds.get_X_train(),
                                                                             self._ds.get_y_train())
        except AttributeError:
            self._ds._X, self._ds._y = self._balancer.fit_sample(self._ds.get_X(), self._ds.get_y())
        return self._ds
