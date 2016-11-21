from imblearn.over_sampling import RandomOverSampler

from datasets.Dataset import Dataset


class DatasetBalancer:
    def __init__(self, dataset):
        self._ds = dataset  # type: Dataset
        self._balancer = RandomOverSampler()

    def balance(self):
        """
        Balance the classes of the dataset using a random oversampling
        :return: the dataset oversampled
        """
        self._ds._X, self._ds._y = self._balancer.fit_sample(self._ds.get_X(), self._ds.get_y())
        return self._ds
