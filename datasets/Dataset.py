from abc import ABCMeta, abstractmethod

import numpy as np

from datasets.DatasetSampleParser import DatasetSampleParser


class Dataset:
    __metaclass__ = ABCMeta
    """
        Dataset is an abstract class to combine a list of DatasetSample into a feature matrix
        input : filenames of the datasets
        output : a (Ngenes as cols, Msamples as rows) matrix
    """

    def __init__(self, filenames, sample_parser):
        """

        :param filenames: filenames of the dataset to parse
        :param sample_parser: class derived from DatasetParser to use to parse each DatasetSample
        """
        self._sample_parser = sample_parser  # type: DatasetSampleParser
        self._X = []
        self._y = []
        self._parse_data(filenames)
        self._parse_labels(filenames)
        self._parse_features(filenames[0])

    def _parse_data(self, filenames):
        for f in filenames:
            dss = self._sample_parser(f)
            self._X.append(dss.get_X())

        # convert 2d-array into np array
        self._X = np.array(self._X)

    def _parse_features(self, filename):
        dss = self._sample_parser(filename)
        dss.parse_features_names()

        # idx -> name
        self._features_names = dss.get_features_names()
        # name -> idx
        self._features_indices = {feature: idx for (idx, feature) in enumerate(self._features_names)}

    @abstractmethod
    def _parse_labels(self, filenames):
        pass

    def get_X(self):
        return self._X

    def get_y(self):
        return self._y

    def get_features_names(self, features_idx):
        """
        Return the features names given their indices in the dataset
        :param features_idx: features indices as an array
        :return: array with the name of the given features
        """
        return [self._features_names[idx] for idx in features_idx]

    def get_features_indices(self, features_names):
        """
        Return the features indices given their names in the dataset
        :param features_names: features names as an array
        :return: array with the indices of the given features names
        """
        return [self._features_indices[name] for name in features_names]
