from abc import ABCMeta, abstractmethod

import numpy as np


class Dataset(metaclass=ABCMeta):
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

    def _parse_data(self, filenames):
        for f in filenames:
            dss = self._sample_parser(f)
            self._X.append(dss.get_X())

        # convert 2d-array into np array
        self._X = np.array(self._X)

    @abstractmethod
    def _parse_labels(self, filenames):
        pass

    def get_X(self):
        return self._X

    def get_y(self):
        return self._y
