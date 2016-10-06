from abc import ABCMeta, abstractmethod


class DatasetSampleParser:
    __metaclass__ = ABCMeta
    """
        DatasetSample is an abstract class to parse a Dataset sample
    """

    def __init__(self, filename):
        self._X = None
        self._y = None
        self._features_names = None
        self._filename = filename  # type: str
        self._parse()

    @abstractmethod
    def _parse(self):
        """
        Parse the values and the label from the dataset
        :return: nothing but must set self._X
        """
        pass

    @abstractmethod
    def parse_features_names(self):
        """
        Parse the features names. It is not automatically called in the constructor so you must call this function once
        before calling
        :return:
        """

    def get_features_names(self):
        assert self._features_names is not None, "Features names not parsed yet, call parse_features_name() before !"
        return self._features_names

    def get_X(self):
        """
        Returns the dataset values for this sample
        :return: dataset sample values as 1D array (row)
        """
        return self._X

    def get_y(self):
        """
        Return the dataset's label for this sample
        :return: dataset's label
        """
        return self._y

    def set_y(self, y):
        """
        Set the label for this dataset sample
        :param y:
        :return:
        """
        self._y = y
