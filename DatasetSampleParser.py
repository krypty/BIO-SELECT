from abc import ABCMeta, abstractmethod


class DatasetSampleParser(metaclass=ABCMeta):
    """
        DatasetSample is an abstract class to parse a Dataset sample
    """

    def __init__(self, filename):
        self._X = None
        self._y = None
        self._parse(filename)

    @abstractmethod
    def _parse(self, filename):
        """
        Parse the values and the label from the dataset
        :param filename: to parse
        :return: nothing but must set self._X
        """
        pass

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
