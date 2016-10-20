from abc import ABCMeta, abstractmethod

from datasets.DatasetSplitter import DatasetSplitter


class Algorithm:
    __metaclass__ = ABCMeta

    def __init__(self, dataset):
        self.dataset = dataset  # type: DatasetSplitter

    @abstractmethod
    def get_best_features_by_score(self, n):
        """
        Get the best features by score as a list of tuple (feature, score).
        This list is sorted by score with the best one as first element.
        The score is min max normed.

        :param n: the number of features to get
        :return: a score ranked list of the best n features according to this algorithm
        """
        pass

    @abstractmethod
    def get_best_features_by_rank(self, n):
        """
        Get the ranked list of the n best features
        This list is sorted by feature with the best one as first element.
        The score is min max normed.

        :param n: the number of features to get
        :return: a ranked list of the best n features according to this algorithm
        """
        pass

    @abstractmethod
    def get_best_features(self, n):
        """
        Get the unsorted list of the n best features
        This list is sorted by feature with the best one as first element.
        The score is min max normed.

        :param n: the number of features to get
        :return: a ranked list of the best n features according to this algorithm
        """
        pass
