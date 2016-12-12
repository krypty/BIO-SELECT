from abc import ABCMeta, abstractmethod

import numpy as np

from datasets.DatasetSplitter import DatasetSplitter


class Algorithm:
    __metaclass__ = ABCMeta

    @staticmethod
    def _normalize_scores(features_by_score):
        def min_max_norm(X):
            min_x = np.min(X)
            return (X - min_x) / (np.max(X) - float(min_x))

        features, scores = zip(*features_by_score)
        scores = min_max_norm(scores)
        return zip(features, scores)

    def __init__(self, dataset, n, name):
        """
        :param dataset: the dataset
        :param n: the number of features to get
        :param name: name of the algorithm
        """
        self._dataset = dataset  # type: DatasetSplitter
        self._n = n  # type: int
        self._name = name  # type: str

    def get_best_features_by_score(self):
        # type: (object) -> list
        """
        Get the best features by score as a list of tuple (feature, score).
        This list is sorted by score with the best one as first element.
        The score is min max normed.
        Example : list((2000, 0.7), (3000, 0.3))

        :return: a score ranked list of the best n features according to this algorithm
        """
        n_features_to_keep = min(len(self._get_best_features_by_score_unnormed()), self._n)
        return self._normalize_scores(self._get_best_features_by_score_unnormed())[:n_features_to_keep]

    @abstractmethod
    def get_best_features_by_rank(self):
        # type: (object) -> list
        """
        Get the ranked list of the n best features
        This list is sorted by feature with the best one as first element.
        Example: list(2000, 3000)

        :return: a ranked list of the best n features according to this algorithm
        """
        pass

    @abstractmethod
    def get_best_features(self):
        # type: (object) -> list
        """
        Get the unsorted list of the n best features
        This list is sorted by feature with the best one as first element.
        Example: list(2000, 3000) or list(3000, 2000)

        :return: a ranked list of the best n features according to this algorithm
        """
        pass

    @abstractmethod
    def _get_best_features_by_score_unnormed(self):
        pass

    def get_score(self):
        """
        Returns the score of the algorithm. This can be used to balance the list of feature provided.
        For example, if the score is low then the list might not be as convincing as wanted.
        :return: the score of the algorithm
        """
        raise NotSupportedException()

    @property
    def name(self):
        return self._name


class NotSupportedException(Exception):
    def __init__(self):
        super(NotSupportedException, self).__init__("This algorithm does not supported this function")
