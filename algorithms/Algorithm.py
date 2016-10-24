from abc import ABCMeta, abstractmethod

import numpy as np

from datasets.DatasetSplitter import DatasetSplitter


class Algorithm:
    __metaclass__ = ABCMeta

    @staticmethod
    def _normalize_scores(features_by_score):
        def min_max_norm(X):
            min_x = np.min(X)
            return (X - min_x) / (np.max(X) - min_x)

        features, scores = zip(*features_by_score)
        scores = min_max_norm(scores)
        return zip(features, scores)

    def __init__(self, dataset):
        self._dataset = dataset  # type: DatasetSplitter

    def get_best_features_by_score(self, n):
        # type: (object) -> list
        """
        Get the best features by score as a list of tuple (feature, score).
        This list is sorted by score with the best one as first element.
        The score is min max normed.
        Example : list((2000, 0.7), (3000, 0.3))

        :param n: the number of features to get
        :return: a score ranked list of the best n features according to this algorithm
        """
        return self._normalize_scores(self._get_best_features_by_score_unnormed())[:n]

    @abstractmethod
    def get_best_features_by_rank(self, n):
        # type: (object) -> list
        """
        Get the ranked list of the n best features
        This list is sorted by feature with the best one as first element.
        Example: list(2000, 3000)

        :param n: the number of features to get
        :return: a ranked list of the best n features according to this algorithm
        """
        pass

    @abstractmethod
    def get_best_features(self, n):
        # type: (object) -> list
        """
        Get the unsorted list of the n best features
        This list is sorted by feature with the best one as first element.
        Example: list(2000, 3000) or list(3000, 2000)

        :param n: the number of features to get
        :return: a ranked list of the best n features according to this algorithm
        """
        pass

    @abstractmethod
    def _get_best_features_by_score_unnormed(self):
        pass
