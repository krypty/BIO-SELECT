from skfeature.function.information_theoretical_based import MRMR

from algorithms.Algorithm import Algorithm, NotSupportedException
from algorithms.DeterministAlgorithm import DeterministAlgorithm


class MRMRAlgorithm(Algorithm, DeterministAlgorithm):
    def __init__(self, dataset, n=None):
        super(MRMRAlgorithm, self).__init__(dataset, n, "MRMR")

        X = dataset.get_X()
        y = dataset.get_y()
        self._idx = MRMR.mrmr(X, y, n_selected_features=n)

    def get_best_features(self):
        return self._idx

    def get_best_features_by_rank(self):
        return self._idx

    def _get_best_features_by_score_unnormed(self):
        raise NotSupportedException()
