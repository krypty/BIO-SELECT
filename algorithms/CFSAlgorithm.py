from skfeature.function.statistical_based import CFS

from algorithms.Algorithm import Algorithm, NotSupportedException
from algorithms.DeterministAlgorithm import DeterministAlgorithm


class CFSAlgorithm(Algorithm, DeterministAlgorithm):
    def __init__(self, dataset, n=None):
        super(CFSAlgorithm, self).__init__(dataset, n, "CFS")

        if n is not None:
            print("[warning] n argument will be ignored")

        X = dataset.get_X()
        y = dataset.get_y()
        self._F = CFS.cfs(X, y)

    def get_best_features(self):
        return self._F

    def get_best_features_by_rank(self):
        raise NotSupportedException()

    def _get_best_features_by_score_unnormed(self):
        raise NotSupportedException()
