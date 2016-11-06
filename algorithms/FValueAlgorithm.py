from sklearn.feature_selection import f_classif

from algorithms.Algorithm import Algorithm


class FValueAlgorithm(Algorithm):
    def __init__(self, dataset, n):
        super(FValueAlgorithm, self).__init__(dataset, n, name="F Value")

        X_train = self._dataset.get_X_train()
        y_train = self._dataset.get_y_train()

        F, pvalues = f_classif(X_train, y_train)
        self._f_sorted = sorted(enumerate(F), key=lambda x: x[1], reverse=True)

    def _get_best_features_by_score_unnormed(self):
        return self._f_sorted

    def get_best_features_by_rank(self):
        return [x[0] for x in self._f_sorted[:self._n]]

    def get_best_features(self):
        return self.get_best_features_by_rank()
