from algorithms.Algorithm import Algorithm
from skfeature.function.similarity_based import reliefF


class ReliefFAlgorithm(Algorithm):
    def __init__(self, dataset, n):
        super(ReliefFAlgorithm, self).__init__(dataset, n, name="ReliefF")

        X_train = self._dataset.get_X_train()
        y_train = self._dataset.get_y_train()

        scores = reliefF.reliefF(X_train, y_train)

        self._scores_per_features = sorted(enumerate(scores), key=lambda p: p[1], reverse=True)

    def _get_best_features_by_score_unnormed(self):
        super(ReliefFAlgorithm, self)._get_best_features_by_score_unnormed()
        return self._scores_per_features

    def get_best_features_by_rank(self):
        super(ReliefFAlgorithm, self).get_best_features_by_rank()
        return [x[0] for x in self._scores_per_features[:self._n]]

    def get_best_features(self):
        super(ReliefFAlgorithm, self).get_best_features()
        return self.get_best_features_by_rank()
