from algorithms.Algorithm import Algorithm
from skfeature.function.similarity_based import reliefF


class ReliefFAlgorithm(Algorithm):
    def __init__(self, dataset):
        super(ReliefFAlgorithm, self).__init__(dataset)

        X_train = self._dataset.get_X_train()
        y_train = self._dataset.get_y_train()

        scores = reliefF.reliefF(X_train, y_train)

        self._scores_per_features = sorted(enumerate(scores), key=lambda p: p[1], reverse=True)

    def _get_best_features_by_score_unnormed(self):
        super(ReliefFAlgorithm, self)._get_best_features_by_score_unnormed()
        return self._scores_per_features

    def get_best_features_by_rank(self, n):
        super(ReliefFAlgorithm, self).get_best_features_by_rank(n)
        return [x[0] for x in self._scores_per_features[:n]]

    def get_best_features(self, n):
        super(ReliefFAlgorithm, self).get_best_features(n)
        return self.get_best_features_by_rank(n)
