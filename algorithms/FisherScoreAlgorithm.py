from skfeature.function.similarity_based import fisher_score

from algorithms.Algorithm import Algorithm
from algorithms.DeterministAlgorithm import DeterministAlgorithm


class FisherScoreAlgorithm(Algorithm, DeterministAlgorithm):
    def __init__(self, dataset, n):
        super(FisherScoreAlgorithm, self).__init__(dataset, n, name="Fisher Score")

        X_train = self._dataset.get_X_train()
        y_train = self._dataset.get_y_train()

        score = fisher_score.fisher_score(X_train, y_train)

        self._score_by_features = sorted(enumerate(score), key=lambda p: p[1], reverse=True)

    def _get_best_features_by_score_unnormed(self):
        return self._score_by_features

    def get_best_features_by_rank(self):
        return [x[0] for x in self._score_by_features[:self._n]]

    def get_best_features(self):
        return self.get_best_features_by_rank()
