from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from algorithms.Algorithm import Algorithm


class SVMRFEAlgorithm(Algorithm):
    def __init__(self, dataset, n):
        super(SVMRFEAlgorithm, self).__init__(dataset, n, name="SVM RFE")

        X_train = self._dataset.get_X_train()
        y_train = self._dataset.get_y_train()
        X_test = self._dataset.get_X_test()
        y_test = self._dataset.get_y_test()

        estimator = SVR(kernel="linear", gamma="auto", C=10, cache_size=2000)
        self._clf = RFE(estimator, n_features_to_select=self._n, step=0.2, verbose=True)
        self._clf = self._clf.fit(X_train, y_train)

        # list of tuple, [0] is the feature index, [1] is the rank (higher is better)
        # we inverted the rank since the best features are ranked with increasing values (starting at 1)
        max_rank = max(self._clf.ranking_)
        inverted_rank = [max_rank - rank for rank in self._clf.ranking_]
        self._scores_per_features = sorted(enumerate(inverted_rank), key=lambda x: x[1], reverse=True)

        self._score = self._clf.score(X_test, y_test)

    def get_score(self):
        return self._score

    def _get_best_features_by_score_unnormed(self):
        super(SVMRFEAlgorithm, self)._get_best_features_by_score_unnormed()
        return self._scores_per_features

    def get_best_features_by_rank(self):
        super(SVMRFEAlgorithm, self).get_best_features_by_rank()
        return [feat_by_score[0] for feat_by_score in self._scores_per_features[:self._n]]

    def get_best_features(self):
        super(SVMRFEAlgorithm, self).get_best_features()
        return self.get_best_features_by_rank()
