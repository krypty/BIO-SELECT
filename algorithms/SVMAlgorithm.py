from sklearn import svm

from algorithms.Algorithm import Algorithm


class SVMAlgorithm(Algorithm):
    def __init__(self, dataset):
        super(SVMAlgorithm, self).__init__(dataset, name="SVM")

        X_train = self._dataset.get_X_train()
        y_train = self._dataset.get_y_train()

        clf = svm.SVC(C=1, kernel="linear", cache_size=1024)
        clf.fit(X_train, y_train)

        self._scores_by_features = sorted(enumerate(clf.coef_[0]), key=lambda x: x[1], reverse=True)

    def _get_best_features_by_score_unnormed(self):
        return self._scores_by_features

    def get_best_features_by_rank(self, n):
        return [x[0] for x in self._scores_by_features[:n]]

    def get_best_features(self, n):
        return self.get_best_features_by_rank(n)
