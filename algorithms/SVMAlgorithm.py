from sklearn import svm

from algorithms.GridSearchableAlgorithm import GridSearchableAlgorithm


class SVMAlgorithm(GridSearchableAlgorithm):
    def __init__(self, dataset, gridsearch_params=None):
        super(SVMAlgorithm, self).__init__(dataset, gridsearch_params=gridsearch_params, name="SVM")

    def _init_classifier(self):
        self._clf = svm.SVC(C=100, kernel="linear", cache_size=1024)

    def _retrieve_best_features(self):
        self._scores_by_features = sorted(enumerate(self._clf.coef_[0]), key=lambda x: x[1], reverse=True)

    def _get_best_features_by_score_unnormed(self):
        return self._scores_by_features

    def get_best_features_by_rank(self, n):
        return [x[0] for x in self._scores_by_features[:n]]

    def get_best_features(self, n):
        return self.get_best_features_by_rank(n)
