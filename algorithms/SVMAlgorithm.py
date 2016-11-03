from sklearn import svm
from sklearn.model_selection import ParameterGrid

from algorithms.GridSearchableAlgorithm import GridSearchableAlgorithm


class SVMAlgorithm(GridSearchableAlgorithm):
    def __init__(self, dataset, gridsearch_params=None):
        super(SVMAlgorithm, self).__init__(dataset, gridsearch_params=gridsearch_params, name="SVM")

    def _init_classifier(self):
        self._clf = svm.SVC(C=100, kernel="linear", cache_size=1024)

    # def _grid_search(self):
    #     X_train = self._dataset.get_X_train()
    #     y_train = self._dataset.get_y_train()
    #
    #     X_test = self._dataset.get_X_test()
    #     y_test = self._dataset.get_y_test()
    #
    #     params = list(ParameterGrid(self._gridsearch_params))
    #
    #     best_clf_score = -1
    #
    #     for p in params:
    #         clf = svm.SVC(**p)
    #         clf.fit(X_train, y_train)
    #         score = clf.score(X_test, y_test)
    #         print("score: %.6f for params: %s" % (score, p))
    #
    #         if score > best_clf_score:
    #             best_clf_score = score
    #             self._clf = clf
    #             self._best_params = p

    def _retrieve_best_features(self):
        self._scores_by_features = sorted(enumerate(self._clf.coef_[0]), key=lambda x: x[1], reverse=True)

    def _get_best_features_by_score_unnormed(self):
        return self._scores_by_features

    def get_best_features_by_rank(self, n):
        return [x[0] for x in self._scores_by_features[:n]]

    def get_best_features(self, n):
        return self.get_best_features_by_rank(n)
