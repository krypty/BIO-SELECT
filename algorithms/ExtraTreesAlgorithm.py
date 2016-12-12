from sklearn.ensemble import ExtraTreesClassifier
from algorithms.GridSearchableAlgorithm import GridSearchableAlgorithm
import numpy as np


class ExtraTreesAlgorithm(GridSearchableAlgorithm):
    def __init__(self, dataset, n, gridsearch_params=None):
        super(ExtraTreesAlgorithm, self).__init__(dataset, n, gridsearch_params=gridsearch_params, name="ExtraTrees")

    def _init_classifier(self):
        self._clf = ExtraTreesClassifier(n_jobs=2, n_estimators=100)

    def _retrieve_best_features(self):
        importances = self._clf.feature_importances_

        indices = np.argsort(importances)[::-1]

        n_features = self._dataset.get_X().shape[1]

        feat_importances = [(indices[f], importances[indices[f]]) for f in range(n_features)]

        # filter all irrelevant features
        epsilon = 1e-5
        self._feat_importances_sorted = [t for t in feat_importances if abs(t[1]) > epsilon]

    def _get_best_features_by_score_unnormed(self):
        super(ExtraTreesAlgorithm, self)._get_best_features_by_score_unnormed()
        return self._feat_importances_sorted

    def get_best_features_by_rank(self):
        super(ExtraTreesAlgorithm, self).get_best_features_by_rank()

        n_features_to_keep = min(len(self._feat_importances_sorted), self._n)

        return [feat_by_score[0] for feat_by_score in self._feat_importances_sorted[:n_features_to_keep]]

    def get_best_features(self):
        super(ExtraTreesAlgorithm, self).get_best_features()

        # we can call this function since it is the same except that we cannot use the fact that is a sorted list
        return self.get_best_features_by_rank()
