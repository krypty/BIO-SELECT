from sklearn.ensemble import ExtraTreesClassifier

from algorithms.GridSearchableAlgorithm import GridSearchableAlgorithm


class ExtraTreesAlgorithm(GridSearchableAlgorithm):
    def __init__(self, dataset, gridsearch_params=None):
        super(ExtraTreesAlgorithm, self).__init__(dataset, gridsearch_params=gridsearch_params, name="ExtraTrees")

    def _init_classifier(self):
        self._clf = ExtraTreesClassifier(n_jobs=-1, n_estimators=100)

    def _retrieve_best_features(self):
        feat_importances = self._clf.estimators_[0].feature_importances_
        self._feat_importances_sorted = sorted(enumerate(feat_importances), key=lambda x: x[1], reverse=True)

    def _get_best_features_by_score_unnormed(self):
        super(ExtraTreesAlgorithm, self)._get_best_features_by_score_unnormed()
        return self._feat_importances_sorted

    def get_best_features_by_rank(self, n):
        super(ExtraTreesAlgorithm, self).get_best_features_by_rank(n)
        assert n < len(self._feat_importances_sorted)

        return [feat_by_score[0] for feat_by_score in self._feat_importances_sorted[:n]]

    def get_best_features(self, n):
        super(ExtraTreesAlgorithm, self).get_best_features(n)

        # we can call this function since it is the same except that we cannot use the fact that is a sorted list
        return self.get_best_features_by_rank(n)
