from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier

from algorithms.Algorithm import Algorithm


class ExtraTreesAlgorithm(Algorithm):
    def __init__(self, dataset):
        super(ExtraTreesAlgorithm, self).__init__(dataset)

        X_train = self._dataset.get_X_train()
        y_train = self._dataset.get_y_train()

        classifier = OneVsRestClassifier(
            ExtraTreesClassifier(n_jobs=-1, n_estimators=100), n_jobs=-1)

        classifier.fit(X_train, y_train)

        feat_importances = classifier.estimators_[0].feature_importances_
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
