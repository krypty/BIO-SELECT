from skfeature.function.wrapper import svm_forward

from algorithms.Algorithm import Algorithm, NotSupportedException


class SVMForwardAlgorithm(Algorithm):
    """
    This algorithm is NOT determinist
    """
    def __init__(self, dataset, n=None):
        super(SVMForwardAlgorithm, self).__init__(dataset, n, "SVM Forward")

        # execute this algorithm on all the dataset because internally it uses a KFold
        X = dataset.get_X()
        y = dataset.get_y()
        self._F = svm_forward.svm_forward(X, y, n)

    def _get_best_features_by_score_unnormed(self):
        raise NotSupportedException()

    def get_best_features_by_rank(self):
        raise NotSupportedException()

    def get_best_features(self):
        return self._F
