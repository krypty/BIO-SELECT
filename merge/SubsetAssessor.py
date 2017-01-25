import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


class SubsetAssessor:
    """
    The purpose of this class is to assess a list/subset of features.
    It gives a "averaged" score using several classifiers
    """

    def __init__(self, subset, dataset, k):
        """

        :param subset: the selected features to assess
        :param dataset: the dataset to use
        :param k: the number of times the internal classifiers needs to be run
        """
        self._subset = subset
        self._ds = dataset
        self._k = k

        scores = self._assess()

        # take the median and the std of all the classifiers used to assess the subset
        self._score = np.median(scores)
        self._std = np.std(scores)

    @property
    def score(self):
        return self._score

    @property
    def std(self):
        return self._std

    def _assess(self):
        # hide warnings when the classifier does not converge
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        scores = []

        classifiers = [
            KNeighborsClassifier(algorithm="ball_tree", n_neighbors=4, n_jobs=-1, metric="manhattan"),
            MLPClassifier(hidden_layer_sizes=(20, 10), activation="relu"),
            ExtraTreesClassifier(n_jobs=-1, n_estimators=10)
        ]

        # run the classifiers k times each and take the median of each
        for clf in classifiers:
            scores_clf = self._run_classifier(clf, self._k)
            scores.append(np.median(scores_clf))
        return scores

    def _run_classifier(self, clf, k):
        return cross_val_score(clf, self._ds.get_X_test()[:, self._subset], self._ds.get_y_test(), cv=k, n_jobs=-1,
                               scoring="f1_weighted")
