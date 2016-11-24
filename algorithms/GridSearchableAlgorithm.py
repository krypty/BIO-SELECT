from abc import ABCMeta, abstractmethod

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from algorithms.Algorithm import Algorithm


class GridSearchableAlgorithm(Algorithm):
    __metaclass__ = ABCMeta

    def __init__(self, dataset, n, name, gridsearch_params):
        super(GridSearchableAlgorithm, self).__init__(dataset, n, name)
        self._gridsearch_params = gridsearch_params

        self._init_classifier()

        if self._gridsearch_params is not None:
            self._grid_search()
        else:
            self._fit()

        score = self._clf.score(self._dataset.get_X_test(), self._dataset.get_y_test())

        self._retrieve_best_features()

    @abstractmethod
    def _init_classifier(self):
        pass

    def _grid_search(self):
        X_train = self._dataset.get_X_train()
        y_train = self._dataset.get_y_train()

        self._clf_gs = GridSearchCV(self._clf, param_grid=self._gridsearch_params, n_jobs=2, cv=3)

        self._clf_gs = self._clf_gs.fit(X_train, y_train)

        self._clf = self._clf_gs.best_estimator_

        self._best_params = self._clf_gs.best_params_

    def _fit(self):
        X_train = self._dataset.get_X_train()
        y_train = self._dataset.get_y_train()

        self._clf.fit(X_train, y_train)

    @abstractmethod
    def _retrieve_best_features(self):
        pass

    def get_score(self):
        try:
            X_test = self._dataset.get_X_test()
            y_test = self._dataset.get_y_test()
            return self._clf.score(X_test, y_test)
        except AttributeError:
            super(GridSearchableAlgorithm, self).get_score()

    @property
    def best_params(self):
        try:
            return self._best_params
        except AttributeError:
            raise AttributeError(
                "Impossible to retrieve to best grid search params because grid search has not been used")

    def get_confusion_matrix(self):
        y_pred = self._clf.predict(self._dataset.get_X_test())
        return confusion_matrix(self._dataset.get_y_test(), y_pred)
