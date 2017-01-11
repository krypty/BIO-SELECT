import numpy as np
import itertools
import matplotlib.pyplot as plt


class SimilarityMatrix:
    def __init__(self, subsets, alg_names, compare_func, title):
        self._alg_names = alg_names
        self._subsets = subsets
        self._compare_func = compare_func
        self._title = title

        self._sm = self._compute_similarity_between_subsets()

    def show(self):
        self._plot_feature_subsets_matrix()

    def _compute_similarity_between_subsets(self):
        n_subsets = len(self._subsets)
        similarity_matrix = np.zeros(shape=(n_subsets, n_subsets))

        for i, j in itertools.product(range(n_subsets), range(n_subsets)):
            if isinstance(self._subsets[0][0], int):
                subset_i = set(self._subsets[i])
                subset_j = set(self._subsets[j])
            else:
                subset_i = {i[0] for i in self._subsets[i]}
                subset_j = {j[0] for j in self._subsets[j]}

            similarity_matrix[i, j] = self._compare_func(subset_i, subset_j)

        return similarity_matrix

    def _plot_feature_subsets_matrix(self):
        self._title += "\n"  # add a little margin below the title

        cmap = plt.cm.Blues

        plt.imshow(self._sm, interpolation='nearest', cmap=cmap)
        plt.title(self._title)
        tick_marks = np.arange(len(self._alg_names))
        plt.xticks(tick_marks, self._alg_names, rotation=45)
        plt.yticks(tick_marks, self._alg_names)

        for i, j in itertools.product(range(self._sm.shape[0]), range(self._sm.shape[1])):
            text = "%.2f" % self._sm[i, j]
            plt.text(j, i, text,
                     horizontalalignment="center",
                     backgroundcolor="white",
                     color="black")

        plt.tight_layout()
