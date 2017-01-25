# coding: utf-8
from __future__ import division, print_function

from merge.MariglianoWeightedLists import MariglianoWeightedLists
from merge.techniques.SubsetMerger import SubsetMerger
from utils.Dendrogram import Dendrogram


class WeightedListsMerger(SubsetMerger):
    def __init__(self, subsets, max_features_to_keep):
        """
        :param subsets: subsets need to be ranked
        :param Z:
        :param max_features_to_keep: this number must not exceed the total number of features
        """

        super(WeightedListsMerger, self).__init__(subsets)
        self._n_features_to_keep = max_features_to_keep

    def _merge(self):
        self._f_names, self._f_values = self._extract_lists(self._subsets)

        assert len(self._f_names) == len(self._f_values)

        L = len(self._f_names)

        Z = self._compute_Z(self._f_names, self._f_values)

        mwl = MariglianoWeightedLists(Z, L)
        self._W = mwl.compute_weights()

        return self._compute_merged_list(self._f_values, self._W)

    def show_dendrogram(self):
        self._d.show()

    def get_W_per_list(self):
        return zip(self._f_names, self._W)

    @staticmethod
    def _extract_lists(subsets):
        """
        Only keep the features indices, drop the features occurrences
        """
        f_names, f_values = zip(*subsets.items())

        def extract(f_values):
            for fv in f_values:
                try:
                    yield [f_idx for f_idx, _ in fv]
                except ValueError:
                    pass

        f_values = [i for i in extract(f_values)]

        return f_names, f_values

    def _compute_Z(self, f_names, f_values):
        self._d = Dendrogram(lists=f_values, lists_labels=f_names, metric="dice")
        return self._d.get_Z()

    def _compute_merged_list(self, f_values, W):
        # the features are sorted with the highest ranks appear first
        merged_list = []
        for f_list, w in zip(f_values, W):
            keep = int(self._n_features_to_keep * w)
            merged_list.extend(f_list[:keep])

        return set(merged_list)


if __name__ == '__main__':
    s = {
        "a": [(1, 0), (2, 1), (4, 2)],
        "b": [(2, 0), (1, 1), (3, 2)],
        "c": [(10, 0), (40, 1), (1, 2)],
        "d": [(10, 0), (40, 1), (1, 2)]
    }

    wlm = WeightedListsMerger(s, max_features_to_keep=155)
    merged = wlm.merge()
    print(merged)

    wlm.show_dendrogram()
