from merge.techniques.SubsetMerger import SubsetMerger


class TopNMerger(SubsetMerger):
    """
    The purpose of this merger is to keep the N best features among all the algorithms.
    The best features are the ones that appear at most across all lists
    """

    def __init__(self, subsets, n):
        """
        :param subsets: lists of features
        :param n: number of features to keep for the merged list
        """
        super(TopNMerger, self).__init__(subsets)
        self._n = n

    def _merge(self):
        n_algorithms = len(self._subsets)

        all_feats = []

        # flatten the lists
        for f in self._subsets:
            all_feats.extend(f)

        grouped_list = self._group_by_features(all_feats)
        grouped_list = [self._mean_score_for_feature(f, n_algorithms) for f in grouped_list]

        grouped_list = sorted(grouped_list, key=lambda x: x[1], reverse=True)

        return [x[0] for x in grouped_list[:self._n]]

    @staticmethod
    def _group_by_features(features):
        from itertools import groupby

        def keyfunc(x): return x[0]

        list_of_lists_sorted = sorted(features, key=keyfunc)
        grouped_list = [list(j) for i, j in groupby(list_of_lists_sorted, key=keyfunc)]
        return grouped_list

    @staticmethod
    def _mean_score_for_feature(a, n_algorithms):
        feat_name, feat_scores = zip(*a)
        feat_name = feat_name[0]  # since name is the same for all tuples

        m = sum(feat_scores) / float(n_algorithms)
        return feat_name, m
