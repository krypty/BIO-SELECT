from merge.techniques.SubsetMerger import SubsetMerger


class UnionSubsetMerger(SubsetMerger):
    def __init__(self, subsets):
        super(UnionSubsetMerger, self).__init__(subsets)

    def _merge(self):
        super(UnionSubsetMerger, self)._merge()

        ensemble_set = set()

        if isinstance(self._subsets[0][0], tuple):
            # subset contains features and scores
            for subset in self._subsets:
                features = {feat_by_score[0] for feat_by_score in subset}
                ensemble_set = ensemble_set.union(features)
        else:
            # subset only contains features
            for subset in self._subsets:
                ensemble_set = ensemble_set.union(set(subset))

        return ensemble_set
