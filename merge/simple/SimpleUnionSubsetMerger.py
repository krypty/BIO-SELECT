from merge.simple.SimpleSubsetMerger import SimpleSubsetMerger


class SimpleUnionSubsetMerger(SimpleSubsetMerger):
    def __init__(self, subsets):
        super(SimpleUnionSubsetMerger, self).__init__(subsets)

    def merge(self):
        super(SimpleUnionSubsetMerger, self).merge()

        ensemble_set = set()
        for subset in self._subsets:
            ensemble_set = ensemble_set.union(set(subset))

        return ensemble_set
