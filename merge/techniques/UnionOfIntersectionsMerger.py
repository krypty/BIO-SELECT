# coding: utf-8
from __future__ import division, print_function

from itertools import islice

from merge.techniques.SubsetMerger import SubsetMerger


class UnionOfIntersectionsMerger(SubsetMerger):
    def __init__(self, subsets):
        super(UnionOfIntersectionsMerger, self).__init__(subsets)

    @staticmethod
    def _window(seq, n=2):
        """
        Returns a sliding window (of width n) over data from the iterable
        s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
        source: http://stackoverflow.com/a/6822773
        """
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    def _merge(self):
        super(UnionOfIntersectionsMerger, self)._merge()

        merged_subset = []

        sort_by_len_features = sorted(self._subsets.values(), key=lambda x: len(x), reverse=True)
        lists_of_features = [([a[0] for a in f]) for f in sort_by_len_features]

        for s1, s2 in UnionOfIntersectionsMerger._window(lists_of_features, n=2):
            inter = set(s1).intersection(set(s2))
            merged_subset.extend(inter)

        return set(merged_subset)


if __name__ == '__main__':
    s = [{1, 2, 4}, {1, 2, 3}, {10, 40, 60}]

    merged = UnionOfIntersectionsMerger(s).merge()
    print(merged)
