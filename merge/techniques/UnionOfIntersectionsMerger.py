# coding: utf-8
from __future__ import division, print_function

from itertools import islice

from merge.techniques.SubsetMerger import SubsetMerger


class UnionOfIntersectionsMerger(SubsetMerger):
    """
    Take the union of all the intersections of the lists taken two by two (sliding window).

    Warning: this merging technique does not care about the similarity between the lists.
    Although "a" and "c" have common features, there is no guarantee that this merging technique
    will intersect "a" and "c" at all. For example, this can happens : "a" inter "c", "c" inter "b", "b" inter "d"
    resulting a merged list with only the feature "1" (the only that is in all lists)
    """

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

        # sort the lists by decreasing length allow to always have the lists merged in the same order and it allows
        # to do the intersections between lists with lists about the same size
        sort_by_len_features = sorted(self._subsets.values(), key=lambda x: len(x), reverse=True)
        lists_of_features = [([a[0] for a in f]) for f in sort_by_len_features]

        for s1, s2 in UnionOfIntersectionsMerger._window(lists_of_features, n=2):
            inter = set(s1).intersection(set(s2))
            merged_subset.extend(inter)

        return set(merged_subset)


if __name__ == '__main__':
    s = {
        "a": [(1, 0), (2, 0), (4, 0)],
        "b": [(1, 0), (2, 0), (3, 0)],
        "c": [(10, 0), (40, 0), (1, 0)],
        "d": [(20, 0), (40, 0), (1, 0)]
    }

    merged = UnionOfIntersectionsMerger(s).merge()
    print(merged)
