from merge.techniques.SubsetMerger import SubsetMerger


class TwoByTwoIntersectionsMerger(SubsetMerger):
    """
    This merger sorts the lists by length (desc). Then it takes the intersection
    to the two firsts and use the result to make an other intersection with the next list
    and so on until there is no more lists.

    This merger is very selective because only the features that are in all the lists are kept.
    """

    def __init__(self, subsets):
        super(TwoByTwoIntersectionsMerger, self).__init__(subsets)

    def _merge(self):
        sort_by_len_features = sorted(self._subsets.values(), key=lambda x: len(x), reverse=True)
        print([len(f) for f in sort_by_len_features])

        def inter(x, y):
            intersection = list(set(x).intersection(set(y)))
            print("Intersection length : %d" % len(intersection))
            return intersection

        lists_of_features = [([a[0] for a in f]) for f in sort_by_len_features]

        # keep the lists that contains at least 500 features. Otherwise the intersection will be too small
        lists_of_features = filter(lambda x: len(x) > 500, lists_of_features)
        print([len(f) for f in lists_of_features])

        return reduce(inter, lists_of_features)
