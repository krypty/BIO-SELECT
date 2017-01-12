from abc import abstractmethod, ABCMeta


class SubsetMerger:
    __metaclass__ = ABCMeta

    def __init__(self, subsets):
        """

        :type subsets: list
        """
        self._subsets = subsets

    def merge(self):
        # type: () -> set
        """
        Merge the lists into one list
        :return: the merged list of features
        """
        features = self._merge()
        self._assert_list_contains_only_unique_features(features)
        return features

    @abstractmethod
    def _merge(self):
        pass

    # ensure that when we merge the lists of features, the list remains composed of unique features
    @staticmethod
    def _assert_list_contains_only_unique_features(features):
        assert len(features) == len(set(features))
