from abc import abstractmethod, ABCMeta


class SubsetMerger:
    __metaclass__ = ABCMeta

    def __init__(self, subsets):
        """

        :type subsets: list
        """
        self._subsets = subsets

    @abstractmethod
    def merge(self):
        # type: () -> set
        """
        Merge the lists into one list
        :return: the merged list of features
        """
        pass
