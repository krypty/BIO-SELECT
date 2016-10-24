from abc import ABCMeta, abstractmethod

from merge.SubsetMerger import SubsetMerger


class SimpleSubsetMerger(SubsetMerger):
    __metaclass__ = ABCMeta

    def __init__(self, subsets):
        super(SimpleSubsetMerger, self).__init__(subsets)

    @abstractmethod
    def merge(self):
        pass
