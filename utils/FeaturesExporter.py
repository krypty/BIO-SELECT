from abc import ABCMeta, abstractmethod
from uuid import uuid4


class FeaturesExporter:
    __metaclass__ = ABCMeta

    def __init__(self, subsets, group_name=None):
        self._group_name = group_name
        if self._group_name is None:
            self._group_name = str(uuid4()).split("-")[0]

        self._subsets = subsets

    @abstractmethod
    def export(self):
        pass
