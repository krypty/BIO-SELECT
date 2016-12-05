from abc import ABCMeta, abstractmethod


class FeaturesImporter:
    __metaclass__ = ABCMeta

    def __init__(self, group_name):
        self._group_name = group_name

    @abstractmethod
    def load(self):
        pass
