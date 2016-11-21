from datasets.DatasetBalancer import DatasetBalancer
from datasets.DummyDataset import DummyDataset

import collections


class TestDatasetBalancer:
    def test_oversampling(self):
        ds = DummyDataset()

        print(ds.get_X())
        print(ds.get_y())

        dsb = DatasetBalancer(ds)
        ds = dsb.balance()

        print(ds.get_X())
        print(ds.get_y())

        classes_cnt = collections.Counter(ds.get_y())

        assert all(class_count == classes_cnt.values()[0] for class_count in
                   classes_cnt.values()), "Some classes are more or less represented than the others"

        assert len(ds.get_X()) == len(ds.get_y()), "X and y are not the same length !"
