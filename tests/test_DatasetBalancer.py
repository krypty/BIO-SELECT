from datasets.DatasetBalancer import DatasetBalancer
from datasets.DatasetSplitter import DatasetSplitter
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

    def test_oversampling_split(self):
        ds = DummyDataset()

        print(ds.get_X())
        print(ds.get_y())

        ds = DatasetSplitter(ds, test_size=0.5)
        dsb = DatasetBalancer(ds)
        ds = dsb.balance()

        print(ds.get_X_train())
        print(ds.get_y_train())

        classes_cnt = collections.Counter(ds.get_y_train())

        assert all(class_count == classes_cnt.values()[0] for class_count in
                   classes_cnt.values()), "Some classes are more or less represented than the others"

        assert len(ds.get_X_train()) == len(ds.get_y_train()), "X and y are not the same length !"
