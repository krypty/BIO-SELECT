from algorithms.Algorithm import NotSupportedException
from algorithms.ExtraTreesAlgorithm import ExtraTreesAlgorithm
from algorithms.FValueAlgorithm import FValueAlgorithm
from algorithms.SVMAlgorithm import SVMAlgorithm
from datasets.DatasetEncoder import DatasetEncoder
from datasets.DatasetSplitter import DatasetSplitter
from datasets.Golub99.GolubDataset import GolubDataset


class TestAlgorithm:
    def init(self):
        self._ds = GolubDataset()

        # encode Dataset string classes into numbers
        ds_encoder = DatasetEncoder(self._ds)
        self._ds = ds_encoder.encode()
        self._ds = DatasetSplitter(self._ds, test_size=0.4)

    def test_algorithm(self):
        self.init()

        n = 10
        eta = ExtraTreesAlgorithm(self._ds, n)
        features_by_score = eta.get_best_features_by_score()
        print(features_by_score)

        features_by_rank = eta.get_best_features_by_rank()
        print(features_by_rank)

        assert len(features_by_score) == n, "Features list is not the same size as requested"
        assert len(features_by_rank) == n, "Features list is not the same size as requested"
        assert features_by_score[0][0] == features_by_rank[
            0], "Best feature is not the same between the score/ranking methods"

    def test_get_score(self):
        self.init()

        n = 10
        svm = SVMAlgorithm(self._ds, n)

        assert svm.get_score() > 0.0

    def test_get_score_unsupported(self):
        self.init()

        n = 10
        svm = FValueAlgorithm(self._ds, n)

        try:
            svm.get_score()
            assert False
        except NotSupportedException:
            assert True
