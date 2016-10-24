from algorithms.ExtraTreesAlgorithm import ExtraTreesAlgorithm
from datasets.DatasetEncoder import DatasetEncoder
from datasets.DatasetSplitter import DatasetSplitter
from datasets.Golub99.GolubDataset import GolubDataset


class TestAlgorithm:
    def test_algorithm(self):
        ds = GolubDataset()

        # encode Dataset string classes into numbers
        ds_encoder = DatasetEncoder(ds)
        ds = ds_encoder.encode()
        ds = DatasetSplitter(ds, test_size=0.4)

        n = 10
        eta = ExtraTreesAlgorithm(dataset=ds)
        features_by_score = eta.get_best_features_by_score(n)
        print(features_by_score)

        features_by_rank = eta.get_best_features_by_rank(n)
        print(features_by_rank)

        assert len(features_by_score) == n, "Features list is not the same size as requested"
        assert len(features_by_rank) == n, "Features list is not the same size as requested"
        assert features_by_score[0][0] == features_by_rank[
            0], "Best feature is not the same between the score/ranking methods"
