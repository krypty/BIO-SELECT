import os

from datasets.MILE.MileDataset import MileDataset


class TestMileDataset:
    def test_get_features_names(self):
        ds = MileDataset(samples_limit=10)

        best_features_idx = [4, 10, 2]
        best_features_names = ds.get_features_names(best_features_idx)

        assert best_features_names[0] == "AFFX-HUMRGE/M10098_5_at"
        assert best_features_names[1] == "AFFX-HSAC07/X00351_5_at"
        assert best_features_names[2] == "AFFX-HUMISGF3A/M97935_MB_at"

    def test_get_features_indices(self):
        ds = MileDataset(samples_limit=10)

        best_features_names = ["AFFX-HUMRGE/M10098_5_at", "AFFX-HSAC07/X00351_5_at", "AFFX-HUMISGF3A/M97935_MB_at"]
        best_features_idx = ds.get_features_indices(best_features_names)

        assert best_features_idx[0] == 4
        assert best_features_idx[1] == 10
        assert best_features_idx[2] == 2
