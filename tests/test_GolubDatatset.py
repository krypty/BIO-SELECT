from datasets.Golub99.GolubDataset import GolubDataset


class TestGolubDataset:
    def test_get_features_names(self):
        ds = GolubDataset()
        best_features_idx = [4, 10, 2]
        best_features_names = ds.get_features_names(best_features_idx)

        assert best_features_names[0] == "AFFX-BioC-3_at"
        assert best_features_names[1] == "AFFX-BioB-M_st"
        assert best_features_names[2] == "AFFX-BioB-3_at"

    def test_get_features_indices(self):
        ds = GolubDataset()

        best_features_names = ["AFFX-BioC-3_at", "AFFX-BioB-M_st", "AFFX-BioB-3_at"]
        best_features_idx = ds.get_features_indices(best_features_names)

        assert best_features_idx[0] == 4
        assert best_features_idx[1] == 10
        assert best_features_idx[2] == 2
