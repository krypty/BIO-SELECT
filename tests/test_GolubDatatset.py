from datasets.Golub99.GolubDataset import GolubDataset


class TestGolubDataset:
    def test_get_features_names(self):
        ds = GolubDataset()
        best_features_idx = [4, 10, 2]
        best_features_names = ds.get_features_names(best_features_idx)

        assert best_features_names[0] == "AFFX-BioC-3_at"
        assert best_features_names[1] == "AFFX-BioB-M_st"
        assert best_features_names[2] == "AFFX-BioB-3_at"
