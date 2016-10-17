from datasets.Golub99.GolubDatasetSampleParser import GolubDatasetSampleParser


class TestGolubDatasetSampleParser:
    def test_get_features_names(self):
        sample_parser = GolubDatasetSampleParser(r"./data/golub99/processed/test/sample_0_ALL.csv")
        X = sample_parser.get_X()

        sample_parser.parse_features_names()
        features_names = sample_parser.get_features_names()

        assert features_names[4] == "AFFX-BioC-3_at"
