from datasets.MILE.MileDatasetSampleParser import MileDatasetSampleParser


class TestMileDatasetSampleParser:
    def test_get_features_names(self):
        sample_parser = MileDatasetSampleParser(r"./data/MILE/processed/GSM329407_sample_table.txt")
        X = sample_parser.get_X()
        print(len(X))

        sample_parser.parse_features_names()
        features_names = sample_parser.get_features_names()
        print(features_names)

        assert features_names[4] == "AFFX-HUMRGE/M10098_5_at"
