import os

from utils.FeaturesImporter import FeaturesImporter


class CSVFeaturesImporter(FeaturesImporter):
    def __init__(self, group_name):
        super(CSVFeaturesImporter, self).__init__(group_name)
        self._DELIMITER = ";"
        self._base_filename = "outputs" + os.sep + self._group_name

        self._subsets = {}

    def load(self):
        self._subsets["features"] = self._import_features(features_type="features")
        self._subsets["features_by_rank"] = self._import_features(features_type="features_by_rank")
        self._subsets["features_by_score"] = self._import_features(features_type="features_by_score")

        return self._subsets

    def _import_features(self, features_type):
        filename = self._base_filename + "_" + features_type + ".csv"

        list_of_features = {}
        with open(filename, "rb") as csvfile:
            for line in csvfile:
                line_splited = line.split(self._DELIMITER)

                # ignore the algorithm if it does not provide any list
                if len(line_splited) <= 2:
                    continue

                alg_name, feats = line_splited[0], line_splited[1:]

                feats = [(int(f[0]), float(f[1])) for f in [feat.split(",") for feat in feats]]
                list_of_features[alg_name] = feats
        return list_of_features
