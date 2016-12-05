import os

from utils.FeaturesImporter import FeaturesImporter


class CSVFeaturesImporter(FeaturesImporter):
    def __init__(self, group_name):
        super(CSVFeaturesImporter, self).__init__(group_name)
        self._DELIMITER = ";"
        self._base_filename = "outputs" + os.sep + self._group_name

        self._subsets = {}

    def load(self):
        self._subsets["features"] = self._import_features()
        self._subsets["features_by_rank"] = self._import_features_by_rank()
        self._subsets["features_by_score"] = self._import_features_by_score()

        return self._subsets

    def _import_features(self):
        filename = self._base_filename + "_features.csv"
        return self._extract_list_of_features(filename)

    def _import_features_by_rank(self):
        filename = self._base_filename + "_features_by_rank.csv"
        return self._extract_list_of_features(filename)

    def _import_features_by_score(self):
        filename = self._base_filename + "_features_by_score.csv"

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

    def _extract_list_of_features(self, filename):
        list_of_features = {}
        with open(filename, "rb") as csvfile:
            for line in csvfile:
                line_splited = line.split(self._DELIMITER)

                # ignore the algorithm if it does not provide any list
                if len(line_splited) <= 2:
                    continue

                alg_name, feats = line_splited[0], [int(feat) for feat in line_splited[1:]]
                list_of_features[alg_name] = feats
        return list_of_features


if __name__ == '__main__':
    os.chdir(r'/home/gary/Dropbox/Master/3emeSemestre/TM/code')
    group_name = "golub"
    importer = CSVFeaturesImporter(group_name)
    subsets = importer.load()
    print(subsets["features"].keys())
    print(subsets["features_by_score"]["ReliefF"][:5])
