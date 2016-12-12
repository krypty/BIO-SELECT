import os

from utils.FeaturesExporter import FeaturesExporter


class CSVFeaturesExporter(FeaturesExporter):
    def __init__(self, subsets, group_name=None):
        super(CSVFeaturesExporter, self).__init__(subsets, group_name)
        self._DELIMITER = ";"
        self._base_filename = "outputs" + os.sep + self._group_name

    def export(self):
        self._export_features(features_type="features")
        self._export_features(features_type="features_by_rank")
        self._export_features(features_type="features_by_score")

    def _export_features(self, features_type):
        """
        Export list of features by score
        Line format: algorithm_name;feat0,score0;feat1,score1;featN,scoreN
        """
        filename = self._base_filename + "_" + features_type + ".csv"

        subsets_feats = [(s_name, s_feats[features_type]) for (s_name, s_feats) in self._subsets.items()]

        with open(filename, "wb") as csv_file:
            for feats in subsets_feats:
                line = feats[0] + self._DELIMITER + self._DELIMITER.join(["%d,%.5f" % (f[0], f[1]) for f in feats[1]])
                csv_file.write(line + os.linesep)
