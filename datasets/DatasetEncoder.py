from sklearn.preprocessing import LabelEncoder

from datasets.Dataset import Dataset


class DatasetEncoder:
    """
    Encode a dataset using LabelEncoder to transform string classes into numbers
    """

    def __init__(self, dataset):
        self._dataset = dataset  # type: Dataset
        self._le = LabelEncoder()

    def encode(self):
        """
        Encode y (classes) into numbers
        :return: a shallow copy of the encoded dataset
        """
        self._le.fit(self._dataset.get_y())

        self._dataset._y = self._le.transform(self._dataset.get_y())
        return self._dataset

    def get_label_encoder(self):
        return self._le
