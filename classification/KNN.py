import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from final.classification.base import BaseClassification


class KNN(BaseClassification):
    def __init__(self, n_neighbor=5, name="KNN", **kwargs):
        """
        Nearest means classifier.
        """
        self.class_name = name
        super(KNN, self).__init__(**kwargs)
        self.KNN = KNeighborsClassifier(n_neighbors=n_neighbor)

    def fit(self, data, labels):
        train_data = data.drop("Date", axis=1).to_numpy()
        self.KNN.fit(train_data, labels)
        self.models["parameter"] = np.array(())

    def classify(self, data):
        ret = data.drop("Date", axis=1).to_numpy()
        ret = self.KNN.predict(ret)
        return ret

    def save_model(self):
        """
        there are no model parameter, therefore no model save function.
        :return:
        """

    def load_model(self):
        """
        there are no model parameter, therefore no load model save function.
        :return:
        """
