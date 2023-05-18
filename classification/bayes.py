import numpy as np
from sklearn.naive_bayes import GaussianNB
from final.classification.base import BaseClassification


class Bayes(BaseClassification):
    def __init__(self, name="Bayes", **kwargs):
        """
        Nearest means classifier.
        """
        self.class_name = name
        super(Bayes, self).__init__(**kwargs)
        self.GaussianNB = GaussianNB()

    def fit(self, data, labels):
        train_data = data.drop("Date", axis=1).to_numpy()
        self.GaussianNB.fit(train_data, labels)
        self.models["parameter"] = np.array(())

    def classify(self, data):
        ret = data.drop("Date", axis=1).to_numpy()
        ret = self.GaussianNB.predict(ret)
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
