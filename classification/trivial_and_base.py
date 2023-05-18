import numpy as np
from scipy.spatial.distance import cdist
from final.classification.base import BaseClassification


class Baseline(BaseClassification):
    def __init__(self, name="baseline", **kwargs):
        """
        Nearest means classifier.
        """
        self.class_name = name
        super(Baseline, self).__init__(**kwargs)

    def fit(self, data, labels):
        train_data = data.drop("Date", axis=1).to_numpy()
        x1 = train_data[labels == 0]
        x2 = train_data[labels == 1]
        x1_mean = np.mean(x1, axis=0)
        x2_mean = np.mean(x2, axis=0)
        sample_mean = np.array([x1_mean, x2_mean])
        self.models["parameter"] = sample_mean

    def classify(self, data):
        data = data.drop("Date", axis=1).to_numpy()
        dist = cdist(data, self.models["parameter"])
        ret = np.argmin(dist, axis=1)
        return ret


class Trivial(BaseClassification):
    def __init__(self, name="trivial", **kwargs):
        """
        Nearest means classifier.
        """
        self.class_name = name
        super(Trivial, self).__init__(**kwargs)

    def fit(self, data, labels):
        N1 = np.sum(labels == 0)
        N2 = np.sum(labels == 1)
        mid = N1 / (N1 + N2)
        self.models["parameter"] = mid

    def classify(self, data):
        mid = self.models["parameter"]
        rand = np.random.random(len(data.index))
        rand[rand < mid] = 0
        rand[rand >= mid] = 1
        rand = np.array(rand, dtype=np.bool)
        return rand
