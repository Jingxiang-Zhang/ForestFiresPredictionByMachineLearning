import numpy as np
from sklearn.svm import SVC
from final.classification.base import BaseClassification


class SVM(BaseClassification):
    def __init__(self, name="SVM", kernel='rbf', degree=3, gamma='scale', coef0=0.0, **kwargs):
        """
        SVM parameters:d
            kernel: kernel type, including  'linear', 'poly', 'rbf' and 'sigmoid'
            gamma: used in 'poly', 'rbf' and 'sigmoid', big gamma imply more support vector
                    and over fitting
            degree: used in 'poly'
            coef0: used in 'poly' and 'sigmoid', lower coef0 will cause under fitting,
                    and vise versa.
        """
        self.class_name = name
        super(SVM, self).__init__(**kwargs)
        self.SVM = SVC(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

    def fit(self, data, labels):
        train_data = data.drop("Date", axis=1).to_numpy()
        self.SVM.fit(train_data, labels)
        self.models["parameter"] = np.array(())

    def classify(self, data):
        ret = data.drop("Date", axis=1).to_numpy()
        ret = self.SVM.predict(ret)
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
