import json
import os
import numpy as np


class BaseClassification:
    """
    base classification class, all of the classification class will inherit this class
    """

    def __init__(self, **kwargs):
        """
        init classification class by hyper parameter
        """
        self.models = dict()
        self.models["hyper parameter"] = kwargs
        self.load_model()

    def fit(self, data, labels):
        """
        fill the training data, and start training process
        :param data:
        :param labels:
        :return: None
        """

    def __getattribute__(self, *args, **kwargs):
        """
        save model automatically when do classification
        :param args:
        :param kwargs:
        :return:
        """
        if args[0] == 'classify':
            if not self.models.get("parameter", None).all():
                raise Exception("models is not exist, please fit the data first")
            self.save_model()
        return object.__getattribute__(self, *args, **kwargs)

    def classify(self, data):
        """
        classify data by the model
        :param data:
        :return: prediction
        """

    def get_models(self):
        """
        get all of the model parameter of the model
        :return: dictionary type
        """
        return self.models

    def save_model(self):
        """
        typically automatically save the model when finish fitting data
        :return:
        """

        self.models["parameter"] = self.models["parameter"].tolist()
        dict_json = json.dumps(self.models)
        model_name = os.path.join("models", self.class_name + '_model.json')
        with open(model_name, 'w+') as file:
            file.write(dict_json)

    def load_model(self):
        """
        invoke by init function, automatically load model if model exist
        :return:
        """

        model_name = os.path.join("models", self.class_name + '_model.json')
        if os.path.exists(model_name):
            with open(model_name, 'r+') as file:
                content = file.read()
                content = json.loads(content)
                content["parameter"] = np.array(content["parameter"])
                self.models = content
