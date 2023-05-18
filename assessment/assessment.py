import numpy as np
from os.path import exists
import pandas as pd


def assess(predict_labels, labels):
    """
    assess predict result
    :param predict_labels:
    :param labels:
    :return: precision, recall, accuracy, F1_Score
    """
    TP = np.sum(np.bitwise_and(predict_labels, labels))
    TN = np.sum(np.bitwise_and(predict_labels == 0, labels == 0))
    FP = np.sum(np.bitwise_and(predict_labels == 1, labels == 0))
    FN = np.sum(np.bitwise_and(predict_labels == 0, labels == 1))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    F1_Score = precision * recall / 2 * (precision + recall)
    return dict(precision=precision, recall=recall, accuracy=accuracy, F1_Score=F1_Score,
                TP=TP, TN=TN, FP=FP, FN=FN)


def export_to_excel(accuracy, path, method, preprocess="", **kwargs):
    """
    export assess result to excel
    :param result:
    :param path:
    :return:
    """
    if len(kwargs) == 0:
        kwargs = "None"
    if exists(path):
        xl = pd.read_excel(path)
        data = {"model": method, "preprocess": preprocess,
                "hyper parameter": str(kwargs).rstrip("}").lstrip("{"),
                **accuracy}
        xl = xl.append(data, ignore_index=True)
        xl.to_excel(path, index=False)
    else:
        df = pd.DataFrame(accuracy, index=(1,))
        df.insert(loc=0, column="hyper parameter", value=str(kwargs).rstrip("}").lstrip("{"))
        df.insert(loc=0, column="preprocess", value=preprocess)
        df.insert(loc=0, column="model", value=method)
        df.to_excel(path, index=False)
    return
