import matplotlib.pyplot as plt
from prettytable import prettytable
import numpy as np
import pandas as pd


def show_feature_labels_one_by_one(data, labels):
    """
    show all of the features one by one as x-axis, and fire or not as y-axis
    :param data:
    :param labels:
    :return:
    """
    labels = labels.to_numpy()
    fire_labels = labels[labels == 1]
    no_fire_labels = labels[labels == 0]

    height = __import__("math").ceil(len(data.columns) / 4)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    for i, column_name in enumerate(data.columns, 1):
        column = data[column_name].to_numpy()
        fire_data = column[labels == 1]
        no_fire_data = column[labels == 0]

        plt.subplot(4, height, i)
        plt.xlabel(column_name)
        plt.scatter(fire_data, fire_labels, c="r")
        plt.scatter(no_fire_data, no_fire_labels, c="g")
        plt.yticks([])  # close y ticks

    plt.show()


def show_correlation(data):
    """
    show correlation between any two features
    :param data:
    :return:
    """
    size = len(data.columns)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    for i, column_name1 in enumerate(data.columns, 0):
        for j, column_name2 in enumerate(data.columns, 1):
            column1 = data[column_name1].to_numpy()
            column2 = data[column_name2].to_numpy()
            plt.subplot(size, size, i * size + j)
            plt.scatter(column1, column2, c="r", s=1)
            plt.xticks([])  # close x ticks
            plt.yticks([])  # close y ticks

    for i, column_name in enumerate(data.columns, 1):
        plt.subplot(size, size, i)
        plt.title(column_name)
    for i, column_name in enumerate(data.columns, 0):
        plt.subplot(size, size, i * size + 1)
        plt.ylabel(column_name)
    plt.show()


def export_correlation_coefficient_to_excel(data, path):
    """
    export correlation coefficient matrix to excel
    :param data:
    :param path:
    :return:
    """
    data = data.drop("Date", axis=1)
    num = data.to_numpy()
    num = np.transpose(num)
    correlation_coef_metrix = np.corrcoef(num)
    df = pd.DataFrame(correlation_coef_metrix)
    df.columns = data.columns
    df.index = data.columns
    df.to_excel(path)


def export_data_to_excel(data, path):
    """
    export data to excel
    :param data:
    :param path:
    :return:
    """
    data = data.iloc[:5]
    data.to_excel(path)
