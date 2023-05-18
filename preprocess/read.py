import pandas as pd
import datetime

TEST_SET = "dataset/algerian_fires_test.csv"
TRAIN_SET = "dataset/algerian_fires_train.csv"


# class labels: Date,Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,Classes

def read_data(path):
    """
    read data from path
    :param path: data path
    :return: data
    """
    data = pd.read_csv(path)
    # change data type from string to date
    data['Date'] = pd.to_datetime(data['Date'], format="%d/%m/%Y")
    # increase half of a day to the odd rows
    integer = data['Date'].loc[data.index % 2 == 0]
    half = data['Date'].loc[data.index % 2 == 1] + datetime.timedelta(0.5)
    days = pd.concat([integer, half])
    days = days.sort_index()
    data['Date'] = days
    return data


def read_all(path):
    """
    read data from path
    :param path: data path
    :return: data, and corresponding labels
    """
    data = read_data(path)
    labels = data["Classes"]
    data = data.drop("Classes", axis=1)
    labels = labels.to_numpy()
    return data, labels


