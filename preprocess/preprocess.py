import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def standardization(data, test_data):
    """
    normalization data
    :param data:
    :param test_data:
    :return: train data and test data
    """
    x = data.drop("Date", axis=1).to_numpy()
    x_norm = np.mean(x, axis=0)
    x_var = np.var(x, axis=0)
    x = (x - x_norm) / np.power(x_var, 0.5)
    ret = pd.DataFrame(x)
    ret.insert(loc=0, column="Date", value=data["Date"])
    ret.columns = [column + "(st)" if column != "Date" else column for column in data.columns]

    y = test_data.drop("Date", axis=1).to_numpy()
    y = (y - x_norm) / np.power(x_var, 0.5)
    y_d = pd.DataFrame(y)
    y_d.insert(loc=0, column="Date", value=test_data["Date"])
    y_d.columns = [column + "(st)" if column != "Date" else column for column in test_data.columns]

    return ret, y_d


def PCA_decomposition(data, test_data, components=0.9):
    """
    Principle component analysis
    :param data:
    :param test_data:
    :param components:
    :return: train data and test data
    """
    x = data.drop("Date", axis=1).to_numpy()
    pca = PCA(n_components=components, copy=False)
    pca.fit(x)
    x = pca.transform(x)
    ret = pd.DataFrame(x)
    ret.columns = ["com({})".format(i) for i in range(len(ret.columns))]
    ret.insert(loc=0, column="Date", value=data["Date"])

    y = test_data.drop("Date", axis=1).to_numpy()
    y = pca.transform(y)
    y = pd.DataFrame(y)
    y.columns = ["com({})".format(i) for i in range(len(y.columns))]
    y.insert(loc=0, column="Date", value=test_data["Date"])
    return ret, y


def preprocess_combination(strategies, data, labels, test_data, test_labels):
    """
    this function provide a fast preprocessing solution. give the strategies, and
    it will yield result by the description of each strategy
    :param strategies: a iterable object, stand for each strategy. each strategy is also
                a iterable object, include items, it can be: PCA, STD, DelNight
                Eg, strategies = [
                        ("STD","PCA","DelNight"),
                        ("STD",),
                        ("PCA","DelNight")
                    ]
    :param data:
    :param labels:
    :param test_data:
    :param test_labels:
    :return:
    """
    for strategy in strategies:
        temp_data = data
        temp_labels = labels
        temp_test_data = test_data
        temp_test_labels = test_labels
        for item in strategy:
            if item == "PCA":
                temp_data, temp_test_data = PCA_decomposition(temp_data, temp_test_data)
            elif item == "STD":
                temp_data, temp_test_data = standardization(temp_data, temp_test_data)
            elif item == "DelNight":
                temp_data, temp_labels, temp_test_data, temp_test_labels = \
                    delete_night(temp_data, temp_labels, temp_test_data, temp_test_labels)
        yield temp_data, temp_labels, temp_test_data, temp_test_labels


def all_combination(data, labels, test_data, test_labels):
    """
    this is a encapsulation of preprocess_combination, yield 2*2*2 combination
    :param data:
    :param labels:
    :param test_data:
    :param test_labels:
    :return:
    """
    strategies = [
        [],
        ["PCA"],
        ["STD"],
        ["DelNight"],
        ["PCA", "STD"],
        ["PCA", "DelNight"],
        ["STD", "DelNight"],
        ["PCA", "STD", "DelNight"],
    ]
    i = -1
    for temp_data, temp_labels, temp_test_data, temp_test_labels in \
            preprocess_combination(strategies, data, labels, test_data, test_labels):
        i += 1
        yield temp_data, temp_labels, temp_test_data, temp_test_labels, strategies[i]


def plus_one_feature_iteration(data, labels, test_data, test_labels, days=(2, 3, 4)):
    """
    a automatic yield extra feature function, it will iterate over all possible column,
    and use the days and average/minimal/maximal method to create the new feature,
    and drop the original one at the same time.
    There will be total len(columns) * len(days) * len(["AVE","MIN","MAX"]) data
    :param data:
    :param labels:
    :param test_data:
    :param test_labels:
    :param days:
    :return:
    """
    for column in data.columns:
        if column == "Date":
            continue
        for operation in ["AVE", "MIN", "MAX"]:
            for day in days:
                temp_data = data
                temp_labels = labels
                temp_test_data = test_data
                temp_test_labels = test_labels
                temp_data, temp_labels, temp_test_data, temp_test_labels = \
                    plus_last_n_date_feature(temp_data, temp_labels, temp_test_data,
                                             temp_test_labels, day, [(column, operation)])
                temp_data = temp_data.drop(column, axis=1)
                temp_test_data = temp_test_data.drop(column, axis=1)

                op = temp_data.columns[-1]
                yield temp_data, temp_labels, temp_test_data, temp_test_labels, op


def plus_last_n_date_feature(data, labels, test_data, test_labels, days, operation):
    """
    create a new feature, add it into the data set. the new feature must base on one exist features,
    Use the average/minimal/maximal of the last N days separate the daytime and night.
    For a example, create a humidity feature based on last 3 days average humidity.
    If
    :param data:
    :param labels:
    :param test_data:
    :param test_labels:
    :param days: use last n days to create new feature
    :param operation: a list of adding operations, each item is a tuple, including
           two values: original_tag, method.
            original_tag:  the original feature tag, eg : "humidity"
            method: include MIN, MAX, AVE(stand for average)
            Eg: Adding 3 new features, the maximal value of last N days' humidity, the average
                value of last N days' temperature, and the min value of last N days' rain, then
                the operation would be:
                    [('humidity','MAX'),('temperature','AVE'),('rain','MIN')]
    :return:
    """
    # calculate test set
    before_data = data.iloc[-days * 2 + 2:]  # take out the last N-1 days from the training data set
    test_data = pd.concat([before_data, test_data])  # concatenate those data
    # increase new feature into the test data set
    for original_tag, method in operation:
        plus_last_n_date_feature_single(test_data, original_tag, days * 2, method)
    test_data = test_data.iloc[days * 2 - 2:]  # delete the data

    # calculate train set
    data = data.iloc[:-days * 2 + 2]  # drop the last N-1 days from training data set

    for original_tag, method in operation:
        plus_last_n_date_feature_single(data, original_tag, days * 2, method)
    data = data.iloc[days * 2 - 2:]
    labels = labels[days * 2 - 2:-days * 2 + 2]

    return data, labels, test_data, test_labels


def plus_last_n_date_feature_single(data, original_tag, days, method):
    """
    called by plus_last_n_date_feature
    :param data:
    :param original_tag:
    :param days:
    :param method:
    :return:
    """
    old_feature = data[original_tag]
    new_feature = list()
    for i in range(0, len(old_feature)):
        if i < days:
            if i % 2 == 0:
                data_list = old_feature[0:i + 1:2]
            else:
                data_list = old_feature[1:i + 1:2]
        else:
            data_list = old_feature[i - days:i + 1:2]
        if method == "AVE":
            ave = np.average(data_list)
            new_feature.append(ave)
        elif method == "MIN":
            minn = np.min(data_list)
            new_feature.append(minn)
        elif method == "MAX":
            maxx = np.max(data_list)
            new_feature.append(maxx)

    data.insert(value=new_feature, loc=len(data.columns),
                column=original_tag + "_" + method + "_" + str(int(days / 2)))


def delete_night(data, labels, test_data, test_labels):
    """
    delete all of the night data, you must do this process in the end of all data
    preprocessing process.
    :param data:
    :param labels:
    :param test_data:
    :param test_labels:
    :return:
    """
    data = data.iloc[1::2]
    labels = labels[1::2]
    test_data = test_data.iloc[1::2]
    test_labels = test_labels[1::2]
    return data, labels, test_data, test_labels


def cross_validation(data, labels):
    """
    cross validation
    :param data:
    :param labels:
    :return:
    """
    start_date = ['2012-06-01', '2012-07-01', '2012-08-01']
    end_date = ['2012-06-30', '2012-07-31', '2012-08-31']
    # Because there are lots of fire in August, the data in August will be intolerable
    # biased. And the F1_Score will be surpassing all the other data.Therefore,
    # I will drop August data as validation set.
    for i in range(2):
        validation = (data['Date'] >= start_date[i]) & (data['Date'] <= end_date[i])
        train = ~validation

        validation_labels = labels[validation]
        train_labels = labels[train]

        validation = data.loc[validation]
        train = data.loc[train]

        yield train, train_labels, validation, validation_labels
