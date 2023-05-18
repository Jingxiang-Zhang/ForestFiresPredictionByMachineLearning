import final.preprocess.read as read
from final.preprocess.read import TRAIN_SET, TEST_SET
from final.preprocess import preprocess
from final.classification import KNN
from final.assessment import assessment

export_path = "KNN2.xlsx"
K = 13  # numbers of neighbors, found by iterate over integer number


def main():
    data, labels = read.read_all(TRAIN_SET)
    data_test, labels_test = read.read_all(TEST_SET)

    # using standardization will improve the result
    data, data_test = preprocess.standardization(data, data_test)
    # the best hyper parameter K
    # find_best_K(data, labels)

    KNN_classfy(data, labels, data_test, labels_test)


def find_best_K(data, labels):
    """
    find the best hyper parameter K
    :param data:
    :param labels:
    :return:
    """
    for k in range(2, 20):
        F1 = 0
        accn = None
        for cross_train_data, cross_train_labels, validation_data, validation_labels \
                in preprocess.cross_validation(data, labels):
            classifier = KNN(n_neighbor=k)
            classifier.fit(cross_train_data, cross_train_labels)
            predict = classifier.classify(validation_data)
            acc = assessment.assess(predict, validation_labels)
            if acc["F1_Score"] > F1:
                F1 = acc["F1_Score"]
                accn = acc
        assessment.export_to_excel(accuracy=accn, path=export_path,
                                   method=classifier.class_name,
                                   preprocess="std", n_neighbors=k)


def KNN_classfy(data, labels, data_test, labels_test):
    """
    final check on the test set
    :param data:
    :param labels:
    :param data_test:
    :param labels_test:
    :return:
    """
    classifier = KNN(n_neighbor=10)
    classifier.fit(data, labels)
    predict = classifier.classify(data_test)
    accuracy = assessment.assess(predict, labels_test)

    assessment.export_to_excel(accuracy=accuracy, path=export_path,
                               method=classifier.class_name, preprocess="std", N=13)


if __name__ == "__main__":
    main()
