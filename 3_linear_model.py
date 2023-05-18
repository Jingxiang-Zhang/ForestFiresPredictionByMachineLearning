import final.preprocess.read as read
from final.preprocess.read import TRAIN_SET, TEST_SET
from final.preprocess import preprocess
from final.classification.trivial_and_base import Baseline
from final.assessment import assessment

export_path = "linear2.xlsx"


def main():
    data, labels = read.read_all(TRAIN_SET)
    data_test, labels_test = read.read_all(TEST_SET)

    # here, test data will not be used, only used to stuff the function parameter
    # find_best_preprocessing_method(data, labels, data_test, labels_test)
    # finished, and use standardization will get the best performance.

    data, data_test = preprocess.standardization(data, data_test)
    # find_best_extra_features(data, labels)

    linear_classfy(data, labels, data_test, labels_test)


def find_best_extra_features(data, labels):
    """
    adding one feature of all the combination, and find whether the result would change
    :param data:
    :param labels:
    :return:
    """
    for cross_train_data, cross_train_labels, validation_data, validation_labels \
            in preprocess.cross_validation(data, labels):
        for temp_data, temp_labels, temp_test_data, temp_test_labels, operation \
                in preprocess.plus_one_feature_iteration(cross_train_data,
                   cross_train_labels, validation_data, validation_labels):
            classifier = Baseline()
            classifier.class_name = "linear"
            classifier.fit(cross_train_data, cross_train_labels)
            predict = classifier.classify(validation_data)
            acc = assessment.assess(predict, validation_labels)
            assessment.export_to_excel(accuracy=acc, path=export_path,
                                       method=classifier.class_name,
                                       preprocess=operation)


def find_best_preprocessing_method(data, labels, data_test, labels_test):
    """
    find the best preprocessing method, by using all the combination
    of data preprocessing method
    :param data:
    :param labels:
    :param data_test:
    :param labels_test:
    :return:
    """
    for temp_data, temp_labels, temp_test_data, temp_test_labels, strategy in \
            preprocess.all_combination(data, labels, data_test, labels_test):
        f1 = 0
        assem = None
        for cross_train_data, cross_train_labels, validation_data, validation_labels \
                in preprocess.cross_validation(temp_data, temp_labels):
            classifier = Baseline()
            classifier.class_name = "linear"
            classifier.fit(cross_train_data, cross_train_labels)
            predict = classifier.classify(validation_data)
            acc = assessment.assess(predict, validation_labels)
            if acc["F1_Score"] > f1:
                f1 = acc["F1_Score"]
                assem = acc
        assessment.export_to_excel(accuracy=assem, path=export_path,
                                   method=classifier.class_name,
                                   preprocess=str(strategy).lstrip("[").rstrip("]"))


def linear_classfy(data, labels, data_test, labels_test):
    """
    final check on the test set
    :param data:
    :param labels:
    :param data_test:
    :param labels_test:
    :return:
    """
    classifier = Baseline()
    classifier.fit(data, labels)
    predict = classifier.classify(data_test)
    accuracy = assessment.assess(predict, labels_test)

    assessment.export_to_excel(accuracy=accuracy, path=export_path,
                               method=classifier.class_name, preprocess="std")


if __name__ == "__main__":
    main()
