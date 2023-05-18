import final.preprocess.read as read
from final.preprocess.read import TRAIN_SET, TEST_SET
from final.show import show
from final.preprocess import preprocess
from final.classification import SVM
from final.assessment import assessment
import numpy as np

export_path = "SVM2.xlsx"


def main():
    data, labels = read.read_all(TRAIN_SET)
    data_test, labels_test = read.read_all(TEST_SET)

    # using standardization will improve the result
    data, data_test = preprocess.standardization(data, data_test)

    # linear kernel
    # SVM_linear(data, labels, data_test, labels_test)

    # polynomial kernel parameter selection
    # SVM_Poly_parameter_select(data, labels)

    # the best hyper parameter is coefficient=0.5, gamma=0.03, and degree=3
    # SVM_Poly(data, labels, data_test, labels_test)

    # RBF kernel parameter selection
    # dSVM_RBF_parameter_select(data, labels)

    # the best hyper parameter is gamma=0.02
    # SVM_RBF(data, labels, data_test, labels_test)

    # Sigmoid kernel parameter selection
    # SVM_Sigmoid_parameter_select(data, labels)

    # the best hyper parameter is gamma=0.1
    # SVM_Sigmoid(data, labels, data_test, labels_test)


def SVM_Sigmoid(data, labels, data_test, labels_test):
    """
    check on the test set
    :param data:
    :param labels:
    :param data_test:
    :param labels_test:
    :return:
    """
    classifier = SVM(kernel="sigmoid", gamma=0.1)
    classifier.fit(data, labels)
    predict = classifier.classify(data_test)
    acc = assessment.assess(predict, labels_test)

    assessment.export_to_excel(accuracy=acc, path=export_path,
                               method=classifier.class_name,
                               preprocess="std", kernel="sigmoid", gamma=0.1)


def SVM_Sigmoid_parameter_select(data, labels):
    """
    select the best parameter of sigmoid kernel, in validation set.
    :param data:
    :param labels:
    :return:
    """
    for coef0 in (-0.5, -0.2, 0, 0.2, 0.5):
        for gamma in (0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5):
            F1 = 0
            accn = None
            for cross_train_data, cross_train_labels, validation_data, validation_labels \
                    in preprocess.cross_validation(data, labels):
                classifier = SVM(kernel="sigmoid", gamma=gamma, coef0=coef0)
                classifier.fit(cross_train_data, cross_train_labels)
                predict = classifier.classify(validation_data)
                acc = assessment.assess(predict, validation_labels)
                if acc["F1_Score"] > F1:
                    F1 = acc["F1_Score"]
                    accn = acc
            assessment.export_to_excel(accuracy=accn, path=export_path,
                                       method=classifier.class_name,
                                       preprocess="std", kernel="sigmoid", coef=coef0,
                                       gamma=gamma)


def SVM_RBF(data, labels, data_test, labels_test):
    """
    check on the test set
    :param data:
    :param labels:
    :param data_test:
    :param labels_test:
    :return:
    """
    classifier = SVM(kernel="rbf", gamma=0.02)
    classifier.fit(data, labels)
    predict = classifier.classify(data_test)
    acc = assessment.assess(predict, labels_test)

    assessment.export_to_excel(accuracy=acc, path=export_path,
                               method=classifier.class_name,
                               preprocess="std", kernel="RBF", gamma=0.02)


def SVM_RBF_parameter_select(data, labels):
    """
    select the best RBF parameter
    :param data:
    :param labels:
    :return:
    """
    for gamma in (0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5):
        F1 = 0
        accn = None
        for cross_train_data, cross_train_labels, validation_data, validation_labels \
                in preprocess.cross_validation(data, labels):
            classifier = SVM(kernel="rbf", gamma=gamma)
            classifier.fit(cross_train_data, cross_train_labels)
            predict = classifier.classify(validation_data)
            acc = assessment.assess(predict, validation_labels)
            if acc["F1_Score"] > F1:
                F1 = acc["F1_Score"]
                accn = acc
        assessment.export_to_excel(accuracy=accn, path=export_path,
                                   method=classifier.class_name,
                                   preprocess="std", kernel="RBF", gamma=gamma)


def SVM_Poly(data, labels, data_test, labels_test):
    """
    check on the test set
    :param data:
    :param labels:
    :param data_test:
    :param labels_test:
    :return:
    """
    classifier = SVM(kernel="poly", degree=3, gamma=0.03, coef0=0.5)
    classifier.fit(data, labels)
    predict = classifier.classify(data_test)
    acc = assessment.assess(predict, labels_test)

    assessment.export_to_excel(accuracy=acc, path=export_path,
                               method=classifier.class_name,
                               preprocess="std", kernel="poly", coef=0.5,
                               gamma=0.03, degree=3)


def SVM_Poly_parameter_select(data, labels):
    """
    select the best parameter of polynomial kernel, in validation set.
    :param data:
    :param labels:
    :return:
    """
    for coef0 in (-0.5, 0, 0.5):
        for gamma in (0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5):
            for degree in (2, 3):
                F1 = 0
                accn = None
                for cross_train_data, cross_train_labels, validation_data, validation_labels \
                        in preprocess.cross_validation(data, labels):
                    classifier = SVM(kernel="poly", degree=degree, gamma=gamma, coef0=coef0)
                    classifier.fit(cross_train_data, cross_train_labels)
                    predict = classifier.classify(validation_data)
                    acc = assessment.assess(predict, validation_labels)
                    if acc["F1_Score"] > F1:
                        F1 = acc["F1_Score"]
                        accn = acc
                assessment.export_to_excel(accuracy=accn, path=export_path,
                                           method=classifier.class_name,
                                           preprocess="std", kernel="poly", coef=coef0,
                                           gamma=gamma, degree=degree)


def SVM_linear(data, labels, data_test, labels_test):
    """
    SVM use linear kernel
    :param data:
    :param labels:
    :param data_test:
    :param labels_test:
    :return:
    """
    classifier = SVM(kernel="linear")
    classifier.fit(data, labels)
    predict = classifier.classify(data_test)
    acc = assessment.assess(predict, labels_test)
    assessment.export_to_excel(accuracy=acc, path=export_path,
                               method=classifier.class_name,
                               preprocess="std", kernel="linear")


if __name__ == "__main__":
    main()
