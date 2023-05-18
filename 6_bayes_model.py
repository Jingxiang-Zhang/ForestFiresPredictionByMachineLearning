import final.preprocess.read as read
from final.preprocess.read import TRAIN_SET, TEST_SET
from final.preprocess import preprocess
from final.classification import Bayes
from final.assessment import assessment

export_path = "bayes.xlsx"


def main():
    data, labels = read.read_all(TRAIN_SET)
    data_test, labels_test = read.read_all(TEST_SET)

    data, data_test = preprocess.standardization(data, data_test)
    Gaussian_naive_Bayes_classfy(data, labels, data_test, labels_test)


def Gaussian_naive_Bayes_classfy(data, labels, data_test, labels_test):
    """
    final check on the test set
    :param data:
    :param labels:
    :param data_test:
    :param labels_test:
    :return:
    """
    classifier = Bayes()
    classifier.fit(data, labels)
    predict = classifier.classify(data_test)
    accuracy = assessment.assess(predict, labels_test)

    assessment.export_to_excel(accuracy=accuracy, path=export_path,
                               method=classifier.class_name, preprocess="std")


if __name__ == "__main__":
    main()
