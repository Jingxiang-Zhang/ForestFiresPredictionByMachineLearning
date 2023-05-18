import final.preprocess.read as read
from final.preprocess.read import TRAIN_SET, TEST_SET
import final.assessment.assessment as assessment
import final.classification as classification

export_path = "result.xlsx"


def main():
    # read training data and test data
    data, labels = read.read_all(TRAIN_SET)
    data_test, labels_test = read.read_all(TEST_SET)

    # call trivial model
    # trivial(data, labels, data_test, labels_test)
    # call baseline model
    baseline(data, labels, data_test, labels_test)


def trivial(data, labels, data_test, labels_test):
    # create a trivial classifier
    classifier = classification.Trivial()
    classifier.fit(data, labels)
    # classify test data
    predict = classifier.classify(data_test)
    accuracy = assessment.assess(predict, labels_test)
    print("trivial model:", accuracy)
    # export the result to excel
    assessment.export_to_excel(accuracy=accuracy, path=export_path,
                               method=classifier.class_name, preprocess="None")


def baseline(data, labels, data_test, labels_test):
    # create a baseline classifier
    classifier = classification.Baseline()
    classifier.fit(data, labels)
    # classify test data
    predict = classifier.classify(data_test)
    accuracy = assessment.assess(predict, labels_test)
    print("baseline model:", accuracy)
    # export the result to excel
    assessment.export_to_excel(accuracy=accuracy, path=export_path,
                               method=classifier.class_name, preprocess="None")


if __name__ == "__main__":
    main()
