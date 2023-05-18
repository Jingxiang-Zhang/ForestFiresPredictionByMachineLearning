import final.preprocess.read as read
from final.preprocess.read import TRAIN_SET, TEST_SET
from final.show import show
from final.preprocess import preprocess

export_path = "show.xlsx"


def main():
    data, labels = read.read_all(TRAIN_SET)
    data_test, labels_test = read.read_all(TEST_SET)

    # show original data
    # show.export_data_to_excel(data, export_path)

    # show standardization data
    # std_data = preprocess.standardization(data)
    # show.export_data_to_excel(std_data, export_path)

    # show PCA
    # pca_data,test_data = preprocess.PCA_decomposition(data,data_test, components=0.9)
    # show.export_data_to_excel(pca_data, export_path)

    # adding more features
    data, labels, data_test, labels_test = \
        preprocess.plus_last_n_date_feature(data, labels, data_test, labels_test, days=5,
                                            operation=[("Temperature", "AVE"),
                                                       ("Ws", "MAX"),
                                                       ("Rain", "MIN")])
    data, labels, data_test, labels_test = preprocess.delete_night \
        (data, labels, data_test, labels_test)
    data = data.drop("Temperature", axis=1)
    data = data.drop("Ws", axis=1)
    data = data.drop("Rain", axis=1)
    data = data.drop("RH", axis=1)
    data = data.drop("FFMC", axis=1)
    data = data.drop("DMC", axis=1)
    data = data.drop("DC", axis=1)
    data = data.drop("ISI", axis=1)
    data = data.drop("BUI", axis=1)
    show.export_data_to_excel(data, export_path)


if __name__ == "__main__":
    main()
