from final.preprocess.read import read_all, TRAIN_SET
from final.show.show import show_feature_labels_one_by_one, show_correlation, \
    export_correlation_coefficient_to_excel

correlation_coefficient_export_path = "CC.xlsx"


def main():
    data, labels = read_all(TRAIN_SET)
    # show all data
    show_feature_labels_one_by_one(data, labels)
    # show correlation
    show_correlation(data)
    # show correlation coefficient matrix
    export_correlation_coefficient_to_excel(data, correlation_coefficient_export_path)


if __name__ == "__main__":
    main()
