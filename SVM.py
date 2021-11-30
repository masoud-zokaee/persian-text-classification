from main_builder.main import Main


if __name__ == "__main__":

    svm_train = Main()

    accuracy, report = svm_train.run(
                        input_file="hate_dataset_10.csv",
                        text_column="fulltext",
                        label_column="label",
                        test_size=0.1,
                        random_state=42)
                        
    print(f"\ntotal model accuracy = {accuracy}")

    print("\n", "*"*50)

    print("\npositive results:")
    print(f"precision = {report['0']['precision'] * 100}")
    print(f"recall = {report['0']['recall'] * 100}")
    print(f"f1-score = {report['0']['f1-score'] * 100}")

    print("\nnegative results:")
    print(f"precision = {report['1']['precision'] * 100}")
    print(f"recall = {report['1']['recall'] * 100}")
    print(f"f1-score = {report['1']['f1-score'] * 100}")
