from file_utils.file_handler import FileHandler
from preprocess_utils.preprocess import PreProcess
from model_utils.model_train import ModelTrain


class Main:

    def __init__(self):

        self.file_handler = FileHandler()

        self.preprocess = PreProcess()

        self.model_train = ModelTrain()

    def run(self):

        dataset = self.file_handler.read_data(file_path="hate_dataset_10.csv", 
                                              text_column="fulltext", 
                                              label_column="label")

        for text in dataset["fulltext"]:

            self.preprocess.preprocess_text(text)

        x_train, x_test, y_train, y_test = self.file_handler.split_data(dataset=dataset,
                                                                        text_column="fulltext",
                                                                        label_column="label",
                                                                        test_size=0.1,
                                                                        random_state=42)

        self.model_train.train(x_train=x_train, y_train=y_train)

        accuracy, report = self.model_train.predict(x_test=x_test, y_test=y_test)

        
