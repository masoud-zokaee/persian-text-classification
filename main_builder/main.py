from progressbar import ProgressBar
from file_utils.file_handler import FileHandler
from preprocess_utils.preprocess import PreProcess
from model_utils.model_train import ModelTrain


class Main:

	def __init__(self):

		self.file_handler = FileHandler()

		self.model_train = ModelTrain()

		self.preprocess = PreProcess()

	def run(self, input_file, text_column, label_column, test_size, random_state):

		print("\nReading dataset")

		dataset = self.file_handler.read_data(
					file_path=input_file, 
					text_column=text_column, 
					label_column=label_column)

		print("\nPreprocessing dataset")

		with ProgressBar(max_value=dataset.shape[0]) as bar:

			for index, text in enumerate(dataset[text_column]):

				processed_text = self.preprocess.preprocess_text(text)

				dataset.at[index, text_column] = processed_text

				bar.update(index)

		x_train, x_test, y_train, y_test = self.file_handler.split_data(
											dataset=dataset,
											text_column=text_column,
											label_column=label_column,
											test_size=test_size,
											random_state=random_state)

		print("\nTraining model")

		self.model_train.train(x_train=x_train, y_train=y_train)

		accuracy, report = self.model_train.predict(x_test=x_test, y_test=y_test)

		return accuracy, report
