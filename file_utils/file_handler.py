import pandas
from sklearn.model_selection import train_test_split

class FileHandler:

	def __init__(self):

		pass

	def read_data(self, file_path: str, text_column: str, label_column: str):

		dataset = pandas.read_csv(file_path)

		dataset = dataset[[text_column, label_column]]

		return dataset

	def split_data(self, dataset, text_column: str, label_column: str, test_size: float, random_state: int):

		x_train, x_test, y_train, y_test = train_test_split(dataset[text_column], dataset[label_column],
                                                    		test_size = test_size, random_state = random_state)

		return x_train, x_test, y_train, y_test
		