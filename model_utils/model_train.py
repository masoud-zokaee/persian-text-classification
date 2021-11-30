from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report


class ModelTrain:

	def __init__(self):

		self.label_encoder = self.load_encoder()

		self.vectorizer = self.load_vectorizer()

		self.classifier = self.load_classifier()

	def train(self, x_train, y_train):

		# encoding target labels to binary scope
		y_train_le = self.label_encoder.fit_transform(y_train)

		# fit vectorizer to train set
		self.vectorizer.fit(x_train)

		# transfer train data set to TF-IDF vectors
		x_train_vec = self.vectorizer.transform(x_train)

		# fit train data to model
		self.classifier.fit(x_train_vec, y_train_le)

	def predict(self, x_test, y_test):

		# encoding target labels to binary scope
		y_test_le = self.label_encoder.fit_transform(y_test)

		# transfer test data set to TF-IDF vectors
		x_test_vec = self.vectorizer.transform(x_test)

		# perform prediction on test data set
		prediction = self.classifier.predict(x_test_vec)

		accuracy = accuracy_score(y_test_le, prediction) * 100

		report = classification_report(y_test_le, prediction, output_dict=True)

		return accuracy, report

	def load_encoder(self):

		label_encoder = LabelEncoder()

		return label_encoder

	def load_vectorizer(self):

		# for higher score use : max_features=5000
		vectorizer = TfidfVectorizer(min_df = 1, max_df = 0.95, sublinear_tf = True,
									 use_idf = True, ngram_range = (1, 2))

		return vectorizer

	def load_classifier(self):

		classifier = svm.SVC(C = 1.0, kernel = 'linear', degree = 3, gamma = 'auto')

		return classifier
