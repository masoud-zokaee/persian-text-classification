import re
import demoji
from pathlib import Path
from transformers import AutoTokenizer
from PersianStemmer import PersianStemmer


class PreProcess:

	def __init__(self):

		stopwords_path = Path(__file__).resolve().parent.joinpath("stopwords.txt")

		self.stopwords = self.load_stopwords(file_path=stopwords_path)

		self.tokenizer = self.load_tokenizer(model_path="HooshvareLab/bert-base-parsbert-uncased")

		self.stemmer = self.load_stemmer()

	def preprocess_text(self, text):

		processed_tokens = []

		clean_text = self.text_cleaner(text=text)

		tokens = self.tokenizer.tokenize(clean_text)

		for token in tokens:

			stem_word = self.stemmer.stem(token)

			if stem_word not in self.stopwords:

				processed_tokens.append(stem_word)

		processed_text = ' '.join(token for token in processed_tokens)

		return processed_text

	def load_stopwords(self, file_path):

		stopwords = []

		with open(file_path, "r") as input_file:

			for word in input_file:

				stopwords.append(word.strip())

		return stopwords        

	def load_tokenizer(self, model_path):

		tokenizer = AutoTokenizer.from_pretrained(model_path)

		return tokenizer

	def load_stemmer(self):

		stemmer = PersianStemmer()

		return stemmer

	def text_cleaner(self, text):

		# URL
		text = re.sub(r'((?:https?:\/\/|www\.)(?:[-a-zA-Z0-9]+\.)*[-a-zA-Z0-9]+.*)', '', text)

		# Usernames
		text = re.sub(r'([@][\w_-]+)', '', text)

		# English words
		text = re.sub(r'[A-Za-z]', '', text)

		# Extra spaces
		text = re.sub(r'[ ]{2,}', '', text)

		# Special chars
		text = re.sub(r'[!@#$%^&*(),.?":{}|<>/«»،؛؟…_\-+\\\u200c]', '', text)

		# Farsi numbers and Arabic chars
		text = re.sub(r'[\u064E-\u0655\u06F0-\u06F9\u0660-\u0669]', '', text)

		# Phone symbols
		text = re.sub(r'[\u2600-\u26FF0-9]', '', text)

		# Remove all emoticons
		text = demoji.replace(text)

		return text
