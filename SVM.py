import csv
import re
import demoji
import pandas as pd
from PersianStemmer import PersianStemmer
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report


# reading persian stop words list
stopwords_persian = []
with open("stopwords.txt", "r") as input_file:
    for item in input_file:
        stopwords_persian.append(item.strip())

# loading pars bert pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")


# cleaning and tokenizing tweets
def tokenize_tweet(tweet):

    # clean HTTPS
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # clean user names
    tweet = re.sub(r'([@][\w_-]+)', '', tweet)
    # clean English words
    tweet = re.sub(r'[A-Za-z]', '', tweet)
    # clean extra spaces
    tweet = re.sub(r'[ ]{2,}', '', tweet)
    # special chars
    tweet = re.sub(r'[!@#$%^&*(),.?":{}|<>/«»،؛؟…_\-+\\\u200c]', '', tweet)
    # Farsi numbers and Arabic chars
    tweet = re.sub(r'[\u064E-\u0655\u06F0-\u06F9\u0660-\u0669]', '', tweet)
    # phone symbols
    tweet = re.sub(r'[\u2600-\u26FF0-9]', '', tweet)
    # remove all emoticons
    tweet = demoji.replace(tweet)

    # loading persian stemmer
    ps = PersianStemmer()

    # tokenize tweet
    tweet_tokens = tokenizer.tokenize(tweet)

    tweet_words = []

    # removing stop words and stem each word
    for word in tweet_tokens:
        stem_word = ps.stem(word)
        if stem_word not in stopwords_persian:
            tweet_words.append(stem_word)

    # returning full joined sentence
    return ' '.join(word for word in tweet_words)


tweet_set = []

# loading data set .csv file
with open("your_dataset.csv", "r") as input_file:
    csv_reader = csv.reader(input_file)
    next(csv_reader)
    for row in csv_reader:

        # positive tweet
        if row[1] == "1":
            tweet_set.append([tokenize_tweet(row[0]), "pos"])

        # negative tweet
        elif row[1] == "0":
            tweet_set.append([tokenize_tweet(row[0]), "neg"])

# convert input list to pandas data frame
dataset = pd.DataFrame(tweet_set, columns=['tweet', 'label'])

# split input data set to train and test set
x_train, x_test, y_train, y_test = train_test_split(dataset.tweet, dataset.label,
                                                    test_size = 0.1, random_state = 42)


print("number of words in train set : ", len(x_train))
print("number of words in test set : ", len(x_test))
print("\n", "*******************************", "\n")


# encoding target labels to binary scope
Encoder = LabelEncoder()
y_train_le = Encoder.fit_transform(y_train)
y_test_le = Encoder.fit_transform(y_test)


# loading SVM vectorizer
# for higher score use : max_features=5000
vectorizer = TfidfVectorizer(min_df = 1, max_df = 0.95, sublinear_tf = True,
                             use_idf = True, ngram_range = (1, 2))

# fit vectorizer to the whole data set
vectorizer.fit(dataset.tweet)

# transfer train and test data set to TF-IDF vectors
x_train_vec = vectorizer.transform(x_train)
x_test_vec = vectorizer.transform(x_test)


# set up SVM classification parameters
classifier_linear = svm.SVC(C = 1.0, kernel = 'linear', degree = 3, gamma = 'auto')

# fit train data to model
classifier_linear.fit(x_train_vec, y_train_le)

# perform prediction on data set
prediction_linear = classifier_linear.predict(x_test_vec)

print("SVM Accuracy Score -> ", accuracy_score(y_test_le, prediction_linear) * 100)

# classification report for two classes
report = classification_report(y_test_le, prediction_linear, output_dict=True)


print("\n", "*******************************", "\n")

print("pos results:")
print("precision = ", report['0']['precision'] * 100)
print("recall = ", report['0']['recall'] * 100)
print("f1-score = ", report['0']['f1-score'] * 100)

print()

print("neg results:")
print("precision = ", report['1']['precision'] * 100)
print("recall = ", report['1']['recall'] * 100)
print("f1-score = ", report['1']['f1-score'] * 100)

print("\n", "*******************************", "\n")


# you can use set of unseen sentences to classify , stored in .csv file
pos_true_count, pos_false_count = 0, 0
neg_true_count, neg_false_count = 0, 0

with open("your_dataset.csv", "r") as input_test:
    csv_reader = csv.reader(input_test)
    next(csv_reader)
    for row in csv_reader:

        sentence_vector = vectorizer.transform([tokenize_tweet(row[0])])
        result = classifier_linear.predict(sentence_vector)

        if result == 0 and row[1] == "1":
            pos_true_count += 1

        elif result == 1 and row[1] == "0":
            neg_true_count += 1

        elif result == 1 and row[1] == "1":
            pos_false_count += 1

        elif result == 0 and row[1] == "0":
            neg_false_count += 1

    print("number of true predicted: ", pos_true_count + neg_true_count,
          "number of false predicted: ", pos_false_count + neg_false_count)

    print("pos status - true : ", pos_true_count, " false : ", pos_false_count)
    print("neg status - true : ", neg_true_count, " false : ", neg_false_count)


