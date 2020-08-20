# Farsi text classification

Binary text classification on Farsi textual data set ( review, tweet, etc. ).

## Project description

Using the SVM approach to classify Farsi (Persian) sentences into two classes. implemented python code
provides an efficient text preprocessing on input data set to achieve a high score on prediction.

Useful libraries for preprocessing task are :

 1- ParsBert pre-trained tokenizer
 2- PersianStemmer
 3- Set of 1336 Persian stop words
 4- Set of regular expressions
 5- Demoji ( emoji remover )
 
## Prerequisites

Required python libraries 

1- csv 2- re 3- demoji 4- pandas 5- PersianStemmer 6- transformers 7- sklearn

installing PersianStemmer

```
pip install PersianStemmer

pip install https://github.com/htaghizadeh/PersianStemmer-Python/archive/master.zip --upgrade

```

## Files description 

1- SVM.py : project's python file provided with inline code description  

2- stopwords.txt : a text file containing Persian stop words
