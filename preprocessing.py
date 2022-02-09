import csv
import json
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')


def clean_text(text_str, stem=False, custom_stopwords=[]):
    tokens = [tok.lower().translate(str.maketrans('', '', string.punctuation)) for tok in word_tokenize(text_str)]
    return " ".join([lemmatizer.lemmatize(tok) if stem else tok for tok in tokens if
                     not tok in stopwords and
                     tok.isalpha() and
                     "amp" not in tok and
                     tok not in custom_stopwords])


def wn_lemmatizer_tokenized_text(texts):
    lemmatized_tokens = [[lemmatizer.lemmatize(token) for token in text] for text in texts]
    return lemmatized_tokens


def lower_tokenized_text(texts):
    lowered_tokens = [[token.lower() for token in text] for text in texts]
    return lowered_tokens


def only_alpha_tokenized_text(texts):
    alpha_only_tokens = [[token for token in text if token.isalpha()] for text in texts]
    return alpha_only_tokens


def remove_stopwords_from_tokenized_text(texts, custom_stopwords, default_stopwords=True):
    # Pass empty list if no custom stopwords
    stopwords_to_remove = []
    if default_stopwords:
        stopwords_to_remove = custom_stopwords + stopwords
    tokens_stopwords_removed = [[token for token in text if token not in stopwords_to_remove] for text in texts]
    return tokens_stopwords_removed

def remove_tokenized_punctuation(texts):
    no_punctuation = [[token for token in text if token not in string.punctuation] for text in texts]
    return no_punctuation


def tokenize_corpus(texts):
    tokenized_corpus = [word_tokenize(text) for text in texts]
    return tokenized_corpus
