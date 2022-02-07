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


def lower_tokenized_text(texts):
    # normal_tokens = [[tok.lower() for tok in doc_tokens] for doc_tokens in token_list]
    # lowered = [text.lower() for text in texts]
    # return lowered
    lowered_tokens = [[token.lower() for token in text] for text in texts]
    return lowered_tokens


def remove_tokenized_punctuation(texts):
    # no_punc_tokens = [[tok for tok in word_tokenize(text) if tok not in string.punctuation] for text in texts]
    # no_punc_joined = [' '.join(sent) for sent in no_punc_tokens]
    no_punctuation = [[token for token in text if token not in string.punctuation] for text in texts]
    return no_punctuation


def tokenize_corpus(texts):
    tokenized_corpus = [word_tokenize(text) for text in texts]
    return tokenized_corpus
