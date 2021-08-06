import pandas as pd
import nltk


def short_in_long_rate(row):
    """
    Returns rate of elements in a short list present in a larger one
    :param row: pandas.Series
    :return rate: similarity rate
    """
    short = row.name_tok
    long = row.description_tok
    nb_words_from_short_in_long = 0
    nb_words_from_short = len(short)
    for word in short:
        if word in long:
            nb_words_from_short_in_long += 1
    rate = nb_words_from_short_in_long / nb_words_from_short
    return rate


def get_all_words(descriptions):
    all_words = []
    for text in descriptions:
        all_words += text
    all_words = pd.Series(all_words)

    return all_words


def clean_non_alphanum(text):
    tokenizer = nltk.RegexpTokenizer(r'\w+')

    return tokenizer.tokenize(text)


def delete_words(words, frequent_words):
    new_words = [word for word in words if word not in frequent_words]

    return new_words


def stem_string(words, stemmer):
    words_stemmed = [stemmer.stem(word) for word in words]

    return words_stemmed