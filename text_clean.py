import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import udf

data = pd.read_csv('data/Flipkart/flipkart_com-ecommerce_sample_1050.csv')

# Stop words. Stop words are
sw_nltk = stopwords.words('english')
sw_nltk_custom = [
    "key",
    "feature",
    'product',
    'yourselves',
    "should've",
    'yours',
    'to',
    'those',
    'myself',
    's',
    "you're",
    'ourselves',
    've',
    "hadn't",
    'here',
    'of',
    'on',
    're',
    'y',
    'my',
    'that',
    'doing',
    'd',
    "she's",
    'theirs',
    'you',
    'having',
    'has',
    'been',
    "shouldn't",
    'through',
    'your',
    'all',
    'couldn',
    "didn't",
    'this',
    'i',
    'these',
    'needn',
    'why',
    'm',
    'was',
    'we',
    'just',
    'now',
    'how',
    'as',
    'more',
    "won't",
    'are',
    'does',
    'if',
    'his',
    'ain',
    'himself',
    'be',
    'shouldn',
    'they',
    'me',
    'aren',
    'which',
    'during',
    'such',
    "couldn't",
    "mustn't",
    "you'd",
    'then',
    'will',
    'he',
    "that'll",
    'and',
    'but',
    "needn't",
    'being',
    'from',
    'll',
    'hadn',
    'o',
    'themselves',
    'or',
    'the',
    'haven',
    'for',
    'what',
    'ma',
    'their',
    "doesn't",
    'shan',
    'our',
    "hasn't",
    'yourself',
    't',
    "you'll",
    'am',
    'were',
    'while',
    'who',
    "haven't",
    'mightn',
    'down',
    'she',
    "wasn't",
    'again',
    'itself',
    'isn',
    'mustn',
    "it's",
    "weren't",
    "don't",
    'further',
    'hasn',
    'nor',
    "you've",
    'it',
    'wasn',
    'them',
    'can',
    'have',
    'a',
    "mightn't",
    'until',
    'r',
    'weren',
    'by',
    'whom',
    "wouldn't",
    'when',
    'is',
    'an',
    'don',
    'herself',
    'its']
sw_to_keep_anyway = ['ours',
                     'him',
                     'her',
                     'hers',
                     'had',
                     'do',
                     'did',
                     'because',
                     'at',
                     'with',
                     'about',
                     'against',
                     'between',
                     'into',
                     'before',
                     'after',
                     'above',
                     'below',
                     'up',
                     'in',
                     'out',
                     'off',
                     'over',
                     'under',
                     'once',
                     'there',
                     'where',
                     'any',
                     'both',
                     'each',
                     'few',
                     'most',
                     'other',
                     'some',
                     'no',
                     'not',
                     'only',
                     'own',
                     'same',
                     'so',
                     'than',
                     'too',
                     'very',
                     'should',
                     "aren't",
                     'didn',
                     'doesn',
                     "isn't",
                     "shan't",
                     'won',
                     'wouldn']
sw_nltk_custom_bis = [word for word in sw_nltk if word not in sw_to_keep_anyway]
# See the end of the script to understand the detected stopwords or go to udf.detect_useless_words
sw_detected = ['cash',
 'with',
 'flipkart',
 'delivery',
 'at',
 'guarantee',
 'day',
 'replacement',
 'in',
 'buy',
 'com',
 'genuine',
 'free',
 'shipping',
 'price',
 'online',
 'only']
# sw_detected = []
sw = sw_nltk_custom_bis + sw_detected

# Get Categories
data['label_categ_0'] = data.product_category_tree.apply(udf.process_categs,
                                                         level=0)


#  have 7 categs with uniform distribution
# When going one level deeper in the tree we get 62 categs with lot having one item

# We are going to use TFIDF Vectorizer to create the features.
# This Sklearn object uses a tokenizing function and creates the n_grams

def custom_tokenize(text):
    """
    Tokenizes, deletes the words that are'nt fully alphabetical and apply a lemmatizer

    :param text: str. The text we need to tokenize into simple tokens
    :return: tokens_lem: list.
    """

    # Creation of alphanum tokens
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)

    # Deleting numbers
    words = [word.lower() for word in words if word.isalpha()]

    # Getting lemms
    lemmatizer_eng = WordNetLemmatizer()
    tokens_lem = udf.lem_string(words, lemmatizer=lemmatizer_eng)

    return tokens_lem


data['description_tok'] = data.description.apply(custom_tokenize)
data['description_tok'] = data.description_tok.apply(udf.delete_words, frequent_words=sw)
all_words = udf.get_all_words(data.description_tok)

tfidf = TfidfVectorizer(tokenizer=custom_tokenize, stop_words=sw, ngram_range=(1, 2))
values = tfidf.fit_transform(data.description)

dense = values.todense()
feature_names = tfidf.get_feature_names()
denselist = dense.tolist()
data_raw = pd.DataFrame(denselist, columns=feature_names).reset_index()

data_full = data.merge(data_raw, how='left', right_on='index', left_index=True)

data_scale = udf.scale_data(data_raw)

udf.scree_plot(data_scale, 1050, savefig=False)

data_scale_decomp = udf.reduce_dim_pca(data_scale, 700)

data_tsne = udf.train_tsne(data_scale_decomp, data.label_categ_0, learning_rate=600)

# most_baby = udf.get_most_frequent_per_categ(data, 'Baby Care', 20)

df_useless = udf.detect_useless_words(data, 20)
detected_useless = list(df_useless[df_useless.nb_apperances >=3].index)


# TODO faire un transfert learning? VGGNET/RESNET images. BERT : texte
