import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from datetime import date

import udf

PATH_DATA = 'data/'
DATA_FILE_NAME = 'Flipkart/flipkart_com-ecommerce_sample_1050.csv'
# We are going to use TFIDF Vectorizer to create the features.
# This Sklearn object uses a tokenizing function and creates the n_grams

def custom_tokenize(text, mode="lemma"):
    """
    Tokenizes, deletes the words that aren't fully alphabetical and apply a lemmatiser or a stemmer

    :param text: str. The text we need to tokenize into simple tokens
    :param mode: str. Default is "lemma". Chose to use either stemming or lemming after tokenize
    :return: tokens: list.
    """

    # Creation of alphanum tokens
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)

    # Deleting numbers
    words = [word.lower() for word in words if word.isalpha()]
    if mode == "lemma":
        # Getting lemms
        lemmatizer_eng = WordNetLemmatizer()
        tokens = udf.lem_string(words, lemmatizer=lemmatizer_eng)
    elif mode == "stemm":
        stemm_eng = SnowballStemmer("english")
        tokens = udf.stem_string(words, stemmer=stemm_eng)
    else:
        raise "Enter ether mode='stemm' or 'lemm'"

    return tokens


def main(vectorizer=None):

    # data = pd.read_csv('data/Flipkart/flipkart_com-ecommerce_sample_1050.csv')
    data = pd.read_csv(PATH_DATA + DATA_FILE_NAME)
    # Stop words. Stop words are from nltk standard english stopwords library
    sw_nltk = stopwords.words('english')
    # I chose to keep a few stopwords anyway
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
    sw_nltk_custom = [word for word in sw_nltk if word not in sw_to_keep_anyway]
    # Detected stopwords are words that are at the top of the most frequent words in more than 2 category classifications
    # go at the end of the script or to udf.detect_useless_words
    sw_detected = ['with',
     'day',
     'genuine',
     'delivery',
     'cash',
     'product',
     'r',
     'buy',
     'free',
     'at',
     'only',
     'shipping',
     'in']
    # sw_detected = []
    sw = sw_nltk_custom + sw_detected

    # Get Categories
    data['label_categ_0'] = data.product_category_tree.apply(udf.process_categs,
                                                             level=0)
    # We have 7 categs with uniform distribution
    # When going one level deeper in the tree we get 62 categs with lot having one item
    # Here we reproduce the work done by the tokenizer
    data['description_tok'] = data.description.apply(custom_tokenize)
    data['description_tok'] = data.description_tok.apply(udf.delete_words, frequent_words=sw)
    all_words = udf.get_all_words(data.description_tok)

    # Getting the features
    if vectorizer is None:
        vect_model = TfidfVectorizer(tokenizer=custom_tokenize, stop_words=sw, ngram_range=(1, 2))
        values = vect_model.fit_transform(data.description)
    elif vectorizer.lower() == 'tfidf':
        vect_model = TfidfVectorizer(tokenizer=custom_tokenize, stop_words=sw, ngram_range=(1, 2))
        values = vect_model.fit_transform(data.description)
    elif vectorizer.lower() == 'count':
        vect_model = CountVectorizer(tokenizer=custom_tokenize, stop_words=sw, ngram_range=(1, 2))
        values = vect_model.fit_transform(data.description)
    else:
        print(f"Invalid entry for vectorizer param : {vectorizer}, using Tf Idf insted")
        vect_model = TfidfVectorizer(tokenizer=custom_tokenize, stop_words=sw, ngram_range=(1, 2))
        values = vect_model.fit_transform(data.description)

    dense = values.todense()
    feature_names = vect_model.get_feature_names()
    denselist = dense.tolist()
    data_raw = pd.DataFrame(denselist, columns=feature_names).reset_index()
    data_full = data.merge(data_raw, how='left', right_on='index', left_index=True)

    # Scaling and reducing dimension with StandardScaler and PCA
    data_scale = udf.scale_data(data_raw)
    udf.scree_plot(data_scale, 1050, savefig='scree_plot')
    n_dim = 700
    data_scale_decomp = udf.reduce_dim_pca(data_scale, n_dim)

    data_to_save = pd.DataFrame(data_scale_decomp).merge(data.label_categ_0,
                                                         left_index=True,
                                                         right_index=True).rename(columns={'label_categ_0': 'label'})
    pd.DataFrame(data_to_save).to_csv(PATH_DATA + f"count_vect_decomp_{str(date.today()).replace('-', '')}_{n_dim}comp.csv")
    # Visualization with TSNE
    data_tsne = udf.train_tsne(data_scale_decomp, data.label_categ_0, learning_rate=600)

    # Detecting words that are frequent in various categories
    # df_useless = udf.detect_useless_words(data, 20)
    # detected_useless = list(df_useless[df_useless.nb_apperances >=4].index)

    print("Little example")
    print("let's tokenize and skip stopwords from a sentence")
    print("here's the original sentence: ", )
    sentence = "Cats shouldn't be trying to mess up with humans' lives in any way &AB / 41hours"
    print(sentence, '\n')
    print("result with lemmatizer:")
    print(udf.delete_words(custom_tokenize(sentence), sw), "\n")
    print("result with stemmer:")
    print(udf.delete_words(custom_tokenize(sentence, mode="stemm"), sw))

if __name__ == '__main__':
    main(vectorizer="count")


# TODO faire un transfert learning? VGGNET/RESNET images. BERT : texte
