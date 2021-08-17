import pandas as pd
import nltk
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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


def tokenize_custom(text):
    tokenizer = nltk.RegexpTokenizer(r'\w+') #TODO modifier pour enlever les chiffres
    words = tokenizer.tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]

    return words



def delete_words(words, frequent_words):
    new_words = [word for word in words if word not in frequent_words]

    return new_words


def stem_string(words, stemmer):
    words_stemmed = [stemmer.stem(word) for word in words]

    return words_stemmed

def lem_string(words, lemmatizer):
    words_lemz = [lemmatizer.lemmatize(word) for word in words]

    return words_lemz

def process_categs(product_category_tree, level=0):
    product_category_tree = product_category_tree.strip('[')
    product_category_tree = product_category_tree.strip(']')
    product_category_tree = product_category_tree.strip('"')
    product_category_tree = product_category_tree.split(' >> ')
    try:
        return(product_category_tree[level])
    except IndexError:
        n_categs = len(product_category_tree)
        print(f"Category tree only has {n_categs} categs, level must be between 0 and {n_categs}")

def get_bigrams(tokens):
    return list(nltk.bigrams(tokens))


def scale_data(raw_features):
    scaler = StandardScaler()

    data_scale = scaler.fit_transform(raw_features)

    return data_scale

def scree_plot(data_scale, max_comp, savefig=False):
    pca_scree = PCA(n_components=max_comp)
    pca_scree.fit(data_scale)
    plt.style.use('fivethirtyeight')
    plt.xlabel('Nb de Composantes Principales')
    plt.ylabel('Pourcentage de variance expliquée (cumulée)')
    plt.title('Scree plot PCA')
    plt.plot(np.arange(1, pca_scree.n_components_ + 1), pca_scree.explained_variance_ratio_.cumsum() * 100,
             color='#8c704d')
    if savefig:
     plt.savefig('plots/{}.png'.format(savefig), bbox_inches='tight', dpi=720)
    plt.show()


def reduce_dim_pca(data_scale, n_comp):
    pca = PCA(n_components=n_comp)
    data_scale_decomp = pca.fit_transform(data_scale)

    return data_scale_decomp


def train_tsne(data, labels, perplexity=50, show=True, savefig=False):
    tsne = TSNE(init='pca', random_state=41, n_jobs=-1, perplexity=perplexity)
    data_tsne = tsne.fit_transform(data)
    df_data_tsne_labels = pd.DataFrame(data_tsne, columns=['x', 'y']).merge(labels, left_index=True, right_index=True)
    if show:
        plt.style.use('seaborn')
        plt.figure(figsize=(8, 8))
        plt.axis('equal')
        for label in labels.unique():
            group = df_data_tsne_labels[df_data_tsne_labels[labels.name] == label]
            plt.scatter(group.x, group.y, label=label)
        plt.legend()
        plt.show=()
        if savefig:
            plt.savefig('plots/{}.png'.format(savefig), bbox_inches='tight', dpi=720)

    return data_tsne