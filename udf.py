import pandas as pd
import nltk
import numpy as np
import image_clean
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import time


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


def train_tsne(data, labels, perplexity=50, learning_rate=200, show=True, savefig=False):
    tsne = TSNE(init='pca', random_state=41, n_jobs=-1, perplexity=perplexity, learning_rate=learning_rate)
    data_tsne = tsne.fit_transform(data)
    df_data_tsne_labels = pd.DataFrame(data_tsne, columns=['x', 'y']).merge(labels, left_index=True, right_index=True)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if show:
        plt.style.use('seaborn')
        plt.figure(figsize=(8, 8))
        plt.axis('equal')
        for label, color in zip(labels.unique(), colors):
            group = df_data_tsne_labels[df_data_tsne_labels[labels.name] == label]
            plt.scatter(group.x,
                        group.y,
                        label=label,
                        c=color,
                        s=12)
        plt.legend()
        plt.show=()
        if savefig:
            plt.savefig('plots/{}.png'.format(savefig), bbox_inches='tight', dpi=720)

    return data_tsne


def get_most_frequent_per_categ(data, categ, qtty):
    sub_data = data[data.label_categ_0 == categ]
    sub_data_all_words = get_all_words(sub_data.description_tok)

    return sub_data_all_words.value_counts().head(qtty)


def detect_useless_words(data, qtty):
    dict_frequent = {}
    for categ in data.label_categ_0.unique():
        frequent_words = get_most_frequent_per_categ(data, categ, qtty).index
        dict_frequent[categ] = frequent_words

    most_frequent_all = np.concatenate(list(dict_frequent.values()))
    df = pd.DataFrame(index=list(set(most_frequent_all)), columns=data.label_categ_0.unique())

    for frequent_word in list(set(most_frequent_all)):
        for categ in data.label_categ_0.unique():
            df.loc[frequent_word, categ] = frequent_word in dict_frequent[categ]

    df['nb_apperances'] = df.apply(lambda x : x.sum(), axis=1)

    return df


def tuning_kmeans(data, range_n_clust, show=True, savefig=False):

    slh_scores = {}
    db_scores = {}

    for n_clust in range_n_clust:

        start = time.time()
        model_k_means = KMeans(n_clusters=n_clust)
        print(f"{model_k_means} start training")
        labels = model_k_means.fit_predict(data)
        slh = silhouette_score(data, labels)
        db = davies_bouldin_score(data, labels)
        elapsed = time.time() - start
        print(f"{model_k_means} trained in {elapsed}s")
        print(f"Slh score : {slh}\nDB score : {db}")
        slh_scores[n_clust] = slh
        db_scores[n_clust] = db

    if show:
        fig, ax = plt.subplots(2,1, sharex=True)

        ax[0].plot(slh_scores.keys(), slh_scores.values(), color="#8c704d")
        ax[0].set_ylabel('Silhouette_score')
        ax[1].plot(db_scores.keys(), db_scores.values(), color="#8c704d")
        ax[1].set_ylabel('Davies Bouldin Score')

        plt.title("Scores against Nb Cluster")
        if savefig:
            plt.savefig('plots/{}.png'.format(savefig), bbox_inches='tight', dpi=720)

        plt.show()

    return slh_scores, db_scores


def collect_all_features(path_image):
    original_dir = os.getcwd()
    os.chdir(path_image)

    full_features = pd.DataFrame()
    for file_name in os.listdir():
        descriptors = image_clean.get_descriptors(file_name, equalize=True, show_pre_process_review=False)

        file_df = pd.DataFrame(descriptors)
        try:
            file_df['image_id'] = pd.Series([file_name] * len(descriptors))
            full_features = full_features.append(file_df)
        except TypeError as e:
            print(e)
            print(file_name)
    os.chdir(original_dir)
    return full_features

def train_model_kmeans(data_train, n_clusters):
    data_train_copy = data_train.copy()
    model = KMeans(n_clusters=n_clusters, init="k-means++")
    predicted_labels = model.fit_predict(data_train_copy.drop(columns=['label']))
    data_train_copy['predicted_label'] = predicted_labels

    return data_train_copy


def train_model_GB(data_train):
    X_train, X_test, y_train, y_test = \
        train_test_split(data_train.drop(columns=['label']),
                         data_train.label,
                         test_size=0.3,
                         random_state=41)
    model = GradientBoostingClassifier(random_state=41)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # score = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion_matx = confusion_matrix(y_test, y_pred)
    print(report)
    print(confusion_matx)
    X_test['label_pred'] = y_pred
    return X_test