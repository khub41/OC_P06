import mlflow
import numpy as np

import image_clean
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import os
from numpy import zeros
import udf

PATH_IMAGES = 'data/Flipkart/Images'

PATH_DATA = 'data/'
DATA_FILE_NAME = 'Flipkart/flipkart_com-ecommerce_sample_1050.csv'

DESC_FILE_NAME = "descriptors_all_new_preprocess_1709.csv"
LABELED_DESC_FILE_NAME = "labeled_descriptors.csv"

TEXT_DATA_FILE_NAME = 'decomp_20210910_700comp.csv'
#### GETTING DESCRIPTORS
# descriptors_all = udf.collect_all_features(PATH_IMAGES) # COMPUTING
# descriptors_all.to_csv("data/descriptors_all_new_preprocess_3009.csv") # SAVING
descriptors_all = pd.read_csv(PATH_DATA + DESC_FILE_NAME, index_col=[0]) # IMPORTING

## Scaling descriptors and reducing dimension
descriptors_all_scale = udf.scale_data(descriptors_all.drop(columns=['image_id']))
# udf.scree_plot(descriptors_all_scale, 128)
descriptors_all_scale = udf.reduce_dim_pca(descriptors_all_scale, 80)

#### TSNE VISUALISATION
# udf.train_tsne(descriptors_all_scale, pd.Series(zeros(descriptors_all_scale.shape[0]), name='label'),
#                perplexity=80,
#                savefig="tnse_descriptors_80")

#### KMEANS TUNING
# slh_scores, db_scores = udf.tuning_kmeans(descriptors_all_scale,
#                                           range(260,310, 10),
#                                           "k_means_descriptors",
#                                           show=True,
#                                           savefig="slh_db_range_260_300_10_date_2209")

#### DBSCAN TRAINING
# new_data = udf.train_dbscan(descriptors_all_scale, 'test')
# print(new_data.predicted_label.value_counts())

#### CHOOSING KMEANS 30 CLUSTERS
# new_data = udf.train_model_kmeans(descriptors_all_scale, 30)
# print(new_data.predicted_label.value_counts())
# desc_labeled = descriptors_all.merge(new_data,
#                       left_index=True,
#                       right_index=True)[["predicted_label", 'image_id']]\
#
# desc_labeled.to_csv("labeled_descriptors.csv")

#### TSNE VISUALISATION WITH CLUSTERS
# data_tsne = udf.train_tsne(new_data.drop(columns=["predicted_label"]).values,
#                            new_data.predicted_label,
#                            perplexity=50,
#                            savefig="tnse_descriptors_50_color")


#### COLLECTING LABELS
desc_labeled = pd.read_csv(PATH_DATA + LABELED_DESC_FILE_NAME, index_col=[0])

#### CONVERSION INTO TOKENS
tokens = desc_labeled.groupby('image_id').agg({'predicted_label': list})
tokens.columns = ["descriptors"]
tokens['nb_desc'] = tokens.descriptors.apply(lambda x: len(x))

#### VECTORIZING TOKENS
## Counting each "word"
counts_matrix = tokens.descriptors.apply(udf.get_counts)

## TfIdfVectorizer
vectorizer = TfidfTransformer()
tf_idf_matrix = pd.DataFrame(vectorizer.fit_transform(counts_matrix.values).todense(), index=counts_matrix.index)

#### GETTING GROUND LABELS
original_data = pd.read_csv(PATH_DATA + DATA_FILE_NAME)
original_data['label_categ_0'] = original_data.product_category_tree.apply(udf.process_categs,
                                                                            level=0)
dict_label_num = dict([(label, num) for label, num in zip(original_data.label_categ_0.unique(),
                                                          list(range(7)))])
## Converting labels into numerical values
original_data['label_categ_0_num'] = original_data.label_categ_0.replace(dict_label_num)

#### TRAINING UNSUPERVISED (KMEANS)
tokens['label_K_Means_tfidf'] = KMeans(n_clusters=7).fit_predict(StandardScaler().fit_transform(tf_idf_matrix))
tokens['label_K_Means_counts'] = KMeans(n_clusters=7).fit_predict(StandardScaler().fit_transform(counts_matrix))
tokens = tokens.merge(original_data[['label_categ_0_num', 'image']],
             right_on='image',
             left_index=True).set_index('image')

## Performance scores

print(tokens['label_K_Means_tfidf'].value_counts())
print(tokens['label_K_Means_counts'].value_counts())

ARI_tfidf = adjusted_rand_score(tokens.label_categ_0_num, tokens.label_K_Means_tfidf)
AMI_tfidf = adjusted_mutual_info_score(tokens.label_categ_0_num, tokens.label_K_Means_tfidf)

ARI_counts = adjusted_rand_score(tokens.label_categ_0_num, tokens.label_K_Means_counts)
AMI_counts = adjusted_mutual_info_score(tokens.label_categ_0_num, tokens.label_K_Means_counts)

print("ARI and AMI scores with tfidf: ", ARI_tfidf, AMI_tfidf)
print("ARI and AMI scores with counts: ", ARI_counts, AMI_counts)

## TRAINING SUPERVISED (GB)
tf_idf_matrix['label'] = tokens['label_categ_0_num'].values
counts_matrix['label'] = tokens['label_categ_0_num'].values
confusion_GB_tfidf = udf.train_model_GB(tf_idf_matrix, seed=100)
confusion_GB_counts = udf.train_model_GB(counts_matrix, seed=100)

#### MAKING COMPOSITE MODEL
text_data = pd.read_csv(PATH_DATA + TEXT_DATA_FILE_NAME, index_col=[0])
text_data.drop(columns=['label'], inplace=True)

# TFIDF
image_data = tf_idf_matrix.copy()
image_data = image_data.merge(original_data.image, left_index=True, right_on='image')
image_data.drop(columns=['label', 'image'], inplace=True)
image_data = pd.DataFrame(StandardScaler().fit_transform(image_data))

composite_data_tfidf = text_data.merge(image_data, left_index=True, right_index=True)

composite_data_tfidf['label_Kmeans_tfidf'] = KMeans(n_clusters=7).fit_predict(StandardScaler().fit_transform(composite_data_tfidf))
print(composite_data_tfidf['label_Kmeans_tfidf'].value_counts())

composite_data_tfidf['label'] = original_data.label_categ_0
confusion_GB_compo_tfidf = udf.train_model_GB(composite_data_tfidf)


# COUNTS
image_data = counts_matrix.copy()
image_data = image_data.merge(original_data.image, left_index=True, right_on='image')
image_data.drop(columns=['label', 'image'], inplace=True)
image_data = pd.DataFrame(StandardScaler().fit_transform(image_data))

composite_data_counts = text_data.merge(image_data, left_index=True, right_index=True)

composite_data_counts['label_Kmeans_counts'] = KMeans(n_clusters=7).fit_predict(StandardScaler().fit_transform(composite_data_counts))
print(composite_data_counts['label_Kmeans_counts'].value_counts())

composite_data_counts['label'] = original_data.label_categ_0
confusion_GB_compo_counts = udf.train_model_GB(composite_data_counts)

# Performance scores
ARI_tfidf = adjusted_rand_score(composite_data_tfidf['label'], composite_data_tfidf['label_Kmeans_tfidf'])
AMI_tfidf = adjusted_mutual_info_score(composite_data_tfidf['label'], composite_data_tfidf['label_Kmeans_tfidf'])

ARI_counts = adjusted_rand_score(composite_data_counts['label'], composite_data_counts['label_Kmeans_counts'])
AMI_counts = adjusted_mutual_info_score(composite_data_counts['label'], composite_data_counts['label_Kmeans_counts'])

print("COMPOSITE : ARI and AMI scores with tfidf: ", ARI_tfidf, AMI_tfidf)
print("COMPOSITE : ARI and AMI scores with counts: ", ARI_counts, AMI_counts)