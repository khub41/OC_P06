import mlflow

import image_clean
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import os

import udf

PATH_IMAGES = 'data/Flipkart/Images'

# data_img = udf.collect_all_features(PATH_IMAGES)
# data_img.to_csv("data/descriptors_all_new_preprocess_1709.csv")
data_img = pd.read_csv("data/descriptors_all_new_preprocess_1709.csv", index_col=[0])

data_img_scale = udf.scale_data(data_img.drop(columns=['image_id']))
# udf.scree_plot(data_img_scale, 128)
data_img_scale = udf.reduce_dim_pca(data_img_scale, 80)

# mlflow.create_experiment("k_means_descriptors")
# slh_scores, db_scores = udf.tuning_kmeans(data_img_scale,
#                                           range(100,200, 10),
#                                           "k_means_descriptors",
#                                           show=True,
#                                           savefig="slh_db_range_100_200_10_date_1709")

new_data = udf.train_model_kmeans(data_img_scale, 30)
print(new_data.predicted_label.value_counts())
