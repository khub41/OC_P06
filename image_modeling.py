import image_clean
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

PATH_IMAGES = 'data/Flipkart/Images'


def collect_all_features():
    original_dir = os.getcwd()
    os.chdir(PATH_IMAGES)

    full_features = pd.DataFrame()
    for file_name in os.listdir():
        descriptors = image_clean.get_descriptors(file_name)

        file_df = pd.DataFrame(descriptors)
        try:
            file_df['image_id'] = pd.Series([file_name] * len(descriptors))
            full_features = full_features.append(file_df)
        except TypeError as e:
            print(e)
            print(file_name)

    return full_features


data_img = collect_all_features()
scaler = StandardScaler()
data_img_scale = scaler.fit_transform(data_img.drop(columns=['image_id']))

def tuning_kmeans(data):
    pass