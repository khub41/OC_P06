import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
path_data = 'data/'
name_data_file = 'decomp_2308_700comp.csv'
data = pd.read_csv(path_data + name_data_file, index_col=[0])

def train_model_kmeans(data_train, n_clusters):
    data_train_copy = data_train.copy()
    model = KMeans(n_clusters=n_clusters, init="k-means++")
    predicted_labels = model.fit_predict(data_train_copy.drop(columns=['label']))
    data_train_copy['predicted_label'] = predicted_labels

    return data_train_copy

def train_model_GB(data_train):
    data_train_copy = data_train.copy()

    X_train, X_test, y_train, y_test = \
        train_test_split(data_train.drop(columns=['label']),
                     data_train.label,
                     test_size=0.3,
                     random_state=41)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # score = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(report)
    X_test['label_pred'] = y_pred
    return X_test


new_data = train_model_kmeans(data, 7)

confusion_GB = train_model_GB(data)