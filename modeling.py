import pandas as pd
import udf
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


PATH_DATA = 'data/'
DATA_FILE_NAME = 'decomp_20210910_700comp.csv'

data = pd.read_csv(PATH_DATA + DATA_FILE_NAME, index_col=[0])

new_data = udf.train_model_kmeans(data.drop(columns=['label']), 7)
print(new_data.predicted_label.value_counts())

AMI = adjusted_mutual_info_score(data.label, new_data.predicted_label)
ARI = adjusted_rand_score(data.label, new_data.predicted_label)

print(AMI, ARI)

confusion_GB = udf.train_model_GB(data)
