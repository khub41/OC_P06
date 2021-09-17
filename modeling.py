import pandas as pd
import udf


PATH_DATA = 'data/'
DATA_FILE_NAME = 'decomp_20210910_700comp.csv'

data = pd.read_csv(PATH_DATA + DATA_FILE_NAME, index_col=[0])

# new_data = udf.train_model_kmeans(data, 7)
# print(new_data.predicted_label.value_counts())

confusion_GB = udf.train_model_GB(data)
