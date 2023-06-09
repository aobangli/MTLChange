import joblib
import pandas as pd
import numpy as np

from Source.ChangePrediction.TrainConfig import data_path, feature_label_list

df = pd.read_csv(data_path)

# nan_rows = df[df.isna().any(axis=1)]

col_list = feature_label_list

# 处理nan值
nan_rows = df[df[col_list].isna().any(axis=1)]

rows_to_fill = nan_rows.index
df.loc[rows_to_fill] = df.loc[rows_to_fill].fillna(0)

df.to_csv(data_path, index=False)

print(df.shape)
