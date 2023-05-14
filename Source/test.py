import joblib

from Source.DataProcess.DataProcessConfig import change_list_filepath

change_list_df = joblib.load(change_list_filepath)

print(change_list_df.shape)