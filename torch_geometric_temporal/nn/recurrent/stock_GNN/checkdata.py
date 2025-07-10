import pickle
import pandas as pd
import gzip
import os


data_dir = "/home/xu/clean_data"

def load_pickle(filename):
    file_path = os.path.join(data_dir, filename)
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

data = load_pickle("filtered_open.pkl")
print(data.head())
print(data.describe())
data = data.ffill().bfill()
mean = data.values.astype(float).mean()
std =  data.values.astype(float).std()
print(mean)
print(std)
df_zscore = (data - mean) / std
print(df_zscore.head())
print(df_zscore.describe())
#data = data.ffill().bfill()
#print(data.isna().sum().sum())
print(df_zscore.values.mean())
print(df_zscore.values.std())
