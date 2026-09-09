import pandas as pd

import pandas as pd
import numpy as np
import torch
import math

def make_df(file, delimiter):
    """Takes file path and returns a pandas DataFrame."""
    try:
        df = pd.read_csv(file, sep=delimiter)
        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']
        return df
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

s4169 = make_df('../../Binding_Affinity/data/datasets/s4169.csv', '\t')
ddg = s4169['actual'].to_list()

# folds 
bts_df = make_df('../data_meta/bts/BTS_data.tsv', '\t')
fold_1_df = make_df('../data_meta/training/fold_1.tsv', '\t')
fold_2_df = make_df('../data_meta/training/fold_2.tsv', '\t')
fold_3_df = make_df('../data_meta/training/fold_3.tsv', '\t')
fold_4_df = make_df('../data_meta/training/fold_4.tsv', '\t')
fold_5_df = make_df('../data_meta/training/fold_5.tsv', '\t')

folds = [bts_df, fold_1_df, fold_2_df, fold_3_df, fold_4_df, fold_5_df]
tensors = ['bts', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']

ddg_1 = []
for i in range(1,6): 
    print(folds[i].columns)
    print(folds[i]['DDG'].head())

