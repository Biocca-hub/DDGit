import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

paths = [f'../output/training/fold_{i}.tsv' for i in range(1,6)]

for p in paths:
    df = pd.read_csv(p, sep = '\t')

    df['mut'] = df['Clean_mut'].str[-1]
    df['wt'] = df['Clean_mut'].str[0]
    df_a = df[df.mut == 'A'].reset_index(drop=True)
    df_a = df_a[['wt','mut','Unique', 'Unique_position', 'Clean_mut', 'DDG_avg', 'Uniprot_ID', 'Location', 'CC_ID']]
    df_a.to_csv(f'../output/training/to_alanine_{p[24]}.tsv', sep = '\t')
    #print(df.shape[0])
    #print()
    #print(df_a.shape[0])
    #print()
    #print('-'*30)
    #print(df_a.head())
