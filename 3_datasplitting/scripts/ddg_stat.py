import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

aa_list = list("GAVPLIMFWYSTCNQHDEKR")

ala = [f'../output/training/to_alanine_{i}.tsv' for i in range(1,6)]
folds = [f'../output/training/fold_{i}.tsv' for i in range(1,6)]

fold = []
total = []
alanine = []
ratio = []
avg_DDG = []
std_DDG = []
max_DDG = []
min_DDG = []
ala_avg_DDG = []
ala_std_DDG = []
ala_max_DDG = []
ala_min_DDG = []

for i in range(5):
    df = pd.read_csv(ala[i], sep = '\t')
    f = pd.read_csv(folds[i], sep = '\t')
    fold.append(i+1)
    total.append(f.shape[0])
    alanine.append(df.shape[0])
    ratio.append(df.shape[0]/f.shape[0])
    avg_DDG.append(f['DDG_avg'].mean())
    std_DDG.append(f['DDG_avg'].std())
    max_DDG.append(f['DDG_avg'].max())
    min_DDG.append(f['DDG_avg'].min())
    ala_avg_DDG.append(df['DDG_avg'].mean())
    ala_std_DDG.append(df['DDG_avg'].std())
    ala_max_DDG.append(df['DDG_avg'].max())
    ala_min_DDG.append(df['DDG_avg'].min())

bias_stats = pd.DataFrame({'Fold': fold,
                           'SRVs': total,
                           'DDG_mean': avg_DDG,
                           'std': std_DDG,
                           'max': max_DDG,
                           'min': min_DDG,
                           'SRVs_to_alanine': alanine,
                           'DDG_mean_alanine': ala_avg_DDG,
                           'std_alanine': ala_std_DDG,
                           'max_alanine': ala_max_DDG,
                           'min_alanine': ala_min_DDG,
                           'Ratio_SRVs_to_alanine': ratio
                           }) 

bias_stats.to_csv('../output/training/bias.tsv', sep = '\t')
