import pandas as pd 
import numpy as np

folds = [f"../output/training/fold_{i}.tsv" for i in range(1,6)]
ala_folds = [f"../output/training/alanine/to_alanine_{i}.tsv" for i in range(1,6)]

fold = []
destabilizing = []
stabilizing = []
ala_destabilizing = []
ala_stabilizing = []

for i in range(5):
    df = pd.read_csv(folds[i], sep = '\t')
    destabilizing.append(df[df.DDG_avg < 0].shape[0])
    stabilizing.append(df[df.DDG_avg > 0].shape[0])
    df = pd.read_csv(ala_folds[i], sep = '\t')
    ala_destabilizing.append(df[df.DDG_avg < 0].shape[0])
    ala_stabilizing.append(df[df.DDG_avg > 0].shape[0])
    fold.append(i+1)

pd.DataFrame({'Fold': fold,
              'Destabilizing': destabilizing,
              'Stabilizing': stabilizing,
              'Destabilizing_A': ala_destabilizing,
              'Stabilizing_A': ala_stabilizing}).to_csv('../output/training/ddg_info/ddg_info.tsv', sep = '\t')


