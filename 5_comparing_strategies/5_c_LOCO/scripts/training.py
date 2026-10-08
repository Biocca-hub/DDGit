import pandas as pd 
import numpy as np
import torch 

df = pd.read_csv('../input/ddgit_dataset.tsv', sep = '\t')
loco_cc = [i for i in range(320)]
X = torch.load('../input/features.pt')
print(X.size())

for i in loco_cc:
    df_small = df[df.loco_cc == i]

    indices = torch.tensor(df_small.index.to_list(), dtype=torch.long)

    # maschera booleana lunga quanto il numero di righe di X, inizialmente sarà tutta True
    mask = torch.ones(X.shape[0], dtype=torch.bool)
    # False nelle posizioni che appartengono al test set
    mask[indices] = False

    X_training = X[mask]
    X_test = X[indices]
