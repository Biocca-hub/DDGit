import pandas as pd 
import numpy as np
import torch 

df = pd.read_csv('../input/ddgit_dataset.tsv', sep = '\t')
#print(df.lobso_cc.max())
#print(df.lobso_cc.min())

lobso_cc = [i for i in range(1,193)]
#print(lobso_cc)
X = torch.load('../input/features.pt')
#print(X.size())

for i in lobso_cc:
    df_small = df[df.lobso_cc == i]

    indices = torch.tensor(df_small.index.to_list(), dtype=torch.long)

    # maschera booleana lunga quanto il numero di righe di X, inizialmente sarà tutta True
    mask = torch.ones(X.shape[0], dtype=torch.bool)
    # False nelle posizioni che appartengono al test set
    mask[indices] = False

    X_training = X[mask]
    X_test = X[indices]
    #print(X_training.size()[0])
    #print(X_test.size()[0])