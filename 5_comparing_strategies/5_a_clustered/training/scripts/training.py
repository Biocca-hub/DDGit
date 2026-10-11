import pandas as pd 
import numpy as np
import torch 
from baseline import MLP

df = pd.read_csv('../input/ddgit_dataset.tsv', sep = '\t')
X = torch.load('../input/features.pt')
Y = torch.load('../input/targets.pt')
V = np.array(torch.load('../input/labels.pt'))

#print(type(X))
#print(type(Y))
#print(type(V))

pat = 100
warm = 500
max_ep = 5000
activations = ['relu', 'sigmoid']
dpouts = [0.1, 0.3, 0.5, 0.7] 
losses = ["L1", "MSE"]
architectures = [[16], [128], [32,16], [128,32], [32,16,16], [128,64,32]]

all_preds = []
all_targets = []
all_labels = []

fold_results = []

for i in range(1,6):
    df_small = df[df.ddgit_fold == i]

    indices = torch.tensor(df_small.index.to_list(), dtype=torch.long)

    # maschera booleana lunga quanto il numero di righe di X, inizialmente sarà tutta True
    mask = torch.ones(X.shape[0], dtype=torch.bool)
    # False nelle posizioni che appartengono al test set
    mask[indices] = False

    X_training = X[mask]
    X_test = X[indices]

    Y_training = Y[mask]
    Y_test = Y[indices]
    all_targets.extend(list(Y_test))

    V_training = V[mask.numpy()]
    V_test = V[indices.numpy()]
    all_labels.extend(list(V_test))

    for h in architectures:
        for a in activations:
            for l in losses:
                for d in dpouts:

