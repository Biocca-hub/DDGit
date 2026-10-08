import pandas as pd 
import numpy as np
import torch 

df = pd.read_csv('../input/ddgit_dataset.tsv', sep = '\t')
X = torch.load('../input/features.pt')
Y = torch.load('../input/targets.pt')
V = np.array(torch.load('../input/labels.pt'))

#print(type(X))
#print(type(Y))
#print(type(V))

for i in range(1,6):
    df_small = df[df.ddgit_fold == i]

    indices = torch.tensor(df_small.index.to_list(), dtype=torch.long)

    # maschera booleana lunga quanto il numero di righe di X, inizialmente sarà tutta True
    mask = torch.ones(X.shape[0], dtype=torch.bool)
    # False nelle posizioni che appartengono al test set
    mask[indices] = False

    X_training = X[mask]
    X_test = X[indices]
    #print(X_training.size()[0])
    #print(X_test.size()[0])

    Y_training = Y[mask]
    Y_test = Y[indices]

    V_training = V[mask.numpy()]
    V_test = V[indices.numpy()]

    #print(X_training.size())
    #print(Y_training.size())
    #print(V_training.shape) 
    #print(X_test.size())
    #print(Y_test.size())
    #print(V_test.shape) 

'''
    I 4 fold di training vanno suddivisi come segue per una CV interna:
        • 3 training, di cui una FRAZIONE per la validazione
        • 1 test
       Per ognuna delle 5 run, ottieni una combinazione (che non è rilevante
       allo scopo attuale) sulla quale eseguirai training sulle 4 e testerai
       sul fold tenuto fuori dalla validazione interna. Di questo, 
        1. salva:
            • target
            • predizioni
        2. calcola:
            • PCC
            • MSE
        3. concatena tutti i target e le predizioni per calcolare nuovamente:
            • PCC
            • MSE
           Queste saranno le metriche relative alla performance che ci
           interessano, in quanto sulle metriche singole dei 5 fold, la media 
           tenderebbe ad essere dominata dai valori estremi.
'''