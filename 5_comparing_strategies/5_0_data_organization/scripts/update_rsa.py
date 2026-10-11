import torch 
import pandas as pd

X = torch.load('../output/wrong_rsa.pt')
X1 = X.clone()
if torch.equal(X, X1):
    print('X == X1')
rsa = pd.read_csv('../input/rsa_out/all_rsa.tsv', sep = '\t')

for i in range(X.size()[0]):
    X1[i][-2] = rsa['rsa_complex'][i]
    X1[i][-1] = rsa['rsa_monomer'][i]

if not torch.equal(X, X1):
    print('X != X1')

torch.save(X1, '../output/features.pt')
