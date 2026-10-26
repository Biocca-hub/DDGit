import torch 

X = torch.load('../input/features.pt')
print(X.max())
print(X[X.argmax().item()//X.shape[1]][X.argmax().item()%X.shape[1]]) # l’indice k corrisponde a riga k // 62, colonna k % 62.
print(X.min())

groups = {
    'blocco_0_19':  slice(0, 20),
    'blocco_20_39': slice(20, 40),
    'blocco_40_59': slice(40, 60),
    'col_60':       slice(60, 61),
    'col_61':       slice(61, 62),
}

for name, cols in groups.items():
    b = X[:, cols]
    print(f'{name:14s} min {b.min().item():10.4f}  max {b.max().item():10.4f}')

print(torch.quantile(X[:, 60], torch.tensor([0.5, 0.95, 0.99])))
print(torch.quantile(X[:, 61], torch.tensor([0.5, 0.95, 0.99])))

soglia = torch.quantile(X[:, 61], 0.99)
idx = torch.where(X[:, 61] > soglia)[0]
print(idx, X[idx, 61])