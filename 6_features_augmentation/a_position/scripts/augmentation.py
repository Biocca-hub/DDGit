import pandas as pd
import torch

features = pd.read_csv('../output/normalized_aa_features.tsv', sep='\t')

n_features = ['n_wt_mw', 'n_mut_mw', #'n_delta_mw',
              'n_wt_v', 'n_mut_v', #'n_delta_v',
              'n_wt_h', 'n_mut_h', #'n_delta_hidropathy',
              'n_wt_i', 'n_mut_i', #'n_delta_iso_p',
              'wt_a', 'mut_a'] #'delta_aromaticity']

X = torch.load('../input/features.pt')
label = torch.load('../input/labels.pt')

# controllo: stesso numero di righe
assert len(features) == X.shape[0], f"{len(features)} righe nel tsv vs {X.shape[0]} in X"

# DataFrame -> tensore (n_esempi, 15), stesso dtype di X
new = torch.tensor(features[n_features].to_numpy(), dtype=X.dtype)

# aggiungi le colonne a destra: (n, 62) + (n, 15) -> (n, 77)
X_aug = torch.cat([X, new], dim=1)

print(X_aug.shape)   # atteso: (n, 77)
torch.save(X_aug, '../output/wt_mut_features.pt')