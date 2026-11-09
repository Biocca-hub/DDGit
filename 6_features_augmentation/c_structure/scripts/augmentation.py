import pandas as pd
import torch

X = torch.load('../input/features.pt')
label = torch.load('../input/labels.pt')

radii = [4, 6, 8, 10, 12]
dfs = {r: pd.read_csv(f'../output/norm_surr_{r}.tsv', sep='\t') for r in radii}

feat_cols = [
    'complex_aromaticity', 'monomer_aromaticity',
    'norm_c_mw', 'norm_m_mw',
    'norm_c_vol', 'norm_m_vol',
    'norm_c_pI', 'norm_m_pI',
    'norm_c_hidropathy', 'norm_m_hidropathy',
]

for r, df in dfs.items():
    assert df.shape[0] == X.shape[0], f"raggio {r}: {df.shape[0]} righe nel tsv vs {X.shape[0]} in X"
    missing = set(feat_cols) - set(df.columns)
    assert not missing, f"raggio {r}: colonne mancanti {missing}"
    assert not df[feat_cols].isna().any().any(), f"raggio {r}: ci sono NaN"

# per ogni feature, i 5 raggi uno accanto all'altro:
# complex_aromaticity_r4, _r6, _r8, _r10, _r12, monomer_aromaticity_r4, ...
blocks = []
names = []
for col in feat_cols:
    for r in radii:
        blocks.append(dfs[r][col])
        names.append(f'{col}_r{r}')

features = pd.concat(blocks, axis=1)
features.columns = names                      # (n, 50)

new = torch.tensor(features.to_numpy(), dtype=X.dtype)
X_aug = torch.cat([X, new], dim=1)            # (n, 62) + (n, 50) -> (n, 112)

assert X_aug.shape == (X.shape[0], 112), X_aug.shape
print(X_aug.shape)

torch.save(X_aug, '../output/aug_features.pt')
# utile per ritrovare a cosa corrisponde ogni colonna (indici 62..111)
pd.Series(names).to_csv('../output/surr_feature_names.tsv', sep='\t', index=False, header=False)