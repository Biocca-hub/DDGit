import torch
import pandas as pd 
import numpy as np

folders = [
           '../input/tensors_old_split/features/',
           '../input/tensors_old_split/labels/',
           '../input/tensors_old_split/targets/'
           ]

original_folds = ['bts_', 'fold_1_', 'fold_2_', 'fold_3_', 'fold_4_', 'fold_5_']

types = ['X_tensor.pt',
         'SRVs_tensor.pt',
         'Y_tensor.pt'
        ]

features = []
for f in original_folds:
    features.append(torch.load(f'{folders[0]}{f}{types[0]}'))

targets = []
for f in original_folds:
    targets.append(torch.load(folders[2]+f+types[2]))

labels = []
for f in original_folds:
    labels.append(torch.load(folders[1]+f+types[1]))

f = torch.cat(features, dim = 0)
t = torch.cat(targets, dim = 0)
l = []
for lab in labels:
    l.extend(lab)

features_col = list(f.numpy())
targets_col = list(t.numpy())

df1 = pd.read_csv('../output/ddgit_dataset.tsv', sep = '\t')
#del df1['Unnamed: 0']
counts = df1['complex'].value_counts()

cluster_map = {
                complex_: cluster_id
                for cluster_id, complex_ in enumerate(counts.index)
              }

df1['loco_cc'] = df1['complex'].map(cluster_map)

df2 = pd.DataFrame({'unique': l,
                    'targets': targets_col,
                    'features': features_col})

df1 = df1.set_index('unique').reindex(df2['unique']).reset_index()
#df1.to_csv('../output/ddgit_dataset.tsv', sep = '\t', index = False)
df1['features']=df2['features']

#df1.to_csv('../output/ddgit_input.tsv', sep = '\t', index = False)


torch.save(f, '../output/features.pt')
torch.save(t, '../output/targets.pt')
torch.save(l, '../output/labels.pt')