import pandas as pd 

df = pd.read_csv('../input/ddgit_dataset.tsv', sep = '\t')

print(len(df.loco_cc.unique()))
counts = df['pdb'].value_counts()
cluster_map = {complex_: cluster_id for cluster_id, complex_ in enumerate(counts.index)}
df['loco_cc'] = df['pdb'].map(cluster_map)
print(len(df.loco_cc.unique()))

df.to_csv('../input/ddgit_dataset.tsv', sep = '\t')