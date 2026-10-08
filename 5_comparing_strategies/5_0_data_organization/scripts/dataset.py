import pandas as pd
import networkx as nx

final = pd.read_csv('../input/final_data.tsv', sep = '\t')
del final['Unnamed: 0']

ddgit_clustering = pd.read_csv('../input/CC_info.tsv', sep = '\t')
del ddgit_clustering['Unnamed: 0']

unique = final['Unique']
compl = final['Complex']
pdb = final['PDB_ID']
chain = final['PDB_ID'].astype(str) + '_' + final['Clean_mut'].str[1]
chain_id = final['Clean_mut'].str[1]
wt = final['Clean_mut'].str[0]
mut = final['Clean_mut'].str[-1]
position = final['Clean_mut'].str[2:-1]
ddg = final['DDG_avg']
location = final['Location']

ddgit_cc = []
for c in chain:
    for i in range(ddgit_clustering.shape[0]):
        if c in ddgit_clustering['Nodes'][i].split(','):
            ddgit_cc.append(ddgit_clustering['CC_ID'][i])

graph = pd.read_csv('../5_b_LOBSO/clustering/output/out_proteins_graph.tsv', sep = '\t')

G = nx.from_pandas_edgelist(graph,
                            source='complex_a',
                            target='complex_b')

components = sorted(nx.connected_components(G),key=len,reverse=True)

component_id = {}

for i, component in enumerate(components, start=1):
    for node in component:
        component_id[node] = f'{i}'

lobso_cc = final['Complex'].map(component_id).to_list()

folds = {'1': [], '2': [], '3': [], '4': [], '5': []}
for i in range(5):
    df = pd.read_csv(f'../5_a_clustered/splitting/output/fold_{i+1}.tsv', sep = '\t')
    folds[str(i+1)].extend(list(set(df['CC_ID'].to_list())))

cc_to_fold = {cc: int(fold_id)
              for fold_id, ccs in folds.items()
              for cc in ccs}

ddgit_fold = [cc_to_fold[cc] for cc in ddgit_cc]

pd.DataFrame({'unique': unique,
              'complex': compl,
              'pdb': pdb,
              'chain': chain,
              'chain_id': chain_id, 
              'wt': wt,
              'mut': mut,
              'position': position,
              'location': location,
              'ddg': ddg,
              'ddgit_cc': ddgit_cc,
              'ddgit_fold': ddgit_fold,
              'lobso_cc': lobso_cc
              }).to_csv('../output/ddgit_dataset.tsv', sep = '\t')