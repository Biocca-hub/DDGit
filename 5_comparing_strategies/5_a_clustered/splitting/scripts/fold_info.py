import pandas as pd

files = [f'../output/fold_{i}.tsv' for i in range(1,6)]
dfs = [pd.read_csv(f, sep='\t') for f in files]

df = pd.concat(dfs, ignore_index=True)

srvs = [] 
pdbs = []
ups = []
ccs = []
for d in dfs:
    srvs.append(d.shape[0])
    pdbs.append(len(d.PDB_ID.unique()))
    ups.append(len(d.Uniprot_ID.unique()))
    ccs.append(','.join([str(i) for i in list(d.CC_ID.unique())]))

infos = pd.DataFrame({'fold': [i for i in range(1,6)],
                      'SRVs': srvs,
                      'complexes': pdbs,
                      'proteins': ups, 
                      'CCs': ccs})
infos.to_csv('../output/folds_infos.tsv', sep = '\t', index = False)