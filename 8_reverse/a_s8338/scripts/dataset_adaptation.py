import pandas as pd

df = pd.read_csv('../input/dataset_to_fix.tsv', sep ='\t')
del df['Unnamed: 0']

colonne = ['unique', 'binding_site', 'pdb', 'chain', 'chain_id', 'wt', 'mut',
           'pdb_pos', 'location', 'ddg', 'ddgit_cc', 'ddgit_fold', 'lobso_cc',
           'loco_cc', 'stratified_cc', 'inner_1', 'inner_2', 'inner_3', 'inner_4',
           'inner_5', 'inner_6', 'inner_7', 'inner_8', 'inner_9', 'inner_10',
           'uniprot', 'unp_pos', 'unp_wt']

forward = pd.DataFrame({'unique': df.unique,
                          'wt': df.wt,
                          'mut': df.mut,
                          'pdb': df.pdb,
                          'chain_id': df.chain_id,
                          'pdb_pos': df.pdb_pos,
                          'uniprot': df.uniprot,
                          'uniprot_pos': df.unp_pos,
                          'unp_wt': df.unp_wt,
                          'location': df.location,
                          'ddg': df.ddg})

forward_splits = pd.DataFrame({'unique': df.unique,
                           'ddgit_cc': df.ddgit_cc,
                           'ddgit_fold': df.ddgit_fold,
                           'lobso_fold': df.lobso_cc,
                           'loco_fold': df.loco_cc,
                           'stratified_cc': df.stratified_cc,
                           'inner_1': df.inner_1, 
                           'inner_2': df.inner_2, 
                           'inner_3': df.inner_3, 
                           'inner_4': df.inner_4,
                           'inner_5': df.inner_5, 
                           'inner_6': df.inner_6, 
                           'inner_7': df.inner_7, 
                           'inner_8': df.inner_8, 
                           'inner_9': df.inner_9, 
                           'inner_10': df.inner_10
                           })

ddg = df.ddg * -1 

mut = df.wt 
wt = df.mut

fw_unique = list(df.unique)
rev_unique = []
for u in fw_unique:
    chain_pos = u.split('_')[-1][1:-1]
    res_mut = u.split('_')[-1][0]
    res_wt = u.split('_')[-1][-1]
    rev_unique.append(u.split('_')[0]+'_'+u.split('_')[1]+'_'+res_wt+chain_pos+res_mut)

reverse = pd.DataFrame({'unique': rev_unique,
                          'wt': wt,
                          'mut': mut,
                          'pdb': df.pdb,
                          'chain_id': df.chain_id,
                          'pdb_pos': df.pdb_pos,
                          'uniprot': df.uniprot,
                          'uniprot_pos': df.unp_pos,
                          'unp_mut': df.unp_wt,
                          'location': df.location,
                          'ddg': ddg})

reverse_splits = pd.DataFrame({'unique': rev_unique,
                           'ddgit_cc': df.ddgit_cc,
                           'ddgit_fold': df.ddgit_fold,
                           'lobso_fold': df.lobso_cc,
                           'loco_fold': df.loco_cc,
                           'stratified_cc': df.stratified_cc,
                           'inner_1': df.inner_1, 
                           'inner_2': df.inner_2, 
                           'inner_3': df.inner_3, 
                           'inner_4': df.inner_4,
                           'inner_5': df.inner_5, 
                           'inner_6': df.inner_6, 
                           'inner_7': df.inner_7, 
                           'inner_8': df.inner_8, 
                           'inner_9': df.inner_9, 
                           'inner_10': df.inner_10
                           })

forward_splits.to_csv('../output/fw_splits.tsv', sep = '\t', index = False)
reverse_splits.to_csv('../output/rev_splits.tsv', sep = '\t', index = False)
forward.to_csv('../output/forward.tsv', sep = '\t', index = False)
reverse.to_csv('../output/reverse.tsv', sep = '\t', index = False)