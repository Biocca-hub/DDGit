import pandas as pd 

df1 = pd.read_csv('../input/final_data.tsv', sep = '\t')
del df1['Unnamed: 0']
df2 = pd.read_csv('../input/ddgit_dataset.tsv', sep = '\t')

'''
['PDB_mut', 'Clean_mut', 'PDB_ID', 'Complex', 'Unique', 'Chain',
       'Uniprot_ID', 'DDG_avg', 'DDG', 'Affinity_WT', 'Affinity_M', 'Location',
       'Experimental_method', 'SKEMPI_version', 'out_type', 'out_protein']

['unique', 'complex', 'pdb', 'chain', 'chain_id', 'wt', 'mut',
       'position', 'location', 'ddg', 'ddgit_cc', 'ddgit_fold', 'lobso_cc',
       'loco_cc', 'stratified_cc', 'inner_1', 'inner_2', 'inner_3', 'inner_4',
       'inner_5', 'inner_6', 'inner_7', 'inner_8', 'inner_9', 'inner_10']
'''

order = df2.unique

df_sorted = df1.set_index('Unique').loc[order].reset_index()

# print(set(df_sorted.Unique == df2.unique))

pdb_position = df_sorted['PDB_mut'].str[2:-1] # lascio stringa per residui di insersione 
#print(pdb_position.head(3), pdb_position.tail(3))
seq_position = df_sorted.Clean_mut.str[2:-1] # lascio stringa per residui di insersione 
#print(seq_position.head(3), seq_position.tail(3))

# 1. sovrascrivi i valori e cambia il nome
df2['position'] = pdb_position
df2 = df2.rename(columns={'position': 'pdb_pos'})

# 2. inserisci accanto e poi elimina quella vecchia
pos = df2.columns.get_loc('pdb_pos')
df2.insert(pos, 'seq_pos', seq_position)

# print((df2.pdb_pos == df2.seq_pos).value_counts())
# False    2713
# True     1456
df2 = df2.rename(columns={'complex': 'binding_site'})

#df2.to_csv('../output/ddgit_dataset.tsv', sep = '\t')

print(df2.location.value_counts())