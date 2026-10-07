import pandas as pd
import numpy as np

def make_df(file, delimiter):
    """Takes file path and returns a pandas DataFrame."""
    try:
        df = pd.read_csv(file, sep=delimiter)
        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']
        return df
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

'''
bts_rsa = make_df('../input/rsa_out/BTS_rsa.tsv', '\t')
fold_1_rsa = make_df('../input/rsa_out/old_split/fold_1_rsa.tsv', '\t')
fold_2_rsa = make_df('../input/rsa_out/old_split/fold_2_rsa.tsv', '\t')
fold_3_rsa = make_df('../input/rsa_out/old_split/fold_3_rsa.tsv', '\t')
fold_4_rsa = make_df('../input/rsa_out/old_split/fold_4_rsa.tsv', '\t')
fold_5_rsa = make_df('../input/rsa_out/old_split/fold_5_rsa.tsv', '\t')

df_all = pd.concat([fold_1_rsa, fold_2_rsa, fold_3_rsa, fold_4_rsa, fold_5_rsa, bts_rsa], ignore_index=True)

complexes = []
for el in list(df_all['pdb_file'].str.split('/')):
    complexes.append(el[-1][:-4])
chain = []
for el in list(df_all['monomer_file'].str.split('/')):
    chain.append(el[-1][:-4])

# 4K71_B_HB161A == complesso_catena_wt-chain-pos-mut

# rsa dovrebbe essere la stessa per la stessa posizione, quindi per tutti gli unique id che hanno stesso complesso, stessa catena, stessa posizione, rsa è quella

rsa = pd.DataFrame({'rsa_complex': df_all.rsa_complex,
                    'rsa_monomer': df_all.rsa_monomer,
                    'complex': complexes,
                    'chain': chain,
                    'chain_id': df_all.chain_id,
                    'position': df_all.res_num})
                
rsa.to_csv('../input/rsa_out/all_rsa.tsv', sep = '\t', index = False)
'''

rsa_all = make_df('../input/rsa_out/all_rsa.tsv', '\t')

for i in range(1, 6):
    df = make_df(f'../input/fold_data/fold_{i}.tsv', '\t')

    chains = df['Chain'].to_list()
    pos = df['Clean_mut'].str[2:-1].to_list()

    rsa_complex = []
    rsa_monomer = []
    k = 0
    z = 0
    for j in range(len(chains)):
        # print("Looking for:", repr(chains[j]), repr(pos[j]))
        rsa = rsa_all[(rsa_all['chain'] == chains[j]) & (rsa_all['position'] == pos[j]) & (rsa_all['chain_id'] == chains[j].split('_')[-1])]
        if len(rsa.rsa_complex.unique()) == 1 and len(rsa.rsa_monomer.unique()) ==1:
            k += 1
        
    print(k)
 
    