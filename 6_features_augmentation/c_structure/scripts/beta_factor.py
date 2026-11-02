import pandas as pd 
import numpy as np 

df = pd.read_csv('../output/ddgit_dataset.tsv', sep = '\t')

# print(df.columns)
'''
['unique', 'complex', 'pdb', 'chain', 'chain_id', 'wt', 'mut',
'position', 'location', 'ddg', 'ddgit_cc', 'ddgit_fold', 'lobso_cc',
'loco_cc', 'stratified_cc', 'inner_1', 'inner_2', 'inner_3', 'inner_4',
'inner_5', 'inner_6', 'inner_7', 'inner_8', 'inner_9', 'inner_10']
'''

def B_factor(df, pdb_folder, radius):
    
    aa_3_1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 
            'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

    b_factor = []

    c_b_avg = []

    m_b_avg = []

    b_avg = []

    for i in range(df.shape[0]):
        # if df.unique[i].split('_')[0] == df.pdb[i] and df.unique[i].split('_')[1] == df.chain_id[i] and df.unique[i].split('_')[2][2:-1] == df.pdb_pos[i]:
        file = f'{pdb_folder}{df.pdb[i]}.pdb'
        c_id = df.chain_id[i]
        pos = str(df.pdb_pos[i]).upper()
        wt = df.wt[i]
        # hits = []
        found = False
        with open(file, 'r') as structure:
            for line in structure:
                if line.startswith('ENDMDL'): # fine del primo modello NMR, tiene questo come buono
                    break
                if not line.startswith('ATOM'):
                    continue
                res_num = line[22:26].strip()
                icode = line[26].strip()
                pdb_res = res_num + (icode if icode else "")
                res = line[17:20]
                if (line[12:16].strip() == 'CA' and line[21] == c_id and pdb_res == pos and aa_3_1.get(res) == wt):
                    
                    b_factor.append(float(line[61:66]))
                    
                    found = True
                    xr = float(line[30:38])
                    yr = float(line[38:46])
                    zr = float(line[46:54])
                    # hits.append(line.rstrip())
                    break # tiene prima posizione come buona
        if not found:
            print(df.unique[i], df.pdb[i], df.chain_id[i], df.pdb_pos[i], df.wt[i])                              # basta il primo CA, ignora altloc B

        with open(file, 'r') as structure:

            # Complex

            c_b = []

            # Monomer

            m_b = []

            # whole 

            b = []

            for line in structure:
                if line.startswith('ENDMDL'): # fine del primo modello NMR, tiene questo come buono
                    break
                if not line.startswith('ATOM'):
                    continue    
                res_num = line[22:26].strip()
                icode = line[26].strip()
                pdb_res = res_num + (icode if icode else "")
                res = line[17:20] 
                if line[12:16].strip() == 'CA' and line[16] in (' ', 'A') and res in aa_3_1: #and pdb_res != pos:
                    xa = float(line[30:38])
                    ya = float(line[38:46])
                    za = float(line[46:54])
                    d = ((xa - xr)**2 + (ya - yr)**2 + (za - zr)**2)**0.5

                    if d <= radius: # complex

                        # MWs 
                        c_b.append(float(line[61:66]))

                        if line[21] == c_id: # monomer
                                
                            # MWs 
                            m_b.append(float(line[61:66]))

                    b.append(float(line[61:66]))

        c_b_avg.append(sum(c_b)/len(c_b))

        m_b_avg.append(sum(m_b)/len(m_b))

        b_avg.append(sum(b)/len(b)) 
 
        # if len(hits) > 1:
            # print(df.unique[i], len(hits))
            # for h in hits:
                # print('   ', h)
# print(counter)

    df_surr = pd.DataFrame({'unique': df.unique,
                'complex_B_factor': c_b_avg,
                'monomer_B_factor': m_b_avg,
                'whole_B_factor': b_avg,
                'position_B_factor': b_factor,
                })

    return df_surr 

df_12 = B_factor(df, '../input/PDBs/', 12)

df_12.to_csv('../output/bf_12.tsv', sep = '\t', index=False)

df_10 = B_factor(df, '../input/PDBs/', 10)

df_10.to_csv('../output/bf_10.tsv', sep = '\t', index=False)

df_8 = B_factor(df, '../input/PDBs/', 8)

df_8.to_csv('../output/bf_8.tsv', sep = '\t', index=False)

df_6 = B_factor(df, '../input/PDBs/', 6)

df_6.to_csv('../output/bf_6.tsv', sep = '\t', index=False)

df_4 = B_factor(df, '../input/PDBs/', 4)

df_4.to_csv('../output/bf_4.tsv', sep = '\t', index=False)