import pandas as pd 
import numpy as np 
from Bio.SeqUtils.ProtParam import ProteinAnalysis
# from quantiprot.metrics.aaindex import get_aa2volume

# aa2vol = get_aa2volume()

# Punto isoelettrico (pH) — valori standard per l'amminoacido libero (Lehninger)
aa_pI = {
    "A": 6.00, "R": 10.76, "N": 5.41, "D": 2.77, "C": 5.07,
    "E": 3.22, "Q": 5.65,  "G": 5.97, "H": 7.59, "I": 6.02,
    "L": 5.98, "K": 9.74,  "M": 5.74, "F": 5.48, "P": 6.30,
    "S": 5.68, "T": 5.60,  "W": 5.89, "Y": 5.66, "V": 5.96,
}

# Volume del residuo (Å³) — Zamyatnin 1972 (AAindex ZAMY720101)
aa_volume = {
    "A": 88.6,  "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "E": 138.4, "Q": 143.8, "G": 60.1,  "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0,  "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

df = pd.read_csv('../output/ddgit_dataset.tsv', sep = '\t')

# print(df.columns)
'''
['unique', 'complex', 'pdb', 'chain', 'chain_id', 'wt', 'mut',
'position', 'location', 'ddg', 'ddgit_cc', 'ddgit_fold', 'lobso_cc',
'loco_cc', 'stratified_cc', 'inner_1', 'inner_2', 'inner_3', 'inner_4',
'inner_5', 'inner_6', 'inner_7', 'inner_8', 'inner_9', 'inner_10']
'''

def phy_chem_surroundings(df, pdb_folder, radius):
    
    aa_3_1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 
            'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

    c_mw_avg = []
    c_vol_avg = []
    c_aro_avg = []
    c_iso_avg = []
    c_hidro_avg = []

    m_mw_avg = []
    m_vol_avg = []
    m_aro_avg = []
    m_iso_avg = []
    m_hidro_avg = []

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

            c_mw = []
            c_vol = []
            c_aro = []
            c_iso = []
            c_hidro = []

            # Monomer

            m_mw = []
            m_vol = []
            m_aro = []
            m_iso = []
            m_hidro = []

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
                        c_mw.append(ProteinAnalysis(aa_3_1.get(res)).molecular_weight())
                        # Volumes 
                        c_vol.append(aa_volume[aa_3_1.get(res)])
                        # Aromaticity 
                        c_aro.append(ProteinAnalysis(aa_3_1.get(res)).aromaticity())
                        # Isoelectric point  
                        c_iso.append(aa_pI[aa_3_1.get(res)])
                        # Hidropathy
                        c_hidro.append(ProteinAnalysis(aa_3_1.get(res)).gravy())

                        if line[21] == c_id: # monomer
                                
                            # MWs 
                            m_mw.append(ProteinAnalysis(aa_3_1.get(res)).molecular_weight())
                            # Volumes 
                            m_vol.append(aa_volume[aa_3_1.get(res)])
                            # Aromaticity 
                            m_aro.append(ProteinAnalysis(aa_3_1.get(res)).aromaticity())
                            # Isoelectric point  
                            m_iso.append(aa_pI[aa_3_1.get(res)])
                            # Hidropathy
                            m_hidro.append(ProteinAnalysis(aa_3_1.get(res)).gravy())


        c_mw_avg.append(sum(c_mw)/len(c_mw))
        c_vol_avg.append(sum(c_vol)/len(c_vol))
        c_aro_avg.append(sum(c_aro)/len(c_aro))
        c_iso_avg.append(sum(c_iso)/len(c_iso))
        c_hidro_avg.append(sum(c_hidro)/len(c_hidro))   

        m_mw_avg.append(sum(m_mw)/len(m_mw))
        m_vol_avg.append(sum(m_vol)/len(m_vol))
        m_aro_avg.append(sum(m_aro)/len(m_aro))
        m_iso_avg.append(sum(m_iso)/len(m_iso))
        m_hidro_avg.append(sum(m_hidro)/len(m_hidro))  
        # if len(hits) > 1:
            # print(df.unique[i], len(hits))
            # for h in hits:
                # print('   ', h)
# print(counter)

    df_surr = pd.DataFrame({'unique': df.unique,
                'complex_mw': c_mw_avg,
                'complex_volume': c_vol_avg,
                'complex_aromaticity': c_aro_avg,
                'complex_isoelectric_point': c_iso_avg,
                'complex_hidropathy': c_hidro_avg,
                'monomer_mw': m_mw_avg,
                'monomer_volume': m_vol_avg,
                'monomer_aromaticity': m_aro_avg,
                'monomer_isoelectric_point': m_iso_avg,
                'monomer_hidropathy': m_hidro_avg,
                })

    return df_surr 

df_12 = phy_chem_surroundings(df, '../input/PDBs/', 12)

df_12.to_csv('../output/up_surr_12.tsv', sep = '\t', index=False)

df_6 = phy_chem_surroundings(df, '../input/PDBs/', 6)

df_6.to_csv('../output/up_surr_6.tsv', sep = '\t', index=False)

df_8 = phy_chem_surroundings(df, '../input/PDBs/', 8)

df_8.to_csv('../output/up_surr_8.tsv', sep = '\t', index=False)

df_10 = phy_chem_surroundings(df, '../input/PDBs/', 10)

df_10.to_csv('../output/up_surr_10.tsv', sep = '\t', index=False)

df_4 = phy_chem_surroundings(df, '../input/PDBs/', 4)

df_4.to_csv('../output/up_surr_4.tsv', sep = '\t', index=False)

