import pandas as pd 
import numpy as np 
from Bio.SeqUtils.ProtParam import ProteinAnalysis
#from quantiprot.metrics.aaindex import get_aa2volume

#aa2vol = get_aa2volume()

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


aa_list = list("GAVPLIMFWYSTCNQHDEKR")

mw = [ProteinAnalysis(i).molecular_weight() for i in aa_list] #
arom = [ProteinAnalysis(i).aromaticity() for i in aa_list]
isop = [aa_pI[i] for i in aa_list] #
hidro = [ProteinAnalysis(i).gravy() for i in aa_list] #
vol = [aa_volume[i] for i in aa_list] #

feature_paths = []
norm_paths = []

for i in [4,6,8,10,12]:
    feature_paths.append(f'../output/up_surr_{i}.tsv')
    norm_paths.append(f'../output/norm_surr_{i}.tsv')

def radius_features(df):

    n_c_mw = []
    n_m_mw = []

    n_c_v = []
    n_m_v = []

    n_c_i = []
    n_m_i = []

    n_c_h = []
    n_m_h = []

    for i in range(df.shape[0]):
        n_c_mw.append(df.complex_mw[i]/max(mw))
        n_m_mw.append(df.monomer_mw[i]/max(mw))
        n_c_v.append(df.complex_volume[i]/max(vol))
        n_m_v.append(df.monomer_volume[i]/max(vol))
        n_c_i.append(df.complex_isoelectric_point[i]/max(isop))
        n_m_i.append(df.monomer_isoelectric_point[i]/max(isop))
        n_c_h.append(df.complex_hidropathy[i]/max(hidro))
        n_m_h.append(df.monomer_hidropathy[i]/max(hidro))
    
    df['norm_c_mw'] = n_c_mw
    df['norm_m_mw'] = n_m_mw
    df['norm_c_vol'] = n_c_v
    df['norm_m_vol'] = n_m_v
    df['norm_c_pI'] = n_c_i
    df['norm_m_pI'] = n_m_i
    df['norm_c_hidropathy'] = n_c_h
    df['norm_m_hidropathy'] = n_m_mw

    return df

for i in range(len(norm_paths)):
    df = radius_features(pd.read_csv(feature_paths[i], sep = '\t'))
    df.to_csv(norm_paths[i], sep = '\t', index = False)


