import pandas as pd 
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from quantiprot.metrics.aaindex import get_aa2volume

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

df = pd.read_csv('../output/aa_update_features.tsv', sep = '\t')

copia = df.copy()
        
max_mw, min_mw = max(mw), min(mw)
df['n_wt_mw'] = df['wt_mw'] / max_mw
df['n_mut_mw'] = df['mut_mw'] / max_mw

df['n_delta_mw'] = df['delta_mw'] / (max_mw - min_mw)

max_v, min_v = max(vol), min(vol)
df['n_wt_v'] = df['wt_v'] / max_v
df['n_mut_v'] = df['mut_v'] / max_v

df['n_delta_v'] = df['delta_v'] / (max_v - min_v)

max_h, min_h = max(hidro), min(hidro)
df['n_wt_h'] = df['wt_h'] / max_h
df['n_mut_h'] = df['mut_h'] / max_h

df['n_delta_hidropathy'] = df['delta_hidropathy'] / (max_h - min_h) 

max_i, min_i = max(isop), min(isop)
df['n_wt_i'] = df['wt_i'] / max_i
df['n_mut_i'] = df['mut_i'] / max_i

df['n_delta_iso_p'] = df['delta_iso_p'] / (max_i - min_i)

to_save = pd.DataFrame({'label': df.label, 
                        'wt': df.wt, 
                        'mut': df.mut, 
                        'wt_mw': df.wt_mw, 
                        'mut_mw': df.mut_mw, 
                        'delta_mw': df.delta_mw,
                        'n_wt_mw': df.n_wt_mw,
                        'n_mut_mw': df.n_mut_mw, 
                        'n_delta_mw': df.n_delta_mw, 
                        'wt_v': df.wt_v, 
                        'mut_v': df.mut_v, 
                        'delta_v': df.delta_v,
                        'n_wt_v': df.n_wt_v,
                        'n_mut_v': df.n_mut_v, 
                        'n_delta_v': df.n_delta_v,
                        'wt_h': df.wt_h, 
                        'mut_h': df.mut_h, 
                        'delta_hidropathy': df.delta_hidropathy, 
                        'n_wt_h': df.n_wt_h,
                        'n_mut_h': df.n_mut_h, 
                        'n_delta_hidropathy': df.n_delta_hidropathy, 
                        'wt_i': df.wt_i, 
                        'mut_i': df.mut_i, 
                        'delta_iso_p': df.delta_iso_p,
                        'n_wt_i': df.n_wt_i, 
                        'n_mut_i': df.n_mut_i, 
                        'n_delta_iso_p': df.n_delta_iso_p,
                        'wt_a': df.wt_a, 
                        'mut_a': df.mut_a, 
                        'delta_aromaticity': df.delta_aromaticity}) #, 
#                      'n_wt_a', 'n_mut_a', 'n_delta_aromaticity'})

to_save.to_csv('../output/normalized_update_features.tsv', sep = '\t')
