import pandas as pd 
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from quantiprot.metrics.aaindex import get_aa2volume

aa_list = list("GAVPLIMFWYSTCNQHDEKR")

df = pd.read_csv('../output/aa_features.tsv', sep = '\t')

# print(df.columns)
#'label', 'wt', 'mut', 'wt_mw', 'mut_mw', 'delta_mw', 
#                      'n_wt_mw','n_mut_mw', 'n_delta_mw', 
#                      'wt_v', 'mut_v', 'delta_v', 
#                      'n_wt_v','n_mut_v', 'n_delta_v', 
#                      'wt_h', 'mut_h', 'delta_hidropathy', 
#                      'n_wt_h','n_mut_h', 'n_delta_hidropathy', 
#                      'wt_i', 'mut_i', 'delta_iso_p',
#                      'n_wt_i', 'n_mut_i', 'n_delta_iso_p', 
#                      'wt_a', 'mut_a', 'delta_aromaticity', 
#                      'n_wt_a', 'n_mut_a', 'n_delta_aromaticity'

copia = df.copy()
        
aa2vol = get_aa2volume() 

mw = [ProteinAnalysis(i).molecular_weight() for i in aa_list] #
arom = [ProteinAnalysis(i).aromaticity() for i in aa_list]
isop = [ProteinAnalysis(i).isoelectric_point() for i in aa_list] #
hidro = [ProteinAnalysis(i).gravy() for i in aa_list] #
vol = [aa2vol.mapping[i] for i in aa_list] #


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

'''print(max(df.mut_a))
print(max(arom))
print(max(df.wt_a))
print(min(df.mut_a))
print(min(arom))
print(min(df.wt_a))
print(sum(df.mut_a)/len(df.mut_a))
print(sum(arom)/len(arom))
print(sum(df.wt_a)/len(df.wt_a))'''

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

to_save.to_csv('../output/normalized_aa_features.tsv', sep = '\t')
