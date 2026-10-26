import pandas as pd
import numpy as np 
from Bio.SeqUtils.ProtParam import ProteinAnalysis


wt = ProteinAnalysis("G")
mut = ProteinAnalysis("A")

df = pd.read_csv('../input/ddgit_dataset.tsv', sep = '\t')
# print(df.columns)
# ['unique', 'complex', 'pdb', 'chain', 'chain_id', 'wt', 'mut','position', 'location', 'ddg', 'ddgit_cc', 'ddgit_fold', 'lobso_cc', ...

wts = df.wt.to_list()
muts =  df.mut.to_list()

wt_mw = []
mut_mw = []
delta_mw = []
wt_a = []
mut_a = []
delta_aromaticity = []
# delta_instabilityidx = []
wt_i = []
mut_i = []
delta_iso = []
# delta_flexibility = []
wt_h = []
mut_h = []
delta_hidropathy = []

if len(wts) == len(muts):
    for i in range(len(muts)):
        # MWs and delta_mw
        wt_mw.append(ProteinAnalysis(wts[i]).molecular_weight())
        mut_mw.append(ProteinAnalysis(muts[i]).molecular_weight())
        delta_mw.append(ProteinAnalysis(muts[i]).molecular_weight() - ProteinAnalysis(wts[i]).molecular_weight()) 

        # Aromaticity and delta_aromaticity
        '''Calculate the aromaticity according to Lobry, 1994.
        Calculates the aromaticity value of a protein according to Lobry, 1994. It is simply the relative frequency of Phe+Trp+Tyr'''
        wt_a.append(ProteinAnalysis(wts[i]).aromaticity())
        mut_a.append(ProteinAnalysis(muts[i]).aromaticity())
        delta_aromaticity.append(ProteinAnalysis(muts[i]).aromaticity() - ProteinAnalysis(wts[i]).aromaticity())
        
        # delta_instabilityidx.append(ProteinAnalysis(muts[i]).instability_index() - ProteinAnalysis(wts[i]).instability_index()) only zeros

        # Isoelectric points and delta ip 
        wt_i.append(ProteinAnalysis(wts[i]).isoelectric_point())
        mut_i.append( ProteinAnalysis(muts[i]).isoelectric_point())
        delta_iso.append(ProteinAnalysis(muts[i]).isoelectric_point() - ProteinAnalysis(wts[i]).isoelectric_point())

        # delta_flexibility.append(ProteinAnalysis(muts[i]).flexibility() - ProteinAnalysis(wts[i]).flexibility()) uses window, to be used with sequence 

        # Hidropathies and delta_h
        '''Calculate the GRAVY (Grand Average of Hydropathy) according to Kyte and Doolitle, 1982.
        Utilizes the given Hydrophobicity scale, by default uses the original proposed by Kyte and Doolittle (KyteDoolitle). 
        Other options are: Aboderin, AbrahamLeo, Argos, BlackMould, BullBreese, Casari, Cid, Cowan3.4, Cowan7.5, Eisenberg, 
        Engelman, Fasman, Fauchere, GoldSack, Guy, Jones, Juretic, Kidera, Miyazawa, Parker,Ponnuswamy, Rose, Roseman, Sweet, 
        Tanford, Wilson and Zimmerman.'''
        wt_h.append(ProteinAnalysis(wts[i]).gravy()) 
        mut_h.append(ProteinAnalysis(muts[i]).gravy())
        delta_hidropathy.append(ProteinAnalysis(muts[i]).gravy() - ProteinAnalysis(wts[i]).gravy())

# code
# https://bioinformatics.stackexchange.com/questions/16417/python-script-to-calculate-properties-of-amino-acids-and-storing-them-in-matrix
# reference 
# https://link.springer.com/article/10.1186/s12859-017-1751-4#Sec15 
from quantiprot.metrics.aaindex import get_aa2volume

aa2vol = get_aa2volume()      # una volta, fuori dal ciclo
# print(aa2vol.mapping) 
#       {'A': 1.0, 'C': 2.43, 'D': 2.78, 'E': 3.78, 'F': 5.89, 'G': 0.0, 'H': 4.66, ...

wt_v = []
mut_v = []
delta_vol = []
for wt, mut in zip(wts, muts):
    wt_v.append(aa2vol.mapping[wt])
    mut_v.append(aa2vol.mapping[mut])
    delta_vol.append(aa2vol.mapping[mut] - aa2vol.mapping[wt])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (-1, 1))

def norm_list(l):
    ''' input a list (l) of values, 
        converts in np.array() to scale it (mean = 0, min = -1, max = 1)
        returns list of normalized values'''
    l = np.array(l).reshape(-1,1)
    scaler.fit(l)
    l = scaler.transform(l).flatten()
    l = [float(i) for i in l]
    return l 

n_wt_mw = norm_list(wt_mw)
n_mut_mw = norm_list(mut_mw)
n_delta_mw = norm_list(delta_mw)

n_wt_a = norm_list(wt_a)
n_mut_a = norm_list(mut_a)
n_delta_aromaticity = norm_list(delta_aromaticity)

n_wt_i = norm_list(wt_i)
n_mut_i = norm_list(mut_i)
n_delta_iso = norm_list(delta_iso)

n_wt_h = norm_list(wt_h)
n_mut_h = norm_list(mut_h)
n_delta_hidropathy = norm_list(delta_hidropathy)

n_wt_v = norm_list(wt_v)
n_mut_v = norm_list(mut_v)
n_delta_vol = norm_list(delta_vol)

aa_features = pd.DataFrame({'label': df.unique,
              'wt': df.wt,
              'mut': df.mut, 

              'wt_mw': wt_mw,
              'mut_mw': mut_mw,
              'delta_mw': delta_mw,

              'n_wt_mw': n_wt_mw,
              'n_mut_mw': n_mut_mw,
              'n_delta_mw': n_delta_mw,

              'wt_v': wt_v,
              'mut_v': mut_v,
              'delta_v': delta_vol,

              'n_wt_v': n_wt_v,
              'n_mut_v': n_mut_v,
              'n_delta_v': n_delta_vol,
              
              'wt_h': wt_h,
              'mut_h': mut_h,
              'delta_hidropathy': delta_hidropathy,

              'n_wt_h': n_wt_h,
              'n_mut_h': n_mut_h,
              'n_delta_hidropathy': n_delta_hidropathy,

              'wt_i': wt_i,
              'mut_i': mut_i,
              'delta_iso_p': delta_iso,

              'n_wt_i': n_wt_i,
              'n_mut_i': n_mut_i,
              'n_delta_iso_p': n_delta_iso,

              'wt_a': wt_a,
              'mut_a': mut_a,
              'delta_aromaticity': delta_aromaticity,

              'n_wt_a': n_wt_a,
              'n_mut_a': n_mut_a,
              'n_delta_aromaticity': n_delta_aromaticity
              })

aa_features.to_csv('../output/aa_features.tsv', sep = '\t', index = False)