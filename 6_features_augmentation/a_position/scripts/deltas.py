import pandas as pd
import numpy as np 
from Bio.SeqUtils.ProtParam import ProteinAnalysis

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

wt_v = []
mut_v = []
delta_vol = []

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
        #wt_i.append(ProteinAnalysis(wts[i]).isoelectric_point())
        #mut_i.append( ProteinAnalysis(muts[i]).isoelectric_point())
        #delta_iso.append(ProteinAnalysis(muts[i]).isoelectric_point() - ProteinAnalysis(wts[i]).isoelectric_point())

        wt_i.append(aa_pI[wts[i]])
        mut_i.append( aa_pI[muts[i]])
        delta_iso.append(aa_pI[muts[i]] - aa_pI[wts[i]])

        # Volume 
        wt_v.append(aa_volume[wts[i]])
        mut_v.append(aa_volume[muts[i]])
        delta_vol.append(aa_volume[muts[i]] - aa_volume[wts[i]])

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
# from quantiprot.metrics.aaindex import get_aa2volume

# aa2vol = get_aa2volume()      # una volta, fuori dal ciclo
# print(aa2vol.mapping) 
#       {'A': 1.0, 'C': 2.43, 'D': 2.78, 'E': 3.78, 'F': 5.89, 'G': 0.0, 'H': 4.66, ...

aa_features = pd.DataFrame({'label': df.unique,
              'wt': df.wt,
              'mut': df.mut, 

              'wt_mw': wt_mw,
              'mut_mw': mut_mw,
              'delta_mw': delta_mw,

              'wt_v': wt_v,
              'mut_v': mut_v,
              'delta_v': delta_vol,
              
              'wt_h': wt_h,
              'mut_h': mut_h,
              'delta_hidropathy': delta_hidropathy,

              'wt_i': wt_i,
              'mut_i': mut_i,
              'delta_iso_p': delta_iso,

              'wt_a': wt_a,
              'mut_a': mut_a,
              'delta_aromaticity': delta_aromaticity})

aa_features.to_csv('../output/aa_update_features.tsv', sep = '\t', index = False)