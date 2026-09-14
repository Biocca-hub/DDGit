# IMPORTARE FUNZIONE PER CROSS VALIDATION SUI 20 SPLIT RIGOROSI
#import os
import pandas as pd
import numpy as np
#from model_parametrized import cv_run as c

# DEFINIZIONE IPERPARAMETRI PER CROSS VALIDATION
#folder = '../output'
pat = 100
warm = 500
max_ep = 5000
#activation = 'relu' 
activations = ['relu', 'sigmoid']
dpouts = [0.1, 0.3, 0.5, 0.7] 
losses = ["L1", "MSE"]
h_dims = [[16], [128], [32,16], [128,32], [32,16,16], [128,64,32]]
# TRAINING

pcc_mean = [] 
pcc_std = []
pcc_max = []
pcc_min = []
layers = []
actf = []
dps = []
lfun = []
for hidden_dims in h_dims:
    for activation in activations:
        for dpout in dpouts:
            for loss_f in losses:
                folder = f"../output/grid_search/{activation}_{loss_f}_{'_'.join(map(str, hidden_dims))}_{dpout}"
                out = f'{folder}/out.txt'
                df = pd.read_csv(out, sep = '\t')
                #print(out)
                pcc_mean.append(np.mean(df.PCC))
                pcc_std.append(np.std(df.PCC))
                pcc_max.append(np.max(df.PCC))
                pcc_min.append(np.min(df.PCC))
                layers.append(hidden_dims)
                actf.append(activation)
                dps.append(dpout)
                lfun.append(loss_f)
            

df = pd.DataFrame({'Hidden_Layers': layers,
                   'Activation_function': actf,
                   'Dropout': dps,
                   'Loss_function': lfun,
                   'PCC_avg': pcc_mean,
                   'PCC_std': pcc_std,
                   'PCC_max': pcc_max,
                   'PCC_min': pcc_min})
df = df.sort_values(by='PCC_avg', ascending = False, ignore_index = True)
#print(df.PCC_avg.head())
df.to_csv('../output/grid_search/final_eval.tsv', sep = '\t', index = False)
