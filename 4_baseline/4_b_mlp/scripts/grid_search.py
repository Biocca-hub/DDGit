# IMPORTARE FUNZIONE PER CROSS VALIDATION SUI 20 SPLIT RIGOROSI
import os
import pandas as pd
from model_parametrized import cv_run as c

# DEFINIZIONE IPERPARAMETRI PER CROSS VALIDATION
folder = '../output'
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
hyperparams = []
for hidden_dims in h_dims:
    for activation in activations:
        for dpout in dpouts:
            for loss_f in losses:
                folder = f"output/{activation}_{loss_f}_{'_'.join(map(str, hidden_dims))}_{dpout}"
                os.makedirs(folder, exist_ok=True)
                output = folder+'/out.txt'
                scatter = folder+'/scat.txt'
                p_max, p_min, p_mean, p_std = c(hidden_dims, pat, max_ep, warm, output, scatter, folder, activation, dpout, loss_f)
                pcc_max.append(p_max)
                pcc_min.append(p_min)
                pcc_mean.append(p_mean)
                pcc_std.append(p_std)
                hyperparams.append(folder[7:]) 

df = pd.DataFrame({'Hyperparameters': hyperparams,
                   'PCC_avg': pcc_mean,
                   'PCC_std': pcc_std,
                   'PCC_max': pcc_max,
                   'PCC_min': pcc_min})
pd.to_csv('output/final_eval.tsv', sep = '\t')