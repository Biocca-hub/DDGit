import pandas as pd
from model_parametrized import random_run as r

# DEFINIZIONE IPERPARAMETRI PER CROSS VALIDATION
folder = '../output/best_combination/random_split'
pat = 100
warm = 500
max_ep = 5000
activation = 'relu' 
dpout = 0.7 
loss = "MSE"
h_dims = [128,64,32]
output = folder+'/out.txt'
scatter = folder+'/scat.txt'
runs = 500

# TRAINING

p_mean, p_std, p_max, p_min = r(h_dims, pat, max_ep, warm, output, scatter, runs, folder, activation, dpout, loss)

df = pd.DataFrame({
                   'PCC_avg': p_mean,
                   'PCC_std': p_std,
                   'PCC_max': p_max,
                   'PCC_min': p_min}
                   )

pd.to_csv(folder+'/final_eval.tsv', sep = '\t')

