import pandas as pd
#from model_parametrized import cv_run as c

from save_predictions import cv_run as c


# DEFINIZIONE IPERPARAMETRI PER CROSS VALIDATION
folder = '../output/best_combination/good_split_all_predictions'
pat = 100
warm = 500
max_ep = 5000
activation = 'relu' 
dpout = 0.7 
loss = "MSE"
h_dims = [128,64,32]
output = folder+'/out.txt'
scatter = folder+'/scat.txt'

# TRAINING

p_max, p_min, p_mean, p_std = c(h_dims, pat, max_ep, warm, output, scatter, folder, activation, dpout, loss)

df = pd.DataFrame({
                   'PCC_avg': p_mean,
                   'PCC_std': p_std,
                   'PCC_max': p_max,
                   'PCC_min': p_min}
                   )

pd.to_csv('output/final_eval.tsv', sep = '\t')

