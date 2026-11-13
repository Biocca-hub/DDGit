import pandas as pd
import numpy as np
import torch
import seaborn as sns 
import matplotlib.pyplot as plt

def pearson_corr(y_hat, y): 
    vx = y_hat - torch.mean(y_hat) # predictions
    vy = y - torch.mean(y) # targets
    return torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-8) 

file = 'pred_files.txt'
files = []
with open(file, 'r') as reader:
    for line in reader:
        files.append(f'../input/pred_loco/{line.strip()}')

dfs = [pd.read_csv(f, sep='\t') for f in files]
df = pd.concat(dfs, ignore_index=True)
df.to_csv('../output/predictions_loco.tsv', sep = '\t', index = False)

predictions = torch.tensor(df.prediction)
targets = torch.tensor(df.target)

print(float(pearson_corr(predictions, targets)))
