import pandas as pd
import numpy as np
import torch

def pearson_corr(y_hat, y): 
    vx = y_hat - torch.mean(y_hat) # predictions
    vy = y - torch.mean(y) # targets
    return torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-8) 

file = 'predictions.tsv'

df = pd.read_csv(file, sep='\t')

predictions = torch.tensor(df.prediction)
targets = torch.tensor(df.target)

print(float(pearson_corr(predictions, targets)))

