import pandas as pd 
from sklearn.metrics import mean_squared_error
import numpy as np

results = [f'../input/predictions_{i}.tsv' for i in ['ddgit', 'loco', 'lobso', 'stratified']]

strategies = ['ddgit', 'loco', 'lobso', 'stratified']
pcc = []
mse = []
rmse = []
for p in results:
    df = pd.read_csv(p, sep ='\t')
    pcc.append(np.corrcoef(df.target, df.prediction)[0, 1])
    mse.append(mean_squared_error(df.target, df.prediction))
    rmse.append(np.sqrt(mean_squared_error(df.target, df.prediction)))

print(pd.DataFrame({'CV': strategies,
              'PCC': pcc,
              'MSE': mse,
              'RMSE': rmse}))