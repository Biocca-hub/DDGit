import pandas as pd 

df = pd.read_csv('../input/S4169.csv', sep = '\t')
print(df.head())

df['unique'] = df.pdb+'_'+df.mutation.str[1]+'_'+df.mutation

print(df.head())

mcsm = pd.DataFrame({'unique': df.unique,
                     'target': df.actual,
                     'prediction': df.mcsm_ppi2_prediction})

mcsm.to_csv('../output/mcsm_predictions.tsv', sep = '\t', index = False)