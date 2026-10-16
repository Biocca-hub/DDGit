import ast
import pandas as pd

df = pd.read_csv('../input/fold_results.tsv', sep='\t')

cols = ['architecture', 'activation', 'loss', 'dropout']
df = df.drop_duplicates(subset=cols)

# architecture è salvata come stringa "[32, 16]": la riconvertiamo in lista.
combinations = [(ast.literal_eval(r.architecture), r.activation, r.loss, float(r.dropout)) for r in df.itertuples()]

#print(len(combinations))