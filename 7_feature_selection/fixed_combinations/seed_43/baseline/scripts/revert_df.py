import pandas as pd

results = pd.read_csv('../output/results.tsv', sep = '\t')

print(results.T)