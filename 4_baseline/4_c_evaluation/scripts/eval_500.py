import pandas as pd
import numpy as np

file = '../../4_b_mlp/output/best_combination/random_split/out.txt'

with open(file, 'r') as reader:
    for line in reader:
        if line.startswith('PCC: '):
            pcc = np.array([float(x) for x in line[5:].split(',')])
        else:
            loss = np.array([float(x) for x in line[6:].split(',')])

print("PCC")
print("Mean:", np.mean(pcc))
print("Std:", np.std(pcc))

print("\nLoss")
print("Mean:", np.mean(loss))
print("Std:", np.std(loss))
