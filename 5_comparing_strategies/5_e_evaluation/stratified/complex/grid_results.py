import pandas as pd 
import itertools

folders = []
for i in range(10):
        folders.append(f'out_{i+1}')

best = -1
b_combo = []
b_pcc = []
for i in range(10):
    for j in range(36):
            file = f'output_grid/{folders[i]}/{i+1}_{j+1}.txt'
            # print(file)
            df = pd.read_csv(file, sep = '\t')
            pcc = df.pcc.mean()
            if pcc > best:
                best = pcc 
                combo = j
    b_combo.append(combo)
    b_pcc.append(pcc)
    i += 1

df = pd.DataFrame({'best_combo': b_combo, 'pcc': b_pcc})

activations = ['relu', 'sigmoid']
dpouts = [0.1, 0.3, 0.7]
losses = ["L1", "MSE"]
architectures = [[32,16], [32,16,16], [128,64,32]]
combos = list(itertools.product(architectures, activations, losses, dpouts))

best_combination = []
for i in list(df.best_combo):
    best_combination.append(combos[i])

architecture = []
activation = []
loss = []
dropout = []
for i in best_combination:
    arc, act, l, dp = i
    architecture.append(arc)
    activation.append(act)
    loss.append(l)
    dropout.append(dp)
df['architecture'] = architecture
df['activation'] = activation
df['loss'] = loss 
df['dropout'] = dropout
df['out'] = [i for i in range(1,11)]

df.to_csv('grid_results.tsv', sep = '\t', index = False)