import pandas as pd
import itertools
activations = ['relu', 'sigmoid']
dpouts = [0.1, 0.3, 0.5, 0.7]
losses = ["L1", "MSE"]
architectures = [[16], [128], [32, 16], [128, 32], [32, 16, 16], [128, 64, 32]]

combos = list(itertools.product(architectures, activations, losses, dpouts))

'''
ddgit = pd.read_csv('../input/predictions_ddgit.tsv', sep = '\t')
best_ddgit = [i for i in ddgit['combo'].unique()]
print(best_ddgit)

best_stratified = [23, 22]
'''

arch = []
act = []
l = []
d = []
for i in [71, 12, 67, 9, 23, 22]:
    arch.append(combos[i][0])
    act.append(combos[i][1])
    l.append(combos[i][2])
    d.append(combos[i][3])

best_combos = pd.DataFrame({'clustering': ['ddgit', 'ddgit', 'ddgit', 'ddgit', 'stratified', 'stratified'],
             'combo': [71, 12, 67, 9, 23, 22],
             'architecture': arch,
             'activation': act,
             'loss': l,
             'dropout': d})

best_combos.to_csv('../output/best_combos.tsv', sep = '\t')