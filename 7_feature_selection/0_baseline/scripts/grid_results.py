import pandas as pd
import itertools

n_outer = 10
n_combo = 32

activations = ["sigmoid"]
dpouts = [0.1, 0.3, 0.5, 0.7]
losses = ["L1", "MSE"]
architectures = [[16], [128], [32, 16], [128, 32]]

# deve avere lo stesso ordine usato in training_grid_slurm.py
combos = list(itertools.product(architectures, activations, losses, dpouts))
assert len(combos) == n_combo

rows = []
for outer in range(1, n_outer + 1):
    best_pcc = -1          # azzerato per ogni outer fold
    best_combo = None

    for combo in range(1, n_combo + 1):
        file = f'../input/grid_stratified/out_{outer}/{outer}_{combo}.txt'
        df = pd.read_csv(file, sep='\t')
        pcc = df.pcc.mean()            # media sulle inner fold
        if pcc > best_pcc:
            best_pcc = pcc
            best_combo = combo

    arc, act, loss, dp = combos[best_combo - 1]    # file 1-based -> lista 0-based
    train_folds = [f for f in range(1, n_outer + 1) if f != outer]

    rows.append({
        'outer_fold': outer,
        'train_folds': ','.join(map(str, train_folds)),   # es. "2,3,4,5"
        'best_combo': best_combo,                          # 1-based, come i file
        'pcc': best_pcc,
        'architecture': arc,
        'activation': act,
        'loss': loss,
        'dropout': dp,
    })

df = pd.DataFrame(rows)
print(df)
df.to_csv('stratified_grid_results.tsv', sep='\t', index=False)