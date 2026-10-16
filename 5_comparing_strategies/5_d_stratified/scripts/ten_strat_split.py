import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold

# ============================================================
# DATA
# ============================================================

df = pd.read_csv('../input/ddgit_dataset_str.tsv', sep='\t')
Y = torch.load('../input/targets.pt').float().view(-1)
V = np.array(torch.load('../input/labels.pt'))

# Sign of ddG: 1 = ddG >= 0, 0 = ddG < 0 (only for stratification).
D = (Y.numpy() >= 0).astype(int)

# Outer fold labels (1..10) saved in the previous step.
folds = df['stratified_cc'].to_numpy()

# df, Y and V must be in the same row order, otherwise the labels
# in stratified_cc do not refer to the right mutations.
assert len(df) == len(D) == len(V), "df, Y and V have different lengths"
assert (df['unique'].astype(str).to_numpy() == V.astype(str)).all(), "df and V are not in the same order"

# ============================================================
# SETTINGS
# ============================================================

n_outer = 10
n_inner = 10
n_combos = 36
seed = 42

# ============================================================
# CHECK: same inner folds for all 36 combinations?
# ============================================================

for i in range(1, n_outer + 1):

    # Leave out outer group i, keep the other 9.
    test_idx = np.where(folds == i)[0]
    training_idx = np.where(folds != i)[0]

    reference = None

    for c in range(1, n_combos + 1):

        # Same construction as in training.py: rebuilt for every
        # combination here, to check that it gives the same result.
        inner_skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + i)

        # inner_labels[k] = inner fold (1..10) of row training_idx[k]
        inner_labels = np.zeros(len(training_idx), dtype=int)
        for j, (_, inner_test_pos) in enumerate(
            inner_skf.split(np.zeros(len(training_idx)), D[training_idx]), start=1
        ):
            inner_labels[inner_test_pos] = j

        if reference is None:
            reference = inner_labels
        elif not np.array_equal(inner_labels, reference):
            print(f"outer {i} | combo {c}: inner folds DIFFERENT from combo 1")
            break
    else:
        # Size and positives of each inner fold (for the stratification).
        sizes = [int((reference == j).sum()) for j in range(1, n_inner + 1)]
        pos = [int(D[training_idx][reference == j].sum()) for j in range(1, n_inner + 1)]
        print(f"outer {i} | test {len(test_idx)} (pos {D[test_idx].sum()}) | "
              f"same inner folds in all {n_combos} combos")
        print(f"    inner sizes: {sizes}")
        print(f"    inner pos:   {pos}")
    