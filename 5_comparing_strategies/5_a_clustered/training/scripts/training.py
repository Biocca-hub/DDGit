import os
import itertools
import logging
import warnings

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from tqdm import tqdm

from baseline import MLP, EarlyStoppingWithWarmup, pearson_corr

# ============================================================
# DATA
# ============================================================

df = pd.read_csv('../input/ddgit_dataset.tsv', sep = '\t')
X = torch.load('../input/features.pt')
Y = torch.load('../input/targets.pt')
V = np.array(torch.load('../input/labels.pt'))

X = X.float()
# The model returns shape [N]: a target of shape [N, 1] would
# broadcast silently in the loss.
Y = Y.float().view(-1)

# Rows of df, X, Y and V are matched by position.
assert len(df) == X.shape[0] == Y.shape[0] == len(V), "df, X, Y and V have different lengths"

# Fold of each row and group of each row. All rows that share the
# same group must stay on the same side of every split.
folds = df['ddgit_fold'].to_numpy()
groups = np.array([str(v).split('_')[0] for v in V])

# ============================================================
# SETTINGS
# ============================================================

pat = 100
warm = 500
max_ep = 5000
activations = ['relu', 'sigmoid']
dpouts = [0.1, 0.3, 0.5, 0.7]
losses = ["L1", "MSE"]
architectures = [[16], [128], [32,16], [128,32], [32,16,16], [128,64,32]]

val_frac = 0.1          # fraction of the training rows used for early stopping
batch_size = 64         # not defined in the original code: set it as needed
seed = 42
monitor = "val_loss"    # early stopping / best epoch: "val_loss" or "val_pcc"
out_dir = '../output'

# Quick end-to-end check: 2 combinations and a few epochs.
SMOKE_TEST = False
if SMOKE_TEST:
    pat, warm, max_ep = 2, 2, 10
    activations, dpouts, architectures = ['relu'], [0.1], [[16]]

combos = list(itertools.product(architectures, activations, losses, dpouts))

# Keep the output readable over ~2000 trainings.
for name in ("lightning.pytorch", "lightning.fabric"):
    logging.getLogger(name).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*does not have many workers.*")

# ============================================================
# HELPERS
# ============================================================

def split_train_val(idx, seed):
    """
    Split the rows `idx` into training and validation.

    Whole groups are moved to validation, in random order, until
    the validation set holds about `val_frac` of the rows.
    """

    rng = np.random.default_rng(seed)

    uniq, counts = np.unique(groups[idx], return_counts=True)
    order = rng.permutation(len(uniq))
    cum = np.cumsum(counts[order])
    target = val_frac * len(idx)

    # Number of groups whose total size is closest to the target.
    k = int(np.searchsorted(cum, target)) + 1
    if k > 1 and abs(cum[k - 2] - target) < abs(cum[min(k, len(cum)) - 1] - target):
        k -= 1
    k = max(1, min(k, len(uniq) - 1))

    is_val = np.isin(groups[idx], uniq[order[:k]])

    return idx[~is_val], idx[is_val]


def rows(tensor, idx):
    """Select rows of a tensor by position."""
    return tensor[torch.as_tensor(idx, dtype=torch.long)]


def fit_model(train_idx, val_idx, h, a, l, d):
    """
    Train one MLP with early stopping on `val_idx` and return it
    with the weights of its best epoch.
    """

    pl.seed_everything(seed, workers=True)

    train_loader = DataLoader(
        TensorDataset(rows(X, train_idx), rows(Y, train_idx)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(rows(X, val_idx), rows(Y, val_idx)),
        batch_size=len(val_idx),
    )

    model = MLP(
        input_dim=X.shape[1],
        hidden_dims=h,
        activation=a,
        dpout=d,
        loss_f=l,
        monitor=monitor,
    )

    early_stopping = EarlyStoppingWithWarmup(
        monitor=monitor,
        mode="min" if monitor == "val_loss" else "max",
        patience=pat,
        warmup=warm,
    )

    trainer = pl.Trainer(
        max_epochs=max_ep,
        callbacks=[early_stopping],
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)

    # Training ends `pat` epochs after the best one.
    model.load_best_weights()

    return model


def log(msg):
    """
    Write a message to the terminal and to the log file.

    `tqdm.write` prints above the progress bar without breaking it.
    The bar itself is drawn only on the terminal, never in the file.
    """
    tqdm.write(msg)
    log_file.write(msg + "\n")
    log_file.flush()


@torch.no_grad()
def predict(model, idx):
    """Predictions for the rows `idx`, with dropout switched off."""
    model.eval()
    return model(rows(X, idx))

# ============================================================
# NESTED CROSS VALIDATION
# ============================================================

all_preds = []
all_targets = []
all_labels = []
all_folds = []

fold_results = []
inner_results = []

os.makedirs(out_dir, exist_ok=True)
log_file = open(f'{out_dir}/log.txt', 'w')

# The 5 folds are taken as they are: only warn if they split a group.
folds_per_group = pd.Series(folds).groupby(groups).nunique()
if (folds_per_group > 1).any():
    log(f"WARNING: {(folds_per_group > 1).sum()} groups are spread over more than one ddgit_fold")

# One step per training: 4 inner folds per combination, plus the
# refit, for each of the 5 outer folds.
progress = tqdm(total=5 * (len(combos) * 4 + 1), desc="trainings", unit="fit")

for i in range(1,6):

    # Outer split: fold i is the test set, the other 4 folds
    # enter the inner cross validation.
    test_idx = np.where(folds == i)[0]
    training_idx = np.where(folds != i)[0]
    inner_folds = [j for j in range(1,6) if j != i]

    # --------------------------------------------------------
    # INNER CV: one PCC per combination and inner fold
    # --------------------------------------------------------

    best_combo = None
    best_inner_pcc = -float("inf")

    for n, (h, a, l, d) in enumerate(combos, start=1):

        pccs = []

        for j in inner_folds:

            # Fold j is the inner test set; the other 3 folds are
            # split into training and validation (early stopping).
            # The split depends only on (i, j), so every combination
            # is compared on the same data.
            inner_test_idx = np.where(folds == j)[0]
            pool_idx = np.where((folds != i) & (folds != j))[0]
            train_idx, val_idx = split_train_val(pool_idx, seed=seed + 10 * i + j)

            model = fit_model(train_idx, val_idx, h, a, l, d)
            progress.update(1)

            pcc = pearson_corr(predict(model, inner_test_idx), rows(Y, inner_test_idx)).item()
            pccs.append(pcc)

            inner_results.append({
                "outer_fold": i, "inner_fold": j,
                "architecture": str(h), "activation": a, "loss": l, "dropout": d,
                "pcc": pcc, "best_epoch": model.best_epoch,
            })

        mean_pcc = float(np.mean(pccs))
        log(f"outer {i} | combo {n}/{len(combos)} {h} {a} {l} {d} | inner PCC {mean_pcc:.4f}")

        if mean_pcc > best_inner_pcc:
            best_inner_pcc = mean_pcc
            best_combo = (h, a, l, d)

    # --------------------------------------------------------
    # REFIT: best combination on the 4 folds, then outer test
    # --------------------------------------------------------

    h, a, l, d = best_combo
    train_idx, val_idx = split_train_val(training_idx, seed=seed + 100 * i)

    model = fit_model(train_idx, val_idx, h, a, l, d)
    progress.update(1)

    preds = predict(model, test_idx)
    targets = rows(Y, test_idx)

    all_preds.append(preds)
    all_targets.append(targets)
    all_labels.extend(list(V[test_idx]))
    all_folds.extend([i] * len(test_idx))

    fold_results.append({
        "outer_fold": i,
        "architecture": str(h), "activation": a, "loss": l, "dropout": d,
        "inner_pcc": best_inner_pcc,
        "test_pcc": pearson_corr(preds, targets).item(),
        "test_mse": torch.mean((preds - targets) ** 2).item(),
        "best_epoch": model.best_epoch,
    })
    log(f"outer {i} | best {best_combo} | test PCC {fold_results[-1]['test_pcc']:.4f} | test MSE {fold_results[-1]['test_mse']:.4f}")

    # Saved after every outer fold, so a stopped run keeps its results.
    pd.DataFrame(inner_results).to_csv(f'{out_dir}/inner_cv_results.tsv', sep='\t', index=False)
    pd.DataFrame(fold_results).to_csv(f'{out_dir}/fold_results.tsv', sep='\t', index=False)

# ============================================================
# OVERALL METRICS
# ============================================================

progress.close()

all_preds = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

overall_pcc = pearson_corr(all_preds, all_targets).item()
overall_mse = torch.mean((all_preds - all_targets) ** 2).item()

log(f"Overall PCC: {overall_pcc:.4f}")
log(f"Overall MSE: {overall_mse:.4f}")

pd.DataFrame({
    "label": all_labels,
    "fold": all_folds,
    "target": all_targets.numpy(),
    "prediction": all_preds.numpy(),
}).to_csv(f'{out_dir}/predictions.tsv', sep='\t', index=False)

log_file.close()