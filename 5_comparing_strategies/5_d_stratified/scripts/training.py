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

from sklearn.model_selection import StratifiedKFold

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

# Sign of ddG, used ONLY to stratify the folds (the model is
# still trained on the continuous Y).
# 1 = ddG >= 0 (increases affinity), 0 = ddG < 0 (decreases).
D = (Y.numpy() >= 0).astype(int)

# Groups used by split_train_val to build the early-stopping
# validation set. No real groups here: every row is its own group.
# Replace with real groups (e.g. complex id) if needed.
groups = np.arange(len(Y))

# Rows of df, X, Y and V are matched by position.
assert len(df) == X.shape[0] == Y.shape[0] == len(V) == len(D), "df, X, Y, V and D have different lengths"

# ============================================================
# SETTINGS
# ============================================================

pat = 100
warm = 500
max_ep = 5000
activations = ['relu', 'sigmoid']
dpouts = [0.1, 0.3, 0.7]
losses = ["L1", "MSE"]
architectures = [[32,16], [32,16,16], [128,64,32]]

n_outer = 10            # outer folds (test)
n_inner = 10            # inner folds (grid search), rebuilt on each outer training set

val_frac = 0.1          # fraction of the training rows used for early stopping
batch_size = 64
seed = 42
monitor = "val_loss"    # early stopping / best epoch: "val_loss" or "val_pcc"
out_dir = '../output'

# Quick end-to-end check: 1 combination, few epochs, 2x2 folds.
SMOKE_TEST = False
if SMOKE_TEST:
    pat, warm, max_ep = 2, 2, 10
    activations, dpouts, losses, architectures = ['relu'], [0.1], ["MSE"], [[16]]
    n_outer, n_inner = 2, 2

combos = list(itertools.product(architectures, activations, losses, dpouts))

# Keep the output readable over thousands of trainings.
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

# One step per training: n_inner inner folds per combination,
# plus the refit, for each outer fold.
progress = tqdm(total=n_outer * (len(combos) * n_inner + 1), desc="trainings", unit="fit")

# Outer split: stratified by the sign of ddG.
outer_skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)

for i, (training_idx, test_idx) in enumerate(outer_skf.split(np.zeros(len(D)), D), start=1):

    log(f"outer {i} | train {len(training_idx)} (pos {D[training_idx].sum()}) "
        f"| test {len(test_idx)} (pos {D[test_idx].sum()})")

    # --------------------------------------------------------
    # INNER SPLITS: new stratified 10-fold CV on the outer
    # training set. Built once per outer fold, so every
    # combination is compared on exactly the same data.
    # --------------------------------------------------------

    inner_skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + i)

    inner_splits = []
    for j, (pool_pos, inner_test_pos) in enumerate(
        inner_skf.split(np.zeros(len(training_idx)), D[training_idx]), start=1
    ):
        # inner_skf returns positions inside training_idx:
        # map them back to row indices of the full dataset.
        pool_idx = training_idx[pool_pos]
        inner_test_idx = training_idx[inner_test_pos]

        # The pool is further split into training and validation
        # (early stopping); the inner test fold is never seen.
        train_idx, val_idx = split_train_val(pool_idx, seed=seed + 100 * i + j)
        inner_splits.append((j, train_idx, val_idx, inner_test_idx))

    # --------------------------------------------------------
    # INNER CV: grid search, one PCC per combination and fold
    # --------------------------------------------------------

    best_combo = None
    best_inner_pcc = -float("inf")

    for n, (h, a, l, d) in enumerate(combos, start=1):

        pccs = []

        for j, train_idx, val_idx, inner_test_idx in inner_splits:

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
    # REFIT: best combination on the whole outer training set,
    # then evaluation on the outer test fold
    # --------------------------------------------------------

    h, a, l, d = best_combo
    train_idx, val_idx = split_train_val(training_idx, seed=seed + 1000 * i)

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

fold_df = pd.DataFrame(fold_results)
log(f"Overall PCC (pooled): {overall_pcc:.4f}")
log(f"Overall MSE (pooled): {overall_mse:.4f}")
log(f"Mean test PCC over folds: {fold_df.test_pcc.mean():.4f} +/- {fold_df.test_pcc.std():.4f}")

pd.DataFrame({
    "label": all_labels,
    "fold": all_folds,
    "target": all_targets.numpy(),
    "prediction": all_preds.numpy(),
}).to_csv(f'{out_dir}/predictions.tsv', sep='\t', index=False)

log_file.close()