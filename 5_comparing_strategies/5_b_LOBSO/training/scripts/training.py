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

from best_combinations import combinations

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
folds = df['lobso_cc'].to_numpy()
groups = np.array([str(v).split('_')[0] for v in V])

# ============================================================
# SETTINGS
# ============================================================

pat = 100
warm = 500
max_ep = 5000
# h, a, l, d = [32, 16, 16], 'relu', 'MSE', 0.7
combos = combinations

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

#combos = list(itertools.product(architectures, activations, losses, dpouts))


# Keep the output readable over ~2000 trainings.
for name in ("lightning.pytorch", "lightning.fabric"):
    logging.getLogger(name).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*does not have many workers.*")

# ============================================================
# HELPERS
# ============================================================

def split_train_val(idx, seed, by=None):
    """
    Split the rows `idx` into training and validation.

    Whole units of `by` (default: `groups`) are moved to validation,
    in random order, until it holds about `val_frac` of the rows.
    """
    by = groups if by is None else by
    rng = np.random.default_rng(seed)

    uniq, counts = np.unique(by[idx], return_counts=True)
    order = rng.permutation(len(uniq))
    cum = np.cumsum(counts[order])
    target = val_frac * len(idx)

    k = int(np.searchsorted(cum, target)) + 1
    if k > 1 and abs(cum[k - 2] - target) < abs(cum[min(k, len(cum)) - 1] - target):
        k -= 1
    k = max(1, min(k, len(uniq) - 1))

    is_val = np.isin(by[idx], uniq[order[:k]])
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
# CROSS VALIDATION (5 combinazioni fisse)
# ============================================================

os.makedirs(out_dir, exist_ok=True)
log_file = open(f'{out_dir}/log.txt', 'w')

folds_per_group = pd.Series(folds).groupby(groups).nunique()
if (folds_per_group > 1).any():
    log(f"WARNING: {(folds_per_group > 1).sum()} groups are spread over more than one fold")

outer_folds = np.unique(folds)
log(f"lobso_cc: {len(outer_folds)} folds | {len(combos)} combinations")

progress = tqdm(total=len(outer_folds) * len(combos), desc="trainings", unit="fit")

fold_results, combo_results, pred_tables = [], [], []

for c, (h, a, l, d) in enumerate(combos, start=1):

    preds_c, targets_c, labels_c, folds_c = [], [], [], []

    for i in outer_folds:

        # Il fold i è il test set, tutti gli altri sono training (+ validation per l'early stopping).
        # Lo split dipende solo da i: tutte le combinazioni usano gli stessi dati.
        test_idx = np.where(folds == i)[0]
        training_idx = np.where(folds != i)[0]
        train_idx, val_idx = split_train_val(training_idx, seed=seed + 100 * int(i), by=folds)

        model = fit_model(train_idx, val_idx, h, a, l, d)
        progress.update(1)

        preds = predict(model, test_idx)
        targets = rows(Y, test_idx)

        preds_c.append(preds)
        targets_c.append(targets)
        labels_c.extend(list(V[test_idx]))
        folds_c.extend([i] * len(test_idx))

        fold_results.append({
            "combo": c, "outer_fold": i, "n_test": len(test_idx),
            "architecture": str(h), "activation": a, "loss": l, "dropout": d,
            # Con un fold di 1-2 righe il PCC non ha senso.
            "test_pcc": pearson_corr(preds, targets).item() if len(test_idx) > 2 else float("nan"),
            "test_mse": torch.mean((preds - targets) ** 2).item(),
            "best_epoch": model.best_epoch,
        })

    preds_c = torch.cat(preds_c)
    targets_c = torch.cat(targets_c)

    pcc = pearson_corr(preds_c, targets_c).item()
    mse = torch.mean((preds_c - targets_c) ** 2).item()
    log(f"combo {c}/{len(combos)} {h} {a} {l} {d} | overall PCC {pcc:.4f} | MSE {mse:.4f} | RMSE {mse ** 0.5:.4f}")

    combo_results.append({
        "combo": c, "architecture": str(h), "activation": a, "loss": l, "dropout": d,
        "overall_pcc": pcc, "overall_mse": mse, "overall_rmse": mse ** 0.5,
    })
    pred_tables.append(pd.DataFrame({
        "combo": c, "label": labels_c, "fold": folds_c,
        "target": targets_c.numpy(), "prediction": preds_c.numpy(),
    }))

    # Salvato dopo ogni combinazione, così un run interrotto tiene i risultati.
    pd.DataFrame(fold_results).to_csv(f'{out_dir}/fold_results.tsv', sep='\t', index=False)
    pd.DataFrame(combo_results).to_csv(f'{out_dir}/combo_results.tsv', sep='\t', index=False)
    pd.concat(pred_tables).to_csv(f'{out_dir}/predictions.tsv', sep='\t', index=False)

progress.close()
log_file.close()