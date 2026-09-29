import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import ast


def single_scatter(ax, targets, predictions, title):

    df_plot = pd.DataFrame({
        "targets": targets,
        "predictions": predictions
    })

    sns.scatterplot(
        data=df_plot,
        x="targets",
        y="predictions",
        color="cornflowerblue",
        s=20,
        edgecolor="navy",
        linewidth=0.1,
        alpha=0.5,
        legend=True,
        ax=ax
    )

    min_val = -12.352
    max_val = 12.352
    pad = 0.05 * (max_val - min_val)

    ax.set_xlim(min_val - pad, max_val + pad)
    ax.set_ylim(min_val - pad, max_val + pad)

    # y = x
    ax.plot(
        [min_val- pad, max_val+ pad],
        [min_val- pad, max_val+ pad],
        color="orangered",
        linestyle="--",
        linewidth=1.5
    )

    # x = 0 e y = 0
    ax.axvline(0, color="orangered", linestyle=":", linewidth=1)
    ax.axhline(0, color="orangered", linestyle=":", linewidth=1)

    # medie
    mean_target = df_plot["targets"].mean()
    mean_prediction = df_plot["predictions"].mean()

    mean_diag = (mean_target + mean_prediction) / 2

    ax.scatter(
        mean_target,
        mean_target,
        marker="o",
        s=20,
        color="orangered",
        linewidth=1,
        zorder=10
    )

    pcc = np.corrcoef(targets, predictions)[0, 1]

    ax.text(
        0.05,
        0.95,
        f"PCC = {pcc:.3f}",
        transform=ax.transAxes,
        fontsize=25,
        verticalalignment="top",
        bbox=dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="grey"
        )
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(title)

#tr_file = '../../4_b_mlp/output/best_combination/final/train_predictions.tsv'
tr_file = '../../4_b_mlp/output/best_combination/stringent_split/train_predictions.tsv'
tr_df = pd.read_csv(tr_file, sep = '\t')
#val_file = '../../4_b_mlp/output/best_combination/final/validation_predictions.tsv'
val_file = '../../4_b_mlp/output/best_combination/stringent_split/validation_predictions.tsv'
val_df = pd.read_csv(val_file, sep = '\t')
#test_file = '../../4_b_mlp/output/best_combination/final/test_predictions.tsv'
test_file = '../../4_b_mlp/output/best_combination/stringent_split/test_predictions.tsv'
test_df = pd.read_csv(test_file, sep = '\t')

splits = pd.read_csv('../output/splits.tsv', sep = '\t')

for i in range(20):
    sns.set_theme(style="whitegrid", 
                rc={
                    "grid.color": "lightblue",       # Colore delle linee (es. "black", "#E0E0E0")
                    "grid.linestyle": ":",     # Tipo di linea ("-", "--", "-.", ":")
                    "grid.linewidth": 0.5,      # Spessore della linea
                    "axes.grid": True           # Forza l'attivazione della griglia
                },
                palette="viridis")

    plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 20,
            "axes.labelsize": 25,
            "axes.titlesize": 25,
            "xtick.labelsize": 25,
            "ytick.labelsize": 25,
            "legend.fontsize": 20
        })

    fig, axes = plt.subplots(
            1, 3,
            figsize=(30, 10)
        )

    tr_f = splits['Training'][i]
    val_f = splits['Validation'][i]
    test_f = splits['Test'][i]

    train_targets = ast.literal_eval(tr_df.loc[i, "best_targ"])
    train_predictions = ast.literal_eval(tr_df.loc[i, "best_pred"])
    single_scatter(axes[0],train_targets,train_predictions,f"Training: {tr_f}\n")

    val_targets = ast.literal_eval(val_df.loc[i, "best_targ"])
    val_predictions = ast.literal_eval(val_df.loc[i, "best_pred"])
    single_scatter(axes[1], val_targets, val_predictions, f"Validation: {val_f}\n")

    test_targets = ast.literal_eval(test_df.loc[i, 'best_targ'])
    test_predictions = ast.literal_eval(test_df.loc[i, 'best_pred'])
    single_scatter(axes[2], test_targets, test_predictions, f"Test: {test_f}\n")

    fig.suptitle(
            f"Run {i+1}\nBest Epoch: {tr_df.best_epoch[i]}\n",
            fontsize=30
        )

    plt.tight_layout()

    plt.savefig(
            f"../output/best_combination/20_splits/scatter/best_ep/{i+1}_scatter_{'_'.join(tr_f.split(' '))}_{val_f}_{test_f}.png",
            dpi=600,
            bbox_inches="tight"
        )

    plt.close()

    sns.set_theme(style="whitegrid", 
                rc={
                    "grid.color": "lightblue",       # Colore delle linee (es. "black", "#E0E0E0")
                    "grid.linestyle": ":",     # Tipo di linea ("-", "--", "-.", ":")
                    "grid.linewidth": 0.5,      # Spessore della linea
                    "axes.grid": True           # Forza l'attivazione della griglia
                },
                palette="viridis")

    plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 20,
            "axes.labelsize": 25,
            "axes.titlesize": 25,
            "xtick.labelsize": 25,
            "ytick.labelsize": 25,
            "legend.fontsize": 20
        })

    fig, axes = plt.subplots(
            1, 3,
            figsize=(30, 10)
        )

    train_targets = ast.literal_eval(tr_df.loc[i, "last_target"])
    train_predictions = ast.literal_eval(tr_df.loc[i, "last_pred"])
    single_scatter(axes[0],train_targets,train_predictions,f"Training: {tr_f}\n")

    val_targets = ast.literal_eval(val_df.loc[i, "last_target"])
    val_predictions = ast.literal_eval(val_df.loc[i, "last_pred"])
    single_scatter(axes[1], val_targets, val_predictions, f"Validation: {val_f}\n")

    test_targets = ast.literal_eval(test_df.loc[i, 'last_targ'])
    test_predictions = ast.literal_eval(test_df.loc[i, 'last_pred'])
    single_scatter(axes[2], test_targets, test_predictions, f"Test: {test_f}\n")

    fig.suptitle(
            f"Run {i+1}\nLast Epoch: {tr_df.last_epoch[i]}\n",
            fontsize=30
        )

    plt.tight_layout()

    plt.savefig(
            f"../output/best_combination/20_splits/scatter/last_ep/{i+1}_scatter_{'_'.join(tr_f.split(' '))}_{val_f}_{test_f}.png",
            dpi=600,
            bbox_inches="tight"
        )

    plt.close()
