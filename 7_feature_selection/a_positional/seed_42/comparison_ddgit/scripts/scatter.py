import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np

def single_scatter(ax, df, title):

    sns.scatterplot(
        data=df,
        x="target",
        y="prediction",
        color="cornflowerblue",
        s=20,
        edgecolor="navy",
        linewidth=0.1,
        alpha=0.5,
        legend=True,
        ax=ax)

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
    mean_target = df["target"].mean()
    mean_prediction = df["prediction"].mean()

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

    pcc = np.corrcoef(df.target, df.prediction)[0, 1]

    ax.text(0.05,
            0.95,
            f"PCC = {pcc:.3f}",
            transform=ax.transAxes,
            fontsize=25,
            verticalalignment="top",
            bbox=dict(facecolor="white",
                      alpha=0.8,
                      edgecolor="grey"))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(title)

results = [f'../input/predictions_{i}.tsv' for i in ['ddgit', 'loco', 'lobso', 'stratified']]

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

fig, axes = plt.subplots(1, 4,figsize=(30, 10))

single_scatter(axes[0],pd.read_csv(results[0], sep ='\t'),f"Predictions on 5 fold CV\n(Connected Components Clustering)\n")
single_scatter(axes[1],pd.read_csv(results[1], sep ='\t'),f"Predictions on LOBSO\n(Leave One Binding Site Out)\n")
single_scatter(axes[2],pd.read_csv(results[2], sep ='\t'),f"Predictions on LOCO\n(Leave One Complex Out)\n")
single_scatter(axes[3],pd.read_csv(results[3], sep ='\t'),f"Predictions on 10 fold stratified CV\n")
    
fig.suptitle("Performance on different training procedures", fontsize=30)

plt.tight_layout()

plt.savefig(f"../output/comparison_scatter.png",
            dpi=600,
            bbox_inches="tight")

plt.close()

