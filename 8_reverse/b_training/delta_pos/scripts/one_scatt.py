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


fig, ax = plt.subplots(figsize=(10, 15))

df = pd.read_csv('../output/predictions_S8338_stratified.tsv', sep='\t')
single_scatter(ax, df, "Predictions on 10 fold CV\n(Stratified split)\n")

plt.tight_layout()
plt.savefig("../output/stratified_S8338_scatter.png", dpi=600, bbox_inches="tight")
plt.close()
