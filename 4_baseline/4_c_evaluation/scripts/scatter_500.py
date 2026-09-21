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
        color="royalblue",
        s=10,
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
        linewidth=1
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
        linewidth=0.5,
        zorder=10
    )

    pcc = np.corrcoef(targets, predictions)[0, 1]

    ax.text(
        0.05,
        0.95,
        f"PCC = {pcc:.3f}",
        transform=ax.transAxes,
        fontsize=8,
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

file = '../../4_b_mlp/output/best_combination/random_split/scat.txt'

with open(file, 'r') as reader:
    ps = []
    ts = []
    for line in reader:
        if line.startswith('Predictions'):
            predictions = line.split(' ')[1:]
            ps.append([float(val[:-1]) for val in predictions])
            
        elif line.startswith('Targets'):
            targets = line.split(' ')[1:]
            ts.append([float(val[:-1]) for val in targets])

all_ps = []
all_tgs = []
for i in range(len(ps)):
    #df = pd.DataFrame({'predictions': ps[i], 'targets': ts[i]})
    #figname = f'../output/best_plots/scatter_run_{i+1}'
    #scatter_plot(df, figname)
    all_ps.extend(ps[i])
    all_tgs.extend(ts[i])

df = pd.DataFrame({'predictions': all_ps, 'targets': all_tgs})

sns.set_theme(style="whitegrid", 
                rc={
                    "grid.color": "lightblue",       # Colore delle linee (es. "black", "#E0E0E0")
                    "grid.linestyle": ":",     # Tipo di linea ("-", "--", "-.", ":")
                    "grid.linewidth": 0.3,      # Spessore della linea
                    "axes.grid": True           # Forza l'attivazione della griglia
                },
                palette="viridis")

plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8
        })


fig, ax = plt.subplots(figsize=(10, 10))

single_scatter(
    ax,
    df.targets,
    df.predictions,
    "All runs"
)

plt.tight_layout()

plt.savefig(
    "../output/scatter_500.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()