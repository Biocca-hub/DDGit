import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# stile
# ------------------------------------------------------------------
sns.set_theme(style="whitegrid",
              rc={
                  "grid.color": "lightblue",
                  "grid.linestyle": ":",
                  "grid.linewidth": 0.5,
                  "axes.grid": True
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


# ------------------------------------------------------------------
# confusion matrix
# ------------------------------------------------------------------
def single_confusion(ax, df, title):

    true_pos = df.target >= 0
    pred_pos = df.prediction >= 0

    tp = int(( true_pos &  pred_pos).sum())
    fp = int((~true_pos &  pred_pos).sum())
    tn = int((~true_pos & ~pred_pos).sum())
    fn = int(( true_pos & ~pred_pos).sum())

    cm = np.array([[tp, fn],
                   [fp, tn]])

    total = cm.sum()
    annot = np.array([[f"{c}\n({c / total:.1%})" for c in row] for row in cm])

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        square=True,
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 25},
        xticklabels=["ΔΔG ≥ 0", "ΔΔG < 0"],
        yticklabels=["ΔΔG ≥ 0", "ΔΔG < 0"],
        ax=ax
    )

    ax.grid(False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.tick_params(axis="y", rotation=0)
    ax.set_title(f"{title}\nMCC = {mcc:.3f}\n")

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "mcc": mcc}


# ------------------------------------------------------------------
# scatter
# ------------------------------------------------------------------
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
        ax=ax)

    min_val = -12.352
    max_val = 12.352
    pad = 0.05 * (max_val - min_val)

    ax.set_xlim(min_val - pad, max_val + pad)
    ax.set_ylim(min_val - pad, max_val + pad)

    # y = x
    ax.plot([min_val - pad, max_val + pad],
            [min_val - pad, max_val + pad],
            color="orangered", linestyle="--", linewidth=1.5)

    # x = 0 e y = 0
    ax.axvline(0, color="orangered", linestyle=":", linewidth=1)
    ax.axhline(0, color="orangered", linestyle=":", linewidth=1)

    # punto medio (media target, media predizione)
    ax.scatter(df["target"].mean(),
               df["prediction"].mean(),
               marker="o", s=20, color="orangered", linewidth=1, zorder=10)

    pcc = np.corrcoef(df.target, df.prediction)[0, 1]

    ax.text(0.05, 0.95, f"PCC = {pcc:.3f}",
            transform=ax.transAxes,
            fontsize=25,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey"))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(title)

    return pcc


# ------------------------------------------------------------------
# figura: riga 1 = confusion matrix, riga 2 = scatter
# ------------------------------------------------------------------
splits = ["ddgit", "lobso", "loco", "stratified"]

tests = ['', '_S8338', '_rev']

fig, axes = plt.subplots(2, 4, figsize=(40, 20))

for t in tests:
    for col, split in enumerate(splits):

        df = pd.read_csv(f"../output/predictions{t}_{split}.tsv", sep="\t")

        single_confusion(axes[0, col], df, f"{split.upper()} split")
        single_scatter(axes[1, col], df, f"{split.upper()} split\n")

    fig.suptitle("Classification (ΔΔG sign) and regression performance per split\n",
                fontsize=30)

    plt.tight_layout()
    plt.savefig(f"../output/cm_scatter{t}_all_splits.png", dpi=600, bbox_inches="tight")
    plt.close()