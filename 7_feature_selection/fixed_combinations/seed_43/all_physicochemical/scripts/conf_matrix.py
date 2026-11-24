import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# stile come gli scatter
sns.set_theme(style="white")
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

splits = ['ddgit', 'lobso', 'loco', 'stratified']

for split in splits:

    df = pd.read_csv(f'../output/predictions_{split}.tsv', sep='\t')

    true_pos = df.target >= 0
    pred_pos = df.prediction >= 0

    groups = {
        'tp': df.loc[ true_pos &  pred_pos, 'label'].tolist(),
        'fp': df.loc[~true_pos &  pred_pos, 'label'].tolist(),
        'tn': df.loc[~true_pos & ~pred_pos, 'label'].tolist(),
        'fn': df.loc[ true_pos & ~pred_pos, 'label'].tolist(),
    }

    # ---- TSV: riga 1 = lista id, riga 2 = conteggio ----
    out = pd.DataFrame({
        k: [','.join(map(str, v)), len(v)] for k, v in groups.items()
    })
    out.to_csv(f'../output/cf_{split}.tsv', sep='\t', index=False)

    # ---- confusion matrix ----
    n = {k: len(v) for k, v in groups.items()}
    cm = np.array([[n['tp'], n['fn']],
                   [n['fp'], n['tn']]])

    # annotazioni: conteggio + percentuale sul totale
    total = cm.sum()
    annot = np.array([[f"{c}\n({c / total:.1%})" for c in row] for row in cm])

    # MCC
    tp, fp, tn, fn = n['tp'], n['fp'], n['tn'], n['fn']
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    fig, ax = plt.subplots(figsize=(10, 9))

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        square=True,
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 25},
        xticklabels=["ΔΔG ≥ 0", "ΔΔG < 0"],
        yticklabels=["ΔΔG ≥ 0", "ΔΔG < 0"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.set_title(f"Confusion matrix\n({split.upper()} split)\n\nMCC = {mcc:.3f}\n")
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(f'../output/cf_{split}_matrix.png', dpi=600, bbox_inches="tight")
    plt.close()


