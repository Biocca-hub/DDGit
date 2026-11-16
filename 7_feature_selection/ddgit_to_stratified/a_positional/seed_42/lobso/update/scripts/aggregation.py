import pandas as pd
import numpy as np
import torch
import seaborn as sns 
import matplotlib.pyplot as plt

def pearson_corr(y_hat, y): 
    vx = y_hat - torch.mean(y_hat) # predictions
    vy = y - torch.mean(y) # targets
    return torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-8) 

file = 'pred_files.txt'
files = []
with open(file, 'r') as reader:
    for line in reader:
        files.append(f'../input/pred_lobso/{line.strip()}')

dfs = [pd.read_csv(f, sep='\t') for f in files]
df = pd.concat(dfs, ignore_index=True)
df.to_csv('predictions.tsv', sep = '\t', index = False)

predictions = torch.tensor(df.prediction)
targets = torch.tensor(df.target)

print(float(pearson_corr(predictions, targets)))

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

pred = pd.read_csv('predictions.tsv', sep='\t')

fig, ax = plt.subplots(figsize=(8, 8))
single_scatter(ax, pred, "Nested CV (5 fold)")
plt.tight_layout()
plt.savefig('scatter_nested.png', dpi=300)
plt.show()