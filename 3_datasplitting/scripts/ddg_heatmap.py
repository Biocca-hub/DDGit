import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


aa_list = list("GAVPLIMFWYSTCNQHDEKR")

dfs = []

for i in range(1, 6):
    df = pd.read_csv(f'../output/training/alanine/to_alanine_{i}.tsv', sep='\t')
    dfs.append(df)


# Overall DDG range
ddg_min = min(df['DDG_avg'].min() for df in dfs)
ddg_max = max(df['DDG_avg'].max() for df in dfs)

# make bins from DDG range
bins = np.linspace(ddg_min, ddg_max, 41)


# Initialize matrix 
hm_total = None



# Fold specific heat maps

for i, df in enumerate(dfs, start=1):

    df['DDG_avg_binned'] = pd.cut(df['DDG_avg'],bins=bins,include_lowest=True)

    # Counting observations WT x DDG bin
    hm = pd.crosstab(df['wt'],df['DDG_avg_binned']
    )

    # Including all 20 aa
    hm = hm.reindex(index=aa_list, columns=df['DDG_avg_binned'].cat.categories, fill_value=0)

    # Filling complete matrix
    if hm_total is None:
        hm_total = hm.copy()
    else:
        hm_total = hm_total + hm

    # HEATMAPS

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    })

    plt.figure(figsize=(40, 25))

    ax = sns.heatmap(
        hm,
        square=True,
        cmap='Blues',
        annot=True,
        fmt='.0f',
        annot_kws={"fontsize": 30},
        linewidths=0.5,
        linecolor='lightblue',
        xticklabels=True,
        yticklabels=aa_list,
        cbar=True,
        cbar_kws={
            'label': 'Number of mutations',
            'shrink': 0.8
        }
    )

    ax.invert_yaxis()

    # Colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('\nNumber of mutations\n', fontsize=40)
    cbar.ax.tick_params(labelsize=30)

    # Assi
    ax.tick_params(axis='both', labelsize=25)

    ax.set_xlabel("\nDDG bins\n", fontsize=50)
    ax.set_ylabel("\nWild type\n", fontsize=50)

    ax.set_title(
        f"\nDDG mutation distribution (fold {i})\n",
        fontsize=70
    )

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(
        f'ddg_ala_{i}.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


# ============================================================
# Heatmap complessiva dei 5 fold
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
})

plt.figure(figsize=(40, 25))

ax = sns.heatmap(
    hm_total,
    square=True,
    cmap='Blues',
    annot=True,
    fmt='.0f',
    annot_kws={"fontsize": 30},
    linewidths=0.5,
    linecolor='lightblue',
    xticklabels=True,
    yticklabels=aa_list,
    cbar=True,
    cbar_kws={
        'label': 'Number of mutations',
        'shrink': 0.8
    }
)

ax.invert_yaxis()

# Colorbar
cbar = ax.collections[0].colorbar
cbar.set_label('\nNumber of mutations\n', fontsize=40)
cbar.ax.tick_params(labelsize=30)

ax.tick_params(axis='both', labelsize=25)

ax.set_xlabel("\nDDG bins\n", fontsize=50)
ax.set_ylabel("\nWild type\n", fontsize=50)

ax.set_title(
    "\nDDG mutation distribution (Training set)\n",
    fontsize=70
)

plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(
    'ddg_ala.png',
    dpi=300,
    bbox_inches='tight'
)
plt.close()
