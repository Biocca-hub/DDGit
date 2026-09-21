import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

for i in range(1,6):
    df = pd.read_csv(f'../output/training/alanine/to_alanine_{i}.tsv', sep = '\t')

    # Crea 20 bin
    df['DDG_avg_binned'] = pd.cut(df['DDG_avg'], bins=40)
    counts = df['DDG_avg_binned'].value_counts(sort=False)


    # Conta quante osservazioni ci sono per ogni mut e ogni bin
    hm = pd.crosstab(
        df['wt'],
        df['DDG_avg_binned']
    )

    sns.set_theme(style="whitegrid", 
                palette="twilight")

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

    # Heatmap
    plt.figure(figsize=(14, 8))

    sns.heatmap(
        hm,
        annot=True,
        fmt='d',
        cmap='twilight',
        linewidth = 0.5,
        cbar=True
    )

    plt.xlabel('\nDDG bins')
    plt.ylabel('Wild Type\n')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'ddg_ala_{i}.png', dpi=300)
    plt.close()
