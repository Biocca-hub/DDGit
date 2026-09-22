import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

aa_list = list("GAVPLIMFWYSTCNQHDEKR")
#print(len(aa_list))

mut = {}
h = np.zeros((20,20))

for i in range(1,6):
    df = pd.read_csv(f'../output/training/fold_{i}.tsv', sep = '\t')
    del df['Unnamed: 0']
    mut_fold = {}
    heat = np.zeros((20,20))
    for m in df['Clean_mut'].to_list():
        inout = f'{m[0]}{m[-1]}'
        mut[inout] = mut.get(inout, 0) + 1
        mut_fold[inout] = mut_fold.get(inout, 0) + 1
        k = aa_list.index(inout[0])
        j = aa_list.index(inout[1])
        heat[k][j]=heat[k][j]+1
    
    # print(heat)
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        })

    plt.figure(figsize=(25, 25))

    ax = sns.heatmap(
            heat,
            square=True,
            cmap='Blues',
            annot = True,
            fmt='.0f',
            annot_kws={"fontsize": 30},
            linewidth = 0.5,
            linecolor='lightblue',
            xticklabels=aa_list,
            yticklabels=aa_list,
            cbar=True,
            cbar_kws={'label': 'Number of mutations',
            'shrink': 0.8}
        )

    ax.invert_yaxis()

    # Colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('\nNumber of mutations\n', fontsize=40)
    cbar.ax.tick_params(labelsize=30)

    ax.tick_params(axis='both', labelsize=40)
    ax.set_xlabel("\nMutated\n", fontsize=50)
    ax.set_ylabel("\nWild type\n", fontsize=50)
    ax.set_title(f"\nMutation matrix (fold {i})\n", fontsize=70)
    plt.tight_layout()
    plt.savefig(f'hm_{i}.png', dpi=300, bbox_inches='tight')
    plt.close()    
        
    h = h + heat

plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        })

plt.figure(figsize=(25, 25))
ax = sns.heatmap(
            h,
            square=True,
            cmap='Blues',
            annot = True,
            fmt='.0f',
            annot_kws={"fontsize": 30},
            linewidth = 0.5,
            linecolor = 'lightblue',
            xticklabels=aa_list,
            yticklabels=aa_list,
            cbar=True,
            cbar_kws={'label': 'Number of mutations',
            'shrink': 0.8}
        )

ax.invert_yaxis()

# Colorbar
cbar = ax.collections[0].colorbar
cbar.set_label('\nNumber of mutations\n', fontsize=40)
cbar.ax.tick_params(labelsize=30)

ax.tick_params(axis='both', labelsize=40)
ax.set_xlabel("\nMutated\n", fontsize=50)
ax.set_ylabel("\nWild type\n", fontsize=50)
ax.set_title("\nMutation matrix (Training set)\n", fontsize=70)


plt.tight_layout()
plt.savefig(f'hm.png', dpi=300, bbox_inches='tight')
plt.close()
