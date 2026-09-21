import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ala, Arg, Asn, Asp, Cys, Glu, Gln, Gly, His, Ile, Leu, Lys, Met, Phe, Pro, Ser, Thr, Trp, Tyr, 

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
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13
    })

    plt.figure(figsize=(20, 20))

    ax = sns.heatmap(
            heat,
            square=True,
            cmap='twilight',
            annot = True,
            fmt='.0f',
            annot_kws={"fontsize": 20},
            linewidth = 0.5,
            xticklabels=aa_list,
            yticklabels=aa_list
        )
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel("\nMutated", fontsize=25)
    ax.set_ylabel("Wild type\n", fontsize=25)
    ax.set_title(f"Mutation matrix (fold {i})\n", fontsize=25)
    plt.tight_layout()
    plt.savefig(f'hm_{i}.png', dpi=300, bbox_inches='tight')
    plt.close()    
        
    h = h + heat

plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13
    })

plt.figure(figsize=(20, 20))
ax = sns.heatmap(
            h,
            square=True,
            cmap='twilight',
            annot = True,
            fmt='.0f',
            annot_kws={"fontsize": 20},
            linewidth = 0.5,
            xticklabels=aa_list,
            yticklabels=aa_list
        )
ax.tick_params(axis='both', labelsize=20)
ax.set_xlabel("\nMutated", fontsize=25)
ax.set_ylabel("Wild type\n", fontsize=25)
ax.set_title("Mutation matrix (Training set)\n", fontsize=25)


plt.tight_layout()
plt.savefig(f'hm.png', dpi=300, bbox_inches='tight')
plt.close()
