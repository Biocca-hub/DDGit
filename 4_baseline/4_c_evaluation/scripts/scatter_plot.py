import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def scatter_plot(df, figname):
    #=======================================================================================
    # PLOT STYLE
    #=======================================================================================

    sns.set_theme(style="whitegrid", palette="viridis")

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

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = sns.scatterplot(
        data=df,
        x="targets",
        y="predictions",
        color="navy",
        s=15,
        edgecolor='black',
        linewidth=0.1,
        alpha=0.1,
        legend=False,
        ax=ax
    )

    min_val = min(
        df['targets'].min(),
        df['predictions'].min()
    )

    max_val = max(
        df['targets'].max(),
        df['predictions'].max()
    )

    pad = 0.05 * (max_val - min_val)

    ax.set_xlim(min_val - pad, max_val + pad)
    ax.set_ylim(min_val - pad, max_val + pad)

    # ======================================
    # RETTA y = x
    # ======================================

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color='blue',
        linestyle='--',
        linewidth=1,
        label='y = x'
    )

    # ======================================
    # x = 0 e y = 0
    # ======================================

    ax.axvline(
        x=0,
        color='blue',
        linestyle='--',
        linewidth=1
    )

    ax.axhline(
        y=0,
        color='blue',
        linestyle='--',
        linewidth=1
    )

    # ======================================
    # MEDIE
    # ======================================

    mean_target = df['targets'].mean()
    mean_prediction = df['predictions'].mean()

    mean_diag = (mean_target + mean_prediction) / 2

    # Media sulla diagonale
    ax.scatter(
        mean_diag,
        mean_diag,
        marker='*',
        s=120,
        color='yellow',
        edgecolor='black',
        linewidth=1.2,
        zorder=10,
        label='Mean on y=x'
    )

    # Media reale
    ax.scatter(
        mean_target,
        mean_prediction,
        marker='*',
        s=120,
        color='red',
        edgecolor='black',
        linewidth=0.5,
        zorder=20,
        label='Mean'
    )

    # ======================================
    # STILE
    # ======================================

    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title("Predictions vs Targets (all runs)")

    ax.legend(frameon=True)

    plt.tight_layout()

    plt.savefig(
        f'{figname}.png',
        dpi=600,
        bbox_inches='tight'
    )
    plt.close()


"""
file = '../../4_b_mlp/output/grid_search/relu_MSE_128_64_32_0.7/scat.txt'

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
    df = pd.DataFrame({'predictions': ps[i], 'targets': ts[i]})
    #figname = f'../output/best_plots/scatter_run_{i+1}'
    #scatter_plot(df, figname)
    all_ps.extend(ps[i])
    all_tgs.extend(ts[i])

df = pd.DataFrame({'predictions': all_ps, 'targets': all_tgs})
scatter_plot(df, f'../output/best_plots/scatter_20_runs')
"""
"""
sns.set_theme(style="whitegrid", palette="viridis")



file = 'layer_sel/three_layers/dims_64_64_64/scat.txt'

with open(file, 'r') as reader:
    ps = []
    ts = []

    for line in reader:
        if line.startswith('Predictions'):
            predictions = line.split()[1:]
            ps.append([float(val[:-1]) for val in predictions])

        elif line.startswith('Targets'):
            targets = line.split()[1:]
            ts.append([float(val[:-1]) for val in targets])

for i in range(len(ps)):

    df = pd.DataFrame({
        'predictions': ps[i],
        'targets': ts[i]
    })

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter con colore viridis
    sns.scatterplot(
        data=df,
        x="targets",
        y="predictions",
        hue="predictions",
        palette="viridis",
        s=25,
        edgecolor="black",
        linewidth=0.4,
        legend=False,
        ax=ax
    )

    # Limiti assi
    min_val = min(df['targets'].min(), df['predictions'].min())
    max_val = max(df['targets'].max(), df['predictions'].max())

    pad = 0.05 * (max_val - min_val)

    ax.set_xlim(min_val - pad, max_val + pad)
    ax.set_ylim(min_val - pad, max_val + pad)

    # Retta ideale y = x
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color='blue',
        linestyle='--',
        linewidth=2,
        label='y = x'
    )

    # x = 0
    ax.axvline(
        x=0,
        color='blue',
        linestyle=':',
        linewidth=1.5
    )

    # y = 0
    ax.axhline(
        y=0,
        color='blue',
        linestyle=':',
        linewidth=1.5
    )

    # Valori medi
    mean_target = df['targets'].mean()
    mean_prediction = df['predictions'].mean()

    # Proiezione del valore medio sulla retta y=x
    mean_diag = (mean_target + mean_prediction) / 2

    ax.scatter(
        mean_diag,
        mean_diag,
        marker='*',
        s=100,
        color='blue',
        edgecolor='blue',
        linewidth=1.2,
        zorder=11,
        label='Mean on y=x'
    )
    
    # Stella sul punto medio
    ax.scatter(
        mean_target,
        mean_prediction,
        marker='*',
        s=350,
        color='red',
        edgecolor='black',
        linewidth=0.5,
        zorder=10,
        label='Mean'
    )
    ax.set_aspect('equal', adjustable='box')

    


    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(f"Scatter plot {i+1}")

    ax.legend(frameon=True)

    plt.tight_layout()

    plt.savefig(
        f'plots/scatter_{i+1}.png',
        dpi=600,
        bbox_inches='tight'
    )

    plt.close()


# ======================================
# STILE
# ======================================

sns.set_theme(style="whitegrid", palette="viridis")

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

# ======================================
# LETTURA FILE
# ======================================

#file = 'layer_sel/three_layers/dims_16_16_16/scat.txt'
#file = 'random_runs/scatter.txt'
#file = 'out_onehot/scat_onehot.txt'
file = '../../scatt_y.txt'

ps = []
ts = []

with open(file, 'r') as reader:

    for line in reader:

        if line.startswith('Predictions'):

            values = line.split(':')[1].split(',')

            ps.extend(
                [float(v.strip()) for v in values if v.strip()]
            )

        elif line.startswith('Targets'):

            values = line.split(':')[1].split(',')

            ts.extend(
                [float(v.strip()) for v in values if v.strip()]
            )

# ======================================
# DATAFRAME UNICO
# ======================================

df = pd.DataFrame({
    'predictions': ps,
    'targets': ts
})

# ======================================
# FIGURA
# ======================================



# ======================================
# LIMITI ASSI
# ======================================


"""