import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt

aa_list = list("GAVPLIMFWYSTCNQHDEKR")             # wild type (righe): 20
mut_list = [aa for aa in aa_list if aa != 'A']     # mutanti (colonne): 19, senza alanina

df = pd.read_csv('../output/normalized_aa_features.tsv', sep='\t')

# escludo le mutazioni X -> A
df = df[df['mut'] != 'A']

# colonne delta -> (titolo, unità, formato e dimensione delle etichette)
deltas = {
    'delta_mw':          ('Molecular weight',  'Da',              '.0f', 24),
    'delta_v':           ('Volume',            'a.u.',            '.1f', 30),
    'delta_hidropathy':  ('Hydropathy (KD)',   'a.u.',            '.1f', 30),
    'delta_iso_p':       ('Isoelectric point', 'pH units',        '.1f', 30),
    'delta_aromaticity': ('Aromaticity',       'a.u.',            '.0f', 30),
}

# Frequenza di ogni mutazione wt -> mut (colore delle celle)
counts = pd.crosstab(df['wt'], df['mut'])
counts = counts.reindex(index=aa_list, columns=mut_list, fill_value=0)

# celle senza mutazioni (diagonale inclusa) lasciate vuote
mask = counts == 0


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
})


for col, (name, unit, fmt, annot_size) in deltas.items():

    # Valore delta per ogni coppia wt -> mut (etichette delle celle)
    delta = (df.pivot_table(index='wt', columns='mut', values=col, aggfunc='mean')
               .reindex(index=aa_list, columns=mut_list))

    labels = delta.map(lambda v: '' if pd.isna(v) else format(v, fmt))
    labels = labels.replace({'-0': '0', '-0.0': '0.0'})   # evita "-0"

    plt.figure(figsize=(40, 25))

    ax = sns.heatmap(
        counts,
        mask=mask,
        square=True,
        cmap='Blues',
        annot=labels.values,
        fmt='',
        annot_kws={"fontsize": annot_size},
        linewidths=0.5,
        linecolor='lightblue',
        xticklabels=mut_list,
        yticklabels=aa_list,
        cbar=True,
        cbar_kws={
            'label': 'Number of mutations',
            'shrink': 0.8
        }
    )

    # G in basso -> R in alto
    ax.invert_yaxis()

    # Colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('\nNumber of mutations\n', fontsize=40)
    cbar.ax.tick_params(labelsize=30)

    # Assi
    ax.tick_params(axis='both', labelsize=25)

    ax.set_xlabel("\nMutant\n", fontsize=50)
    ax.set_ylabel("\nWild type\n", fontsize=50)

    ax.set_title(
        f"\nΔ {name} (mut − wt) [{unit}]\n",
        fontsize=70
    )

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(
        f'heatmap_{col}_no_ala.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()