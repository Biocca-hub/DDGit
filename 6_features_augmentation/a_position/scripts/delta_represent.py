import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt

aa_list = list("GAVPLIMFWYSTCNQHDEKR")

df = pd.read_csv('../output/normalized_aa_features.tsv', sep='\t')

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
counts = counts.reindex(index=aa_list, columns=aa_list, fill_value=0)

# celle senza mutazioni (diagonale inclusa) lasciate vuote
mask = counts == 0


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
})


for col, (name, unit, fmt, annot_size) in deltas.items():

    # Valore delta per ogni coppia wt -> mut (etichette delle celle)
    # il delta dipende solo dalla coppia, quindi la media dei valori ripetuti è il valore stesso
    delta = (df.pivot_table(index='wt', columns='mut', values=col, aggfunc='mean')
               .reindex(index=aa_list, columns=aa_list))

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
        xticklabels=aa_list,
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

    ''' Peso molecolare: Da (equivalente a g/mol), è quello che restituisce Biopython.
        Punto isoelettrico: è un valore di pH, quindi “pH units”.
        Idropatia Kyte-Doolittle: è una scala relativa senza dimensione, quindi “a.u.” (arbitrary units) o puoi togliere le parentesi.
        Volume: i valori di quantiprot (G = 0, A = 1, F = 5.89…) sono un volume di van der Waals normalizzato dall’AAindex, non Å³, quindi anche qui “a.u.”. Se ti servisse in Å³ dovresti usare un’altra scala (es. Zamyatnin).
        Aromaticità: è una frazione (0 o 1 per un singolo aa), quindi adimensionale.
    '''

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(
        f'heatmap_{col}.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()