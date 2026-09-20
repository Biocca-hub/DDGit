import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ala = [f'../output/training/to_alanine_{i}.tsv' for i in range(1,6)]
one = pd.read_csv(ala[2], sep = '\t')

# Crea 20 bin
one['DDG_avg_binned'] = pd.cut(one['DDG_avg'], bins=40)
counts = one['DDG_avg_binned'].value_counts(sort=False)


# Conta quante osservazioni ci sono per ogni mut e ogni bin
hm = pd.crosstab(
    one['wt'],
    one['DDG_avg_binned']
)

# Heatmap
plt.figure(figsize=(14, 8))

sns.heatmap(
    hm,
    annot=True,
    fmt='d',
    cmap='viridis',
    cbar=True
)

plt.xlabel('DDG_avg')
plt.ylabel('wt')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('ddghm.png', dpi=300)
plt.close()
