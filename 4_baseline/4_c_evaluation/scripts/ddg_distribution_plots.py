import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv('../../4_a_features/input/folds/data/fold_1.tsv', sep = '\t')
df2 = pd.read_csv('../../4_a_features/input/folds/data/fold_2.tsv', sep = '\t')
df3 = pd.read_csv('../../4_a_features/input/folds/data/fold_3.tsv', sep = '\t')
df4 = pd.read_csv('../../4_a_features/input/folds/data/fold_4.tsv', sep = '\t')
df5 = pd.read_csv('../../4_a_features/input/folds/data/fold_5.tsv', sep = '\t')

dfs = [df1, df2, df3, df4, df5]

f = []
for i in range(5):
    f.extend([i]*dfs[i].shape[0])

df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

df['fold'] = f
"""
plt.figure(figsize=(30, 20), dpi=350) # Increased figsize and dpi for better quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria Math'] + plt.rcParams['font.serif']
ax = sns.violinplot(data=df, y='DDG_avg', palette = 'Blues', hue = 'fold', legend =True, fill = True)
sns.despine()
plt.title('DDG: Fold', fontsize = 50)
plt.ylabel('DDG [kcal/mol]', fontsize = 40)
plt.xlabel('Folds', fontsize = 40)
plt.xticks(fontsize=30) # Increased x-axis tick label fontsize
plt.yticks(fontsize=30) # Increased y-axis tick label fontsize
legend = ax.get_legend()
if legend:
    legend.set_title('Fold', prop={'size': 40}) # Set title font size
    for text in legend.get_texts(): # Set item font sizes
        text.set_fontsize(40)
#plt.show()
plt.savefig('Fold_DDG_violin.png', format = 'png')

#df = df[df['Clean_mut'].str()[-1] == 'A']
df = df[df['Clean_mut'].str[-1] == 'A']

plt.figure(figsize=(30, 20), dpi=350) # Increased figsize and dpi for better quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria Math'] + plt.rcParams['font.serif']
ax = sns.violinplot(data=df, y='DDG_avg', palette = 'Blues', hue = 'fold', legend =True, fill = True)
sns.despine()
plt.title('DDG: Fold', fontsize = 50)
plt.ylabel('DDG [kcal/mol]', fontsize = 40)
plt.xlabel('Folds', fontsize = 40)
plt.xticks(fontsize=30) # Increased x-axis tick label fontsize
plt.yticks(fontsize=30) # Increased y-axis tick label fontsize
legend = ax.get_legend()
if legend:
    legend.set_title('Fold', prop={'size': 40}) # Set title font size
    for text in legend.get_texts(): # Set item font sizes
        text.set_fontsize(40)
#plt.show()
plt.savefig('Ala_DDG_violin.png', format = 'png')"""

# Same y-axis limits for both plots
ymin = df['DDG_avg'].min()-5
ymax = df['DDG_avg'].max()+5

# Alanine mutations only
df_ala = df[df['Clean_mut'].str.endswith('A', na=False)].copy()

# Style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria Math'] + plt.rcParams['font.serif']

# Two plots side by side
fig, axes = plt.subplots(
    1, 2,
    figsize=(30, 15),
    dpi=350,
    sharey=True
)

# -------------------------
# LEFT: complete dataset
# -------------------------
sns.violinplot(
    data=df,
    x='fold',
    y='DDG_avg',
    palette='Blues',
    hue='fold',
    legend=False,
    fill=True,
    ax=axes[0]
)

axes[0].set_title('All mutations', fontsize=45)
axes[0].set_xlabel('Fold', fontsize=35)
axes[0].set_ylabel('DDG [kcal/mol]', fontsize=35)

axes[0].tick_params(axis='both', labelsize=28)

# -------------------------
# RIGHT: Alanine only
# -------------------------
sns.violinplot(
    data=df_ala,
    x='fold',
    y='DDG_avg',
    palette='Blues',
    hue='fold',
    legend=False,
    fill=True,
    ax=axes[1]
)

axes[1].set_title('Alanine mutations', fontsize=45)
axes[1].set_xlabel('Fold', fontsize=35)
axes[1].set_ylabel('')

axes[1].tick_params(axis='both', labelsize=28)

# Same y limits from COMPLETE df
axes[0].set_ylim(ymin, ymax)
axes[1].set_ylim(ymin, ymax)

sns.despine()

plt.tight_layout()

plt.savefig(
    'Fold_DDG_violin_comparison.png',
    format='png',
    bbox_inches='tight'
)

plt.close()
