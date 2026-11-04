import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../../../../meeting/data_meta/final_data.tsv', sep = '\t')

chain2uniprot = {}
for chain, up in zip(df['Chain'], df['Uniprot_ID']):
    ids = [u.strip() for u in str(up).split(',') if u.strip()]
    chain2uniprot.setdefault(chain, set()).update(ids)

chain2uniprot = {c: sorted(ids) for c, ids in chain2uniprot.items()}

print(len(chain2uniprot))                                                    # 444
print(sum(v == ['Not Mapped'] for v in chain2uniprot.values()))              # 63
print({c: u for c, u in chain2uniprot.items() if len(u) > 1})                # le 3 catene di TCR


sequences = []
lengths = []
protein = []
with open('../output/complete.fasta', 'r') as reader:
    for line in reader:
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            protein.append(line[1:])      # 'A0A075B6T6' oppure '1BJ1_H'
        else:
            sequences.append(line)
            lengths.append(int(len(line)))

sdf = pd.DataFrame({'protein': protein,
                    'seq_len': lengths,
                    'sequence': sequences})

rows = []
for chain, ups in chain2uniprot.items():
    for up in ups:
        rows.append({'chain': chain,
                     'uniprot': up,
                     'protein': chain if up == 'Not Mapped' else up})

cdf = pd.DataFrame(rows)

final = cdf.merge(sdf, on='protein', how='left')
final = final[['chain', 'uniprot', 'protein', 'seq_len', 'sequence']]

final.to_csv('../output/chain_uniprot_sequences.tsv', sep='\t', index=False)

df_sorted = sdf.sort_values(by='seq_len').reset_index(drop=True)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria Math'] + plt.rcParams['font.serif']
# Create a larger figure with higher resolution
plt.figure(figsize=(40, 20), dpi=350) # Increased figsize and dpi for better quality

sns.barplot(data=df_sorted, x='protein', y='seq_len', color = 'royalblue')
sns.despine()
plt.title('Sequence Length Distribution\n', fontsize = 40) # Increased title fontsize
plt.xlabel('\nProtein', fontsize = 30) # Increased x-label fontsize
plt.ylabel('Length\n', fontsize = 30) # Increased y-label fontsize
plt.xticks(fontsize = 5, rotation = 90) # Increased x-axis tick label fontsize
plt.yticks(fontsize=25) # Increased y-axis tick label fontsize
#plt.show()
plt.savefig('protein_length_distribution.png', format = 'png')




