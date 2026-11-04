import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../output/chain_uniprot_sequences.tsv', sep = '\t')

pdb_fasta = [f'../output/FASTAs/{df.chain.to_list()[i]}.fasta' for i in range(len(df.chain.to_list()))]

pdb_sequence = []
pdb_sequence_length = []
for f in pdb_fasta:
    with open(f, 'r') as reader:
        for line in reader:
            if not line.startswith('>'):
                pdb_sequence.append(line.rstrip())
                pdb_sequence_length.append(len(line.rstrip()))

df['pdb_len'] = pdb_sequence_length
df['pdb_sequence'] = pdb_sequence

df['coverage'] = df['pdb_len'] / df['seq_len']

bad = df[df['coverage'].isna() | np.isinf(df['coverage'])]
print(len(bad))
print(bad[['chain', 'protein', 'pdb_len', 'seq_len']])

cov = df['coverage'].replace([np.inf, -np.inf], np.nan)
print(cov.min(), cov.max(), cov.mean()) 

P03437 = 'MKTIIALSYIFCLALGQDLPGNDNSTATLCLGHHAVPNGTLVKTITDDQIEVTNATELVQSSSTGKICNNPHRILDGIDCTLIDALLGDPHCDVFQNETWDLFVERSKAFSNCYPYDVPDYASLRSLVASSGTLEFITEGFTWTGVTQNGGSNACKRGPGSGFFSRLNWLTKSGSTYPVLNVTMPNNDNFDKLYIWGIHHPSTNQEQTSLYVQASGRVTVSTRRSQQTIIPNIGSRPWVRGLSSRISIYWTIVKPGDVLVINSNGNLIAPRGYFKMRTGKSSIMRSDAPIDTCISECITPNGSIPNDKPFQNVNKITYGACPKYVKQNTLKLATGMRNVPEKQTRGLFGAIAGFIENGWEGMIDGWYGFRHQNSEGTGQAADLKSTQAAIDQINGKLNRVIEKTNEKFHQIEKEFSEVEGRIQDLEKYVEDTKIDLWSYNAELLVALENQHTIDLTDSEMNKLFEKTRRQLRENAEEMGNGCFKIYHKCDNACIESIRNGTYDHDVYRDEALNNRFQIKGVELKSGYKDWILWISFAISCFLLCVVLLGFIMWACQRGNIRCNICI'

mask = df['protein'] == 'P03437'
print(mask.sum())                          # quante righe verranno aggiornate

df.loc[mask, 'sequence'] = P03437
df.loc[mask, 'seq_len'] = len(P03437)

df['coverage'] = df['pdb_len'] / df['seq_len']

bad = df[df['coverage'].isna() | np.isinf(df['coverage'])]
print(len(bad))
print(bad[['chain', 'protein', 'pdb_len', 'seq_len']])

cov = df['coverage'].replace([np.inf, -np.inf], np.nan)
print(cov.min(), cov.max(), cov.mean())

new_df = pd.DataFrame({'chain': df.chain,
                       'uniprot': df.uniprot,
                       'protein': df.protein,
                       'protein_length': df.seq_len,
                       'structure_length': df.pdb_len,
                       'coverage': df.pdb_len/df.seq_len,
                       'protein_sequence': df.sequence,
                       'structure_sequence': df.pdb_sequence})

new_df.to_csv('coverage.tsv', sep = '\t')

sdf = pd.DataFrame({'protein': new_df.protein,
                    'coverage': new_df.coverage}).sort_values(by='coverage').reset_index(drop=True)

print(sdf.tail())

weird = new_df[new_df['coverage']>1]
weird = weird.reset_index()
print(weird[['chain', 'uniprot', 'coverage', 'protein_length', 'structure_length']])

print(f'There are {len(list(weird.uniprot.unique()))} uniprot for which sequence length is < structure length')
print('\n'.join(list(weird.uniprot.unique())))
'''
1AO7_D A0A075B6T6
1AO7_D     P01848
1BD2_D     P01848
1BD2_D     P04437
1BD2_E A0A0K0K1A5
1BD2_E     A0A5B9
'''

'''plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria Math'] + plt.rcParams['font.serif']
# Create a larger figure with higher resolution
plt.figure(figsize=(40, 20), dpi=350) # Increased figsize and dpi for better quality

sns.barplot(data=sdf, x='protein', y='coverage', color = 'royalblue')
sns.despine()
plt.title('Protein Coverage Distribution (%)\n', fontsize = 40) # Increased title fontsize
plt.xlabel('\nProtein', fontsize = 30) # Increased x-label fontsize
plt.ylabel('Coverage (%)\n', fontsize = 30) # Increased y-label fontsize
plt.xticks(fontsize = 5, rotation = 90) # Increased x-axis tick label fontsize
plt.yticks(fontsize=25) # Increased y-axis tick label fontsize
#plt.show()
plt.savefig('protein_coverage_distribution.png', format = 'png')'''