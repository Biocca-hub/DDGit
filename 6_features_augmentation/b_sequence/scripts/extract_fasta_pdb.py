import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

dataset = pd.read_csv('../input/ddgit_dataset.tsv', sep = '\t')

chains = list(dataset.chain.unique())

'ATOM      2  CA  SER A   1      13.781   5.819  -3.336  1.00 45.16           C'  

# linea inizia con ATOM
# caratteri 13 e 14 = CA
# catena carattere 21
# prendi caratteri 17:20 = aa 3 lettere 

aa_3_to_1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
             "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
             "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
             "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
             # ambigui / non standard
             "GLX": "Z", "ASX": "B", "UNK": "X",
             "MSE": "M", "SEC": "U", "PYL": "O"}

chain2seq = {}
for c in chains:
    pdb_id, chain = c.split('_')
    file = f'../input/PDBs/{pdb_id}.pdb'
    sequence = ''
    with open(file, 'r') as fh:
        for line in fh:
            if line.startswith('ENDMDL'):        # NMR: tieni solo il primo modello
                break
            if (line.startswith('ATOM')
                    and line[12:16].strip() == 'CA'
                    and line[21] == chain
                    and line[16] in (' ', 'A')):  # evita duplicati da altloc
                resname = line[17:20]
                if resname not in aa_3_to_1:
                    print(f'{c}: residuo sconosciuto {resname}')
                sequence += aa_3_to_1.get(resname, 'X')
    chain2seq[c] = sequence

cov = pd.read_csv('wrong_coverage.tsv', sep = '\t')
df = cov.copy()
del df['Unnamed: 0']
print(df.columns)

s_seq = []
s_len = []
for c in cov.chain:
    s_seq.append(chain2seq[c])
    s_len.append(len(chain2seq[c]))

cov['structure_length'] = s_len
cov['structure_sequence'] = s_seq
cov['coverage'] = cov.structure_length/cov.protein_length

# cov.to_csv('../output/coverage.tsv', sep = '\t', index = False)

n_unique = cov.groupby(['chain', 'uniprot'])['coverage'].nunique().reset_index(name='n_coverage')
# print(n_unique[n_unique['n_coverage'] > 1])

# print(n_unique)

sdf = cov[cov['coverage'] <= 1]
sdf['label'] = sdf['chain'] + '|' + sdf['uniprot']
sdf = pd.DataFrame({'protein': sdf.label,
                    'coverage': sdf.coverage}).sort_values(by='coverage').reset_index(drop=True)
                
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria Math'] + plt.rcParams['font.serif']
# Create a larger figure with higher resolution
plt.figure(figsize=(40, 20), dpi=350) # Increased figsize and dpi for better quality

sns.barplot(data=sdf, x='protein', y='coverage', color = 'royalblue')
sns.despine()
plt.title('Protein Coverage Distribution\n', fontsize = 70) # Increased title fontsize
plt.xlabel('\nProtein', fontsize = 50) # Increased x-label fontsize
plt.ylabel('Coverage\n', fontsize = 50) # Increased y-label fontsize
plt.xticks(fontsize = 5, rotation = 90) # Increased x-axis tick label fontsize
plt.yticks(fontsize=40) # Increased y-axis tick label fontsize
#plt.show()
plt.savefig('protein_coverage_distribution.png', format = 'png')

sdf = cov[cov['coverage'] > 1]
sdf['label'] = sdf['chain'] + '|' + sdf['uniprot']
sdf = pd.DataFrame({'protein': sdf.label,
                    'coverage': sdf.coverage}).sort_values(by='coverage').reset_index(drop=True)
                
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria Math'] + plt.rcParams['font.serif']
# Create a larger figure with higher resolution
plt.figure(figsize=(40, 40), dpi=350) # Increased figsize and dpi for better quality

sns.barplot(data=sdf, x='protein', y='coverage', color = 'royalblue')
sns.despine()
plt.title('Protein Coverage Out of Range\n', fontsize = 70) # Increased title fontsize
plt.xlabel('\nProtein', fontsize = 50) # Increased x-label fontsize
plt.ylabel('Coverage\n', fontsize = 50) # Increased y-label fontsize
plt.xticks(fontsize = 25, rotation = 90) # Increased x-axis tick label fontsize
plt.yticks(fontsize=40) # Increased y-axis tick label fontsize
#plt.show()
plt.savefig('protein_coverage_out_of_range.png', format = 'png')

'''
aa_3_to_1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
             "GLN": "Q","GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", 
             "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
             "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

chain2seq = {}
for c in chains:
    pdb = c.split('_')[0]
    chain = c.split('_')[1]
    file = f'../input/PDBs/{pdb}.pdb'
    sequence = ''
    with open(file, 'r') as pdb:
        for line in pdb:
            if line.startswith('ATOM') and line[13:15]=="CA" and line[21]==chain:
                sequence += aa_3_to_1[line[17:20]]
    chain2seq[c] = sequence
'''

'''n_unique = cov.groupby('chain')['coverage'].nunique()

print(n_unique.value_counts())          # quante proteine hanno 1, 2, 3... valori diversi
print(n_unique[n_unique > 1])           # le proteine con coverage NON costante
print((n_unique == 1).all())            # True se è costante per tutte

stats = cov.groupby('chain')['coverage'].agg(['min', 'max', 'nunique'])
stats['range'] = stats['max'] - stats['min']
print(stats[stats['nunique'] > 1].sort_values('range', ascending=False))
'''