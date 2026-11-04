import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../input/ddgit_dataset.tsv', sep = '\t')
# print(df.columns)
'''
['unique', 'complex', 'pdb', 'chain', 'chain_id', 'wt', 'mut',
'position', 'location', 'ddg', 'ddgit_cc', 'ddgit_fold', 'lobso_cc',
'loco_cc', 'stratified_cc', 'inner_1', 'inner_2', 'inner_3', 'inner_4',
'inner_5', 'inner_6', 'inner_7', 'inner_8', 'inner_9', 'inner_10']
'''

chains = [f'{c}.fasta' for c in list(df.chain.unique())]

'''file = '../output/seq_list.txt'
count = 0
notin = 0
with open(file, 'r') as reader:
    for line in reader:
        c = line.strip()
        if c in chains:
            count += 1
        else:
            notin += 1'''

# print(count == len(chains))
sequences = []
lengths = []
for c in chains:
    with open(f'../output/FASTAs/{c}', 'r') as reader:
        for line in reader:
            if not line.startswith('>'):
                sequences.append(line.strip())
                lengths.append(len(line.strip()))

sdf = pd.DataFrame({'chain': [c[:-6] for c in chains],
                    'seq_len': lengths,
                    'sequence': sequences})

df_sorted = sdf.sort_values(by='seq_len').reset_index(drop=True)

all_seq = []
all_l = []
all_c = [f'{c}.fasta' for c in list(df.chain)]
for c in all_c:
    with open(f'../output/FASTAs/{c}', 'r') as reader:
        for line in reader:
            if not line.startswith('>'):
                all_seq.append(line.strip())
                all_l.append(len(line.strip()))
sdf = pd.DataFrame({'chain': df.chain,
                    'seq_len': all_l,
                    'sequence': all_seq,
                    'pos': df.position,
                    'wt': df.wt,
                    'mut': df.mut})

ok = 0
out_of_range = []
mismatch = []

for i, row in sdf.iterrows():
    seq, pos, wt = row['sequence'], row['pos'], row['wt']
    if pos < 1 or pos > len(seq):
        out_of_range.append(i)
        print(f'[FUORI RANGE] riga {i} | chain {row["chain"]} | pos {pos} | len seq {len(seq)} | wt {wt}')
    elif seq[pos - 1] != wt:
        mismatch.append(i)
        print(f'[MISMATCH]    riga {i} | chain {row["chain"]} | pos {pos} | atteso {wt} | trovato {seq[pos - 1]}')
    else:
        ok += 1

print(f'\nOK: {ok} / {len(sdf)}')
print(f'Fuori range: {len(out_of_range)}  -> chain: {sdf.loc[out_of_range, "chain"].unique()}')
print(f'Mismatch: {len(mismatch)}  -> chain: {sdf.loc[mismatch, "chain"].unique()}')
