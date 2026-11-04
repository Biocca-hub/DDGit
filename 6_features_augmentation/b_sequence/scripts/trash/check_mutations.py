import pandas as pd

SEQ_FILE = '../output/chain_uniprot_sequences.tsv'
DATA_FILE = '../input/ddgit_dataset.tsv'

seq = pd.read_csv(SEQ_FILE, sep='\t')
data = pd.read_csv(DATA_FILE, sep='\t')

# una catena puo' avere piu' uniprot: tengo tutte le sequenze valide per catena
seq_ok = seq[(seq.uniprot != 'Not Mapped') & seq.sequence.notna()]
chain2seqs = (seq_ok.groupby('chain')
                    .apply(lambda d: list(zip(d.uniprot, d.sequence)))
                    .to_dict())

# catene senza sequenza utilizzabile
no_seq = sorted(set(seq.chain) - set(chain2seqs))


def check(s, pos, wt):
    """None se la posizione e' fuori range, altrimenti True/False."""
    if pos < 1 or pos > len(s):
        return None
    return s[pos - 1] == wt


rows = []
for row in data.itertuples(index=False):
    chain, wt = row.chain, row.wt
    seqs = chain2seqs.get(chain)
    if not seqs:
        rows.append((row.unique, chain, wt, None, None, 'no_sequence'))
        continue

    best = None
    for up, s in seqs:
        r_seq = check(s, int(row.seq_pos), wt)
        r_pdb = check(s, int(row.pdb_pos), wt) if str(row.pdb_pos).isdigit() else None
        if r_seq:
            best = (up, 'seq_pos', 'ok')
            break
        if r_pdb and best is None:
            best = (up, 'pdb_pos', 'ok_pdb_pos')
    if best is None:
        up, s = seqs[0]
        found = s[int(row.seq_pos) - 1] if 1 <= int(row.seq_pos) <= len(s) else 'out_of_range'
        rows.append((row.unique, chain, wt, up, found, 'mismatch'))
    else:
        rows.append((row.unique, chain, wt, best[0], wt, best[2]))

res = pd.DataFrame(rows, columns=['unique', 'chain', 'wt', 'uniprot', 'found', 'status'])

print(res.status.value_counts(), '\n')

print('catene senza sequenza:', len(no_seq))
print(no_seq[:20], '\n')

bad = res[res.status == 'mismatch']
if len(bad):
    print('mismatch per catena:')
    print(bad.groupby(['chain', 'uniprot']).size().sort_values(ascending=False).head(30))
    print()
    print(bad.head(30).to_string(index=False))

res.to_csv('../output/mutation_check.tsv', sep='\t', index=False)