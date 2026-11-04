import pandas as pd
from collections import defaultdict

SEQ_FILE = '../output/chain_uniprot_sequences.tsv'
DATA_FILE = '../input/ddgit_dataset.tsv'

seq = pd.read_csv(SEQ_FILE, sep='\t')
data = pd.read_csv(DATA_FILE, sep='\t')

seq_ok = seq[(seq.uniprot != 'Not Mapped') & seq.sequence.notna()]
chain2seqs = defaultdict(list)
for r in seq_ok.itertuples(index=False):
    chain2seqs[r.chain].append((r.uniprot, r.sequence))

# mutazioni raggruppate per catena: (seq_pos, wt)
muts = defaultdict(list)
for r in data.itertuples(index=False):
    muts[r.chain].append((int(r.seq_pos), r.wt))


def score(s, offset, mlist):
    """quante mutazioni combaciano con questo offset"""
    n = 0
    for pos, wt in mlist:
        i = pos - 1 + offset
        if 0 <= i < len(s) and s[i] == wt:
            n += 1
    return n


results = []
for chain, mlist in muts.items():
    seqs = chain2seqs.get(chain)
    if not seqs:
        results.append((chain, None, None, 0, len(mlist)))
        continue

    maxpos = max(p for p, _ in mlist)
    best = (-1, None, None)  # (n_match, offset, uniprot)
    for up, s in seqs:
        for off in range(0, len(s) - maxpos + 1):
            n = score(s, off, mlist)
            if n > best[0]:
                best = (n, off, up)
            if n == len(mlist):
                break
        if best[0] == len(mlist):
            break

    n, off, up = best
    results.append((chain, up, off, n, len(mlist)))

res = pd.DataFrame(results, columns=['chain', 'uniprot', 'offset', 'n_match', 'n_mut'])
res['perfect'] = res.n_match == res.n_mut

print(f"catene totali:          {len(res)}")
print(f"catene allineate al 100%: {res.perfect.sum()}")
print(f"mutazioni allineate:      {res.n_match.sum()} / {res.n_mut.sum()}\n")

print('distribuzione degli offset (catene allineate):')
print(res[res.perfect].offset.value_counts().sort_index().head(20), '\n')

bad = res[~res.perfect].sort_values('n_mut', ascending=False)
if len(bad):
    print('catene NON allineate:')
    print(bad.to_string(index=False))

res.to_csv('../output/chain_offsets.tsv', sep='\t', index=False)
