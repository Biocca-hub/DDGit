import pandas as pd

seq = pd.read_csv('../output/chain_uniprot_sequences.tsv', sep='\t')
data = pd.read_csv('../input/ddgit_dataset.tsv', sep='\t')
seq = seq[seq.uniprot != 'Not Mapped']

# print(seq[seq.sequence.isna()])
''''
ha = """
MKTIIALSYIFCLALGQDLPGNDNSTATLCLGHHAVPNGTLVKTITDDQIEVTNATELVQ
SSSTGKICNNPHRILDGIDCTLIDALLGDPHCDVFQNETWDLFVERSKAFSNCYPYDVPD
YASLRSLVASSGTLEFITEGFTWTGVTQNGGSNACKRGPGSGFFSRLNWLTKSGSTYPVL
NVTMPNNDNFDKLYIWGIHHPSTNQEQTSLYVQASGRVTVSTRRSQQTIIPNIGSRPWVR
GLSSRISIYWTIVKPGDVLVINSNGNLIAPRGYFKMRTGKSSIMRSDAPIDTCISECITP
NGSIPNDKPFQNVNKITYGACPKYVKQNTLKLATGMRNVPEKQTRGLFGAIAGFIENGWE
GMIDGWYGFRHQNSEGTGQAADLKSTQAAIDQINGKLNRVIEKTNEKFHQIEKEFSEVEG
RIQDLEKYVEDTKIDLWSYNAELLVALENQHTIDLTDSEMNKLFEKTRRQLRENAEEMGN
GCFKIYHKCDNACIESIRNGTYDHDVYRDEALNNRFQIKGVELKSGYKDWILWISFAISC
FLLCVVLLGFIMWACQRGNIRCNICI
"""
'''
"""
ha = ''.join(ha.split())   # toglie a capo e spazi

mask = seq.uniprot == 'P03437'
seq.loc[mask, 'sequence'] = ha
seq.loc[mask, 'seq_len'] = len(ha)   # 566

print(mask.sum(), 'righe aggiornate')

seq.to_csv('../output/chain_uniprot_sequences.tsv', sep='\t', index=False)
"""


match = 0
mismatch = []
for c in set(seq.chain):
    seqs = seq.loc[seq.chain == c, ['uniprot', 'sequence']].values.tolist()
    muts = data[data.chain == c]
    for wt, pos in zip(muts.wt, muts.seq_pos):
        pos = int(pos)
        for up, s in seqs:
            if 1 <= pos <= len(s) and s[pos - 1] == wt:
                match += 1
            else:
                mismatch.append((c, up, wt, pos, len(s)))

print('match:', match, '| mismatch:', len(mismatch))
print(pd.DataFrame(mismatch, columns=['chain', 'uniprot', 'wt', 'pos', 'len'])
        .groupby(['chain', 'uniprot']).size().sort_values(ascending=False))