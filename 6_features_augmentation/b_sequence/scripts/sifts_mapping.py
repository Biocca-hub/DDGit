import pandas as pd

seq = pd.read_csv('../output/sifts_mapping.tsv', sep = '\t')
seq = seq[seq['uniprot'] != 'Not Mapped'].reset_index(drop=True)

df = pd.read_csv('../input/ddgit_dataset.tsv', sep = '\t')

# per ogni catena, la lista dei suoi segmenti
segments = {}
for i in range(seq.shape[0]):
    chain = seq.chain[i]
    if chain not in segments:
        segments[chain] = []
    segments[chain].append((int(seq.pdb_start[i]),
                            int(seq.pdb_end[i]),
                            int(seq.unp_start[i]),
                            str(seq.unp_sequence[i])))

def check(pos_col):
    ok = 0
    wrong = 0
    skipped = 0

    for i in range(df.shape[0]):
        chain = df.chain[i]
        pos = str(df[pos_col][i])

        # catena non mappata o posizione con inserzione (es. 100a)
        if chain not in segments or not pos.isdigit():
            skipped += 1
            continue

        pos = int(pos)
        aa = None

        for pdb_start, pdb_end, unp_start, s in segments[chain]:
            if pdb_start <= pos <= pdb_end:
                idx = unp_start - 1 + (pos - pdb_start)
                if idx < len(s):
                    aa = s[idx]
                break

        if aa == df.wt[i]:
            ok += 1
        else:
            wrong += 1

    print(pos_col, '-> giusti:', ok, 'sbagliati:', wrong, 'saltati:', skipped)

check('pdb_pos')
check('seq_pos')


'''
pdb 1–129  ↔  P00698 19–147

Il residuo 1 della catena corrisponde al residuo 19 di UniProt. Nel lisozima i residui 1–18 di UniProt sono il peptide segnale, 
che non c’è nella proteina matura cristallizzata.
C’è però un punto importante sulla “finestra di residui riportati in struttura”. 
Nella numerazione di PDBe, pdb_start/pdb_end si riferiscono alla sequenza depositata della catena (SEQRES), 
non ai residui che hanno davvero coordinate. Di conseguenza:

Se nella struttura mancano residui non risolti (loop disordinati, code terminali), questi rientrano comunque nella finestra del mapping.
La numerazione pdb_start/pdb_end è progressiva da 1 e non è la numerazione degli autori, cioè quella che vedi nel file PDB, 
che può partire da altri numeri o avere codici di inserzione come 52A. 
Quindi, se le posizioni delle mutazioni nel tuo dataset usano la numerazione del file PDB, non puoi usare direttamente pdb_start come offset.

Per capire quale numerazione hai, controlla una mutazione nota: 
guarda se il residuo in quella posizione nel file PDB corrisponde a quello che ottieni con 

sequence[unp_start - 1 + (pos - pdb_start)]. 

Se non torna, ti serve il mapping per singolo residuo, che SIFTS fornisce, e possiamo vedere come prenderlo.
'''
