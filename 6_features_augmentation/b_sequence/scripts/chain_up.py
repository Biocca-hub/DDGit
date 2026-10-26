import pandas as pd
from pathlib import Path
import urllib.request

# Mappatura ufficiale catena PDB -> UniProt (SIFTS, EBI)
SIFTS_URL = 'https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz'
SIFTS_FILE = Path('../input/pdb_chain_uniprot.tsv.gz')

if not SIFTS_FILE.exists():
    print('Scarico SIFTS...')
    urllib.request.urlretrieve(SIFTS_URL, SIFTS_FILE)

# prima riga = data del rilascio (commento), seconda = intestazione
# dtype=str e keep_default_na=False: chain id come "NA" non diventano NaN
sifts = pd.read_csv(SIFTS_FILE, sep='\t', skiprows=1, dtype=str, keep_default_na=False,
                    usecols=['PDB', 'CHAIN', 'SP_PRIMARY'])
sifts = sifts.drop_duplicates()
sifts['PDB'] = sifts['PDB'].str.upper()


df = pd.read_csv('../input/ddgit_dataset.tsv', sep='\t', dtype=str, keep_default_na=False)

# catene uniche del dataset (444)
chains = df[['chain', 'pdb', 'chain_id']].drop_duplicates()
print(f'Catene uniche nel dataset: {len(chains)}')

mapped = chains.merge(sifts, left_on=['pdb', 'chain_id'], right_on=['PDB', 'CHAIN'], how='left')
mapped = (mapped[['chain', 'SP_PRIMARY']]
          .rename(columns={'SP_PRIMARY': 'uniprot'})
          .drop_duplicates()
          .sort_values(['chain', 'uniprot'])
          .reset_index(drop=True))


# controlli
missing = mapped[mapped['uniprot'].isna()]
n_per_chain = mapped.dropna().groupby('chain')['uniprot'].nunique()
multi = n_per_chain[n_per_chain > 1]

print(f'Coppie chain-uniprot: {len(mapped)}')
print(f'Catene senza UniProt: {len(missing)}')
if len(missing):
    print(missing['chain'].to_list())
print(f'Catene con più di un UniProt: {len(multi)}')
if len(multi):
    print(mapped[mapped['chain'].isin(multi.index)].to_string(index=False))
print(f'UniProt unici: {mapped["uniprot"].nunique()}')


mapped.to_csv('../output/chain_uniprot.tsv', sep='\t', index=False)