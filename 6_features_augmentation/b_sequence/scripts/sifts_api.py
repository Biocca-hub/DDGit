from tqdm import tqdm
import pandas as pd
import requests
import json
from pprint import pprint

df = pd.read_csv('../input/ddgit_dataset.tsv', sep = '\t')

chains = list({i for i in list(df.chain)})

chain_ids = []
for i in chains:
    chain_ids.append(i.split('_')[1])
# print(len(chains))
# 444

notmapped = []
chain = []
uniprot = []
unp_start = []
unp_end = []
pdb_start = []
pdb_end = []

log = open('log.txt', 'w')
log.write('pdb\tchain\tuniprot\n')

for i in tqdm(range(len(chains))):
    
    pdb_id = list(chains)[i].split('_')[0]

    url = f"https://www.ebi.ac.uk/pdbe/api/v2/mappings/isoforms/{pdb_id}"

    '''
    Provides, for a given PDB structure, aggregated SIFTS residue-level mappings linking each PDB chain 
    (auth_asym_id or struct_asym_id) to its corresponding UniProt accession or isoform, selecting the isoform 
    with the highest sequence coverage and identity.
    '''

    response = requests.get(url)

    data = json.loads(response.text)

    # #pprint(data) --> dizionario di dizionari UniProt ID specifici con varie info:
    # a noi interessa [UP ID] --> [mappings] --> [chain_id] deve essere la stessa trovata nel dataset

    unp_info = data[pdb_id.lower()]['UniProt']

    chain_id = chain_ids[i]

    found = False

    for unp in unp_info:

        for mapping in unp_info[unp]['mappings']:
            
            if mapping['chain_id'] == chain_id:
                
                chain.append(chains[i])
                uniprot.append(unp)
                pdb_start.append(mapping['pdb_start'])
                pdb_end.append(mapping['pdb_end'])
                unp_start.append(mapping['unp_start'])
                unp_end.append(mapping['unp_end'])
                
                found = True

    if not found:
        
        chain.append(chains[i])
        uniprot.append('Not Mapped')
        pdb_start.append('Not Mapped')
        pdb_end.append('Not Mapped')
        unp_start.append('Not Mapped')
        unp_end.append('Not Mapped')
        
        notmapped.append((pdb_id, chain_id))
        
        log.write(f'{pdb_id}\t{chain_id}\t-\n')

log.close()

pd.DataFrame({'chain': chain,
              'uniprot': uniprot,
              'pdb_start': pdb_start,
              'pdb_end': pdb_end,
              'unp_start': unp_start,
              'unp_end': unp_end}).to_csv('results.tsv', sep = '\t')