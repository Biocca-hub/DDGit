import pandas as pd 
import requests
import json

df = pd.read_csv('results.tsv', sep = '\t')
del df['Unnamed: 0']
#print(df.shape[0])

#print(len(df.chain.unique()))
# 444 catene 

#print(df[df['uniprot'] == 'Not Mapped'].shape)
# 63 non mappate su uniprot 

seqs = []

for unp in df.uniprot:
    if unp != 'Not Mapped':
        
        if len(unp.split('-')) > 1:
            acc = unp.split('-')[0]
        else: 
            acc = unp

        url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"

        response = requests.get(url)
        data = json.loads(response.text)

        sequence = data['sequence']['value']
        
        seqs.append(sequence)
    
    else:

        seqs.append('')

df['unp_sequence'] = seqs

df.to_csv('../output/sifts_mapping.tsv', sep = '\t', index = False)