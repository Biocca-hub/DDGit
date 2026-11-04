import pandas as pd

pairs = pd.read_csv('../output/chain_uniprot.tsv', sep='\t')

uniprots = sorted(pairs['uniprot'].dropna().unique())
print(len(uniprots))    # 219

with open('../output/uniprot_list.txt', 'w') as f:
    f.write('\n'.join(uniprots) + '\n')