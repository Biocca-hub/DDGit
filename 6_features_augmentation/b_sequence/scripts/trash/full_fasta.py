f_path = '../output/FASTAs/'
n_map = ['1BJ1_H', '1CZ8_L', '1KIP_B', '1KIQ_B', '1KIR_A', '1MHP_H', '1MHP_L', 
         '1N8Z_A', '1N8Z_B', '1NMB_H', '1NMB_L', '1QSE_C', '1XGP_B', '1XGQ_B', 
         '1XGR_B', '1XGT_B', '1XGU_B', '1YY9_C', '1YY9_D', '2B2X_L', '2BDN_H', 
         '2BDN_L', '2E7L_Q', '2NYY_C', '2NYY_D', '2OI9_Q', '3D3V_C', '3EQS_B', 
         '3EQY_C', '3G6D_H', '3HFM_H', '3L5X_L', '3LNZ_B', '3N85_H', '3N85_L', 
         '3NGB_H', '3NGB_L', '3PWP_C', '3QDG_E', '3QDJ_C', '3QDJ_E', '3QFJ_C', 
         '3QIB_D', '3SE8_H', '3SE8_L', '3SE9_H', '3SE9_L', '4FTV_D', '4FTV_E', 
         '4I77_H', '4I77_L', '4JFD_C', '4JFE_C', '4KRL_B', '4KRO_B', '4KRP_B', 
         '4L3E_D', '4L3E_E', '5C6T_H', '5C6T_L', '5M2O_B', '5UFE_B', '5XCO_B']
n_map_path = [f'{f_path}{i}.fasta' for i in n_map]

input_file = '../output/up_mapped.fasta'
output_file = '../output/up_mapped_2lines.fasta'

with open(input_file) as f, open(output_file, 'w') as out:
    header, seq = None, []
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if header is not None:
                out.write(f'{header}\n{"".join(seq)}\n')
            header, seq = line, []
        else:
            seq.append(line)
    if header is not None:                      # ultima proteina
        out.write(f'{header}\n{"".join(seq)}\n')

input_file = '../output/up_mapped_2lines.fasta'
output_file = '../output/up_header.fasta'

with open(input_file) as f, open(output_file, 'w') as out:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            out.write('>' + line.split('|')[1] + '\n')   # >sp|A0A075B6T6|... -> >A0A075B6T6
        else:
            out.write(line + '\n')

updated = '../output/complete.fasta'

with open(updated, 'w') as out:

    # 1. copia il file originale
    with open(output_file) as f:
        content = f.read()
    out.write(content)
    if not content.endswith('\n'):
        out.write('\n')

    # 2. aggiungi le 63 catene non mappate
    for name, path in zip(n_map, n_map_path):
        with open(path) as f:
            seq = ''.join(l.strip() for l in f if not l.startswith('>'))
        out.write(f'>{name}\n{seq}\n')
