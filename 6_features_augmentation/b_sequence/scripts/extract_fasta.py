# https://rayberkeley.com/posts/PDB_to_FASTA_Converter.html

from Bio import SeqIO
from pathlib import Path
import re

def build_fastas(single_output = False):
    
    # create empty FASTA file if necessary
    if single_output is True: 
        with open('../output/FASTAs/all_PDBs.fasta', 'w'): pass
        
    # traverse PDB directory and extract sequence information    
    for pdb_file in Path('../input/PDBs/').iterdir():
        
        for record in SeqIO.parse(f'{pdb_file}', "pdb-seqres"):
            name = re.sub('[^a-zA-Z0-9]', '_', record.id)
            seq = str(record.seq)
            
            # write unique FASTAs
            if single_output is False:
                with open(f'../output/FASTAs/{name}.fasta', 'w') as f:
                    f.write(f"> {name} \n")
                    f.write(seq)
            
            # append to single empty FASTA
            else:
                with open(f'../output/FASTAs/all_PDBs.fasta', 'a') as f:
                    f.write(f"> {name} \n")
                    f.write(seq)
                    f.write('\n\n')

build_fastas()
