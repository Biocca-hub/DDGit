import pandas as pd
import requests
import gzip
import xml.etree.ElementTree as ET
from tqdm import tqdm
import time

df = pd.read_csv('../input/ddgit_dataset.tsv', sep = '\t')

del df['seq_pos']

# creiamo una colonna unp_pos

# =============================================================================
# 1. COME È FATTO IL FILE XML DI SIFTS
# =============================================================================
# Per ogni struttura PDB, SIFTS fornisce un file XML compresso (.xml.gz):
#   https://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{pdb_id minuscolo}.xml.gz
#
# Dentro, ogni residuo della catena è un tag <residue> con questa forma
# (semplificata):
#
#   <residue dbSource="PDBe" dbResNum="25" dbResName="SER">        <- numerazione SEQRES
#       <crossRefDb dbSource="PDB" dbResNum="68" dbResName="SER"
#                   dbChainId="D"/>                                 <- numerazione AUTORI + catena
#       <crossRefDb dbSource="UniProt" dbAccessionId="P01234"
#                   dbResNum="92" dbResName="S"/>                   <- posizione UNIPROT + aa
#       ...altri crossRefDb (Pfam, CATH, ...) che non ci interessano
#   </residue>
#
# Quindi: partendo da (catena autori, numero autori) -> che è quello che abbiamo
# nel dataset (chain_id, pdb_pos) -> arriviamo direttamente a (UniProt, posizione).
#
# NB: i codici di inserzione nell'XML sono attaccati al numero e in MAIUSCOLO
# (es. '100A'), mentre nel nostro dataset sono in minuscolo ('100a').


# Tutti i tag dell'XML di SIFTS hanno davanti questo "namespace".
# ElementTree lo vuole scritto tra graffe davanti al nome del tag,
# altrimenti non trova niente cercando solo 'residue'.
ns = '{http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd}'


# =============================================================================
# 2. FUNZIONE CHE SCARICA E LEGGE L'XML DI UN PDB
# =============================================================================
def sifts_residues(pdb_id):
    '''
    Scarica l'XML di SIFTS per pdb_id e restituisce un dizionario:

        (catena_autori, numero_autori) -> (uniprot, posizione_uniprot, aa_uniprot)

    es. ('D', '68') -> ('P01234', 92, 'S')

    Se il file non si scarica, restituisce un dizionario vuoto
    (così tutte le mutazioni di quel PDB finiranno come "Not Mapped").
    '''

    url = f'https://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{pdb_id.lower()}.xml.gz'
    
    # provo fino a 5 volte; se il server rifiuta, aspetto e riprovo
    
    response = None
    for tentativo in range(5):
        try:
            response = requests.get(url, timeout = 60)
            break                      # scaricato: esco dal ciclo dei tentativi
        except requests.exceptions.RequestException:
            print('Connessione rifiutata per', pdb_id, '- riprovo tra 30 s')
            time.sleep(30)

    if response is None or response.status_code != 200:
        print('Impossibile scaricare', pdb_id)
        return {}

    time.sleep(0.5)   # piccola pausa tra un PDB e l'altro, per non farsi bloccare

    # response.content = byte del file compresso
    # gzip.decompress   = byte dell'XML in chiaro
    # ET.fromstring     = legge l'XML e restituisce la "radice" dell'albero
    root = ET.fromstring(gzip.decompress(response.content))

    mapping = {}

    # root.iter(...) scorre TUTTI i <residue> del file, di tutte le catene
    for residue in root.iter(ns + 'residue'):

        pdb_ref = None   # il crossRefDb con dbSource="PDB"
        unp_ref = None   # il crossRefDb con dbSource="UniProt"

        # findall(...) prende i crossRefDb figli diretti di questo residuo
        for ref in residue.findall(ns + 'crossRefDb'):
            if ref.get('dbSource') == 'PDB':
                pdb_ref = ref
            elif ref.get('dbSource') == 'UniProt' and unp_ref is None:
                # se ce n'è più di uno teniamo il primo
                unp_ref = ref

        # Se manca il riferimento PDB o UniProt il residuo non ci serve:
        # - niente UniProt: tag di espressione, linker, catene non mappate (anticorpi)
        if pdb_ref is None or unp_ref is None:
            continue

        # - residui non osservati: hanno dbResNum="null" nella numerazione autori
        if pdb_ref.get('dbResNum') == 'null':
            continue

        # chiave: (catena autori, numero autori come stringa, es. '68' o '100A')
        key = (pdb_ref.get('dbChainId'), pdb_ref.get('dbResNum'))

        # valore: (accession UniProt, posizione UniProt come intero, aa a una lettera)
        mapping[key] = (unp_ref.get('dbAccessionId'),
                        int(unp_ref.get('dbResNum')),
                        unp_ref.get('dbResName'))

    return mapping


# =============================================================================
# 3. CICLO SULLE MUTAZIONI DEL DATASET
# =============================================================================

# cache: pdb_id -> dizionario di sifts_residues(pdb_id)
# Così ogni PDB viene scaricato UNA volta sola, anche se ha centinaia di mutazioni.
cache = {}

uniprot = []      # accession UniProt di ogni mutazione
unp_pos = []      # posizione del residuo mutato sulla sequenza UniProt
unp_wt = []       # amminoacido che UniProt ha in quella posizione (per il controllo)

log = open('log_unp_pos.txt', 'w')
log.write('unique\tpdb\tchain\tpdb_pos\twt\n')

for i in tqdm(range(df.shape[0])):

    pdb_id = df.pdb[i]

    # scarico l'XML solo se non l'ho già fatto
    if pdb_id not in cache:
        cache[pdb_id] = sifts_residues(pdb_id)

    # costruisco la chiave nello stesso formato dell'XML:
    # str() perché alcune posizioni hanno l'inserzione (es. '100a'),
    # .upper() perché nell'XML è '100A'
    key = (str(df.chain_id[i]), str(df.pdb_pos[i]).upper())

    if key in cache[pdb_id]:
        unp_id, pos, aa = cache[pdb_id][key]
        uniprot.append(unp_id)
        unp_pos.append(pos)
        unp_wt.append(aa)

    else:
        # mutazione non mappabile (catena senza UniProt, residuo fuori dal mapping...)
        uniprot.append('Not Mapped')
        unp_pos.append(None)
        unp_wt.append(None)
        log.write(f'{df.unique[i]}\t{pdb_id}\t{df.chain_id[i]}\t{df.pdb_pos[i]}\t{df.wt[i]}\n')

log.close()

df['uniprot'] = uniprot
df['unp_pos'] = unp_pos
df['unp_wt'] = unp_wt

# unp_pos contiene dei None -> pandas la renderebbe float (92.0).
# 'Int64' (con la I maiuscola) è il tipo intero di pandas che accetta i valori mancanti.
df['unp_pos'] = df['unp_pos'].astype('Int64')


# =============================================================================
# 4. CONTROLLO: IL WILD-TYPE DEL DATASET È UGUALE A QUELLO DI UNIPROT?
# =============================================================================
mapped = df[df['uniprot'] != 'Not Mapped']

ok = (mapped.wt == mapped.unp_wt).sum()
wrong = (mapped.wt != mapped.unp_wt).sum()
not_mapped = (df['uniprot'] == 'Not Mapped').sum()

print('mutazioni totali:', df.shape[0])
print('wt uguale a UniProt:', ok)
print('wt diverso da UniProt:', wrong)
print('non mappate:', not_mapped)

# Le mutazioni con wt diverso vanno guardate a mano:
# di solito sono varianti/mutanti già presenti nella struttura cristallizzata
# (conflitti di sequenza rispetto a UniProt). Le salvo in un file a parte.
mapped[mapped.wt != mapped.unp_wt].to_csv('../output/wt_mismatch.tsv', sep = '\t', index = False)


# =============================================================================
# 5. SALVATAGGIO
# =============================================================================
df.to_csv('../output/ddgit_dataset_unp_pos.tsv', sep = '\t', index = False)