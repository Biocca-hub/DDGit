#######################################################################################
#######################################################################################
#######################################################################################
                                # GRAPH BUILDING #
#######################################################################################
#######################################################################################
#######################################################################################






#######################################################################################
                                # IMPORTING LIBRARIES #
#######################################################################################

import pandas as pd
import numpy as np
import networkx as nx
import itertools

#######################################################################################
                                # IMPORTING DATA #
#######################################################################################

def make_df(file, dfname, delimitor):
    'Takes file path and makes pandas data frame'
    try:
        dfname = pd.read_csv(file, sep = delimitor)
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
    return dfname

foldseek = make_df('out.tsv', 'foldseek', '\t')

                    ############################################
                    ### ONLY KEEPS PAIRS WITH TM-SCORE ≥ 0.6 ###
                    ############################################

foldseek_parsed = foldseek[foldseek['alntmscore']>=0.6].reset_index(drop=True) # global TM-score threshold
foldseek_or_parsed = foldseek_parsed[(foldseek_parsed['qtmscore']>=0.6) | (foldseek_parsed['ttmscore']>=0.6)].reset_index(drop=True) # threshold for tm score normalized wrt query length
foldseek_and_parsed = foldseek_parsed[(foldseek_parsed['qtmscore']>=0.6) & (foldseek_parsed['ttmscore']>=0.6)].reset_index(drop=True) # threshold for tm score normalized wrt query length
#print(foldseek_or_parsed.shape)
#print(foldseek_or_parsed.head())
#print(foldseek_and_parsed.shape)
#print(foldseek_and_parsed.head())

                    ############################################
                    ### GRAPH DATAFRAME: remove index column ###
                    ############################################

graph_df = make_df('Chain_ID_graph.tsv', 'graph_df', '\t')
del graph_df['Unnamed: 0'] # removing index column
print(graph_df.head())

#######################################################################################
                                # CHEKING PAIRS IN DFs #
#######################################################################################

# List of ID1 for chain graph
chain_1 = list(graph_df['Chain_ID1'])
# List of ID2 for chain graph
chain_2 = list(graph_df['Chain_ID2'])
# List of ID1 for FS tsv
foldseek_1 =list(foldseek_and_parsed['query'])
# List of ID2 for FS tsv
foldseek_2 = list(foldseek_and_parsed['target'])

# List of pairs in chain graph
pairs_gf = []
# Dictionary = pair : edge_pdb
pair_pdb = {}
# Dictionary = pair : edge_up
pair_up = {}

for i in range(len(chain_1)):
  p = (chain_1[i], chain_2[i])
  pairs_gf.append(p)
  if graph_df['Edge_PDB'][i] != '' and graph_df['Edge_UP'][i] != '':
    pair_pdb[p]=graph_df['Edge_PDB'][i]
    pair_up [p]=graph_df['Edge_UP'][i]
  elif graph_df['Edge_PDB'][i] != '' and graph_df['Edge_UP'][i] == '':
    pair_pdb[p]=graph_df['Edge_PDB'][i]
    pair_up [p]= np.nan
  elif graph_df['Edge_PDB'][i] == '' and graph_df['Edge_UP'][i] != '':
    pair_pdb[p]=np.nan
    pair_up [p]= graph_df['Edge_UP'][i]    
  else:
    pair_pdb[p]= np.nan
    pair_up [p]= np.nan

# List of pairs in FS
pairs_fs = []
# Dictionary = pair: TM-score
pair_tm = {}
# List of TM-scores
tm_scores = []

for i in range(len(foldseek_1)):
  p = (foldseek_1[i],foldseek_2[i])
  pairs_fs.append(p)
  tm_scores = foldseek_and_parsed['alntmscore'][i]
  pair_tm[p] = foldseek_and_parsed['alntmscore'][i]
  if p not in pairs_gf:
    pair_pdb[p]= np.nan
    pair_up [p]= np.nan
i = 0
not_shared = []
for pair in pairs_gf:
  if pair not in pairs_fs:
    i += 1
    not_shared.append(pair)
    pair_tm[pair]= np.nan

#print(len(not_shared))

#print(len(list(pair_tm.keys())) == len(pairs_fs) + len(not_shared))

id1 = []
id2 = []
pdb = []
up = []
sim = []
for pair in pair_tm:
  id1.append(pair[0])
  id2.append(pair[1])
  pdb.append(pair_pdb[pair])
  up.append(pair_up[pair])
  sim.append(pair_tm[pair])

grafo = pd.DataFrame({'ID1':id1, 'ID2':id2, 'Edge_PDB': pdb, 'Edge_UP': up, 'TM-score': sim})
#grafo = pd.DataFrame({'ID1':id1, 'ID2':id2, 'Edge_UP': up, 'TM-score': sim})
print(grafo.head())
grafo.to_csv('final_graph_nan.tsv', sep = '\t')




'''
#######################################################################################
#######################################################################################
#######################################################################################
                                # GRAPH BUILDING #
#######################################################################################
#######################################################################################
#######################################################################################






#######################################################################################
                                # IMPORTING DATA #
#######################################################################################

s4169 = make_df('s4169.tsv', 's4169', '\t')

#######################################################################################
                          # CONNECTED COMPONENTS ANALYSIS #
#######################################################################################

G = nx.from_pandas_edgelist(grafo, 'ID1', 'ID2', edge_attr=['Edge_PDB', 'Edge_UP', 'TM-score'])
rows = []
for i, cc in enumerate(nx.connected_components(G)):
  for node in cc:
    rows.append({"node": node, "CC_ID": i})

cc_df = pd.DataFrame(rows)

		                ############################################
                    ### 	        CC ID AND ITS NODES	       ###
                    ############################################

cc_tot_ids = {}
cc_ids = {}
ccs = list(set(cc_df['CC_ID']))
for cc in ccs:
  nodes = list(cc_df[cc_df['CC_ID']==cc]['node'])
  cc_tot_ids[cc]=len(nodes)
  cc_ids[cc]=','.join(nodes)

                    ############################################
                    ###  CC ID AND ITS PDB AND UNIPROT IDS   ###
                    ############################################

cc_tot_pdb = {}
cc_tot_up = {}
for cc in ccs:
  cc_tot_pdb[cc]=0
  cc_tot_up[cc]=0
for cc in ccs:
  nodes = list(cc_df[cc_df['CC_ID']==cc]['node'])
  for id in nodes:
    id1_df = grafo[grafo['ID1']==id].reset_index(drop=True)
    for i in range(id1_df.shape[0]):
      if id1_df['Edge_PDB'][i]=='p':
        cc_tot_pdb[cc]+=1
      elif id1_df['Edge_UP'][i]=='u':
        cc_tot_up[cc]+=1
    id2_df = grafo[grafo['ID1']==id].reset_index(drop=True)
    for i in range(id2_df.shape[0]):
      if id2_df['Edge_PDB'][i]=='p':
        cc_tot_pdb[cc]+=1
      elif id2_df['Edge_UP'][i]=='u':
        cc_tot_up[cc]+=1

                    ############################################
                    ###          COLUMNS WRT CC IDS          ###
                    ############################################

nodes = []
tot_nodes = []
tot_pdb = []
tot_up = []
for cc in ccs:
  nodes.append(cc_ids[cc])
  tot_nodes.append(cc_tot_ids[cc])
  tot_pdb.append(cc_tot_pdb[cc])
  tot_up.append(cc_tot_up[cc])

pdb = s4169['pdb'].to_numpy()
chain = pdb + '_' + s4169['mutation'].str[1]
s4169['chain']=chain

srv_chain = s4169['chain'].value_counts().to_dict()

srvs = []
i = 0
for cc in ccs:
  chains = cc_ids[cc].split(',')
  tot = 0
  for c in chains:
    i += 1
    if c in srv_chain:
      tot += srv_chain[c]
  srvs.append(tot)

                    ############################################
                    ###       CC ID ANALYSIS DATA FRAME      ###
                    ############################################

cc_analysis_df = pd.DataFrame({'CC_ID': ccs, 'SRVs': srvs, 'PDB_arches': tot_pdb, 'Uniprot_arches': tot_up, 'Tot_nodes': tot_nodes, 'Nodes': nodes})

print(cc_analysis_df.shape)

#######################################################################################
                          		# TSVs #
#######################################################################################

# Saving chain ID graph and CC analysis as tsv files
'''
#grafo.to_csv('final_graph_v_1.tsv', sep = '\t')
#cc_analysis_df.to_csv('CC_analysis_v_1.tsv', sep = '\t')

