#######################################################################################
                                # IMPORTING LIBRARIES #
#######################################################################################

import pandas as pd
import numpy as np
import networkx as nx

#######################################################################################
                                # DEFINING FUNCTIONS #
#######################################################################################

def make_df(file, dfname, delimitor):
    '''Takes file path and makes pandas data frame'''
    try:
        dfname = pd.read_csv(file, sep = delimitor)
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
    return dfname

def cc_analysis(df_graph, s_c):
    '''Takes graph and data to make CC analysis'''
    G = nx.from_pandas_edgelist(df_graph, 'ID1', 'ID2', edge_attr=['Edge_PDB', 'Edge_UP', 'TM-score'])
    rows = []
    for i, cc in enumerate(nx.connected_components(G)):
        for node in cc:
            rows.append({"node": node, "CC_ID": i})
    
    cc_df = pd.DataFrame(rows)

                    ###########################################
                    ### 	    CC ID AND ITS NODES	        ###
                    ###########################################

    cc_tot_ids = {}
    cc_ids = {}
    ccs = list(set(cc_df['CC_ID']))
    for cc in ccs:
        nodes = list(cc_df[cc_df['CC_ID']==cc]['node'])
        cc_tot_ids[cc]=len(nodes)
        cc_ids[cc]=','.join(nodes)

                    ###########################################
                    ###  CC ID AND ITS PDB AND UNIPROT IDS  ###
                    ###########################################
    
    cc_tot_pdb = {}
    cc_tot_up = {}
    for cc in ccs:
        cc_tot_pdb[cc]=0
        cc_tot_up[cc]=0
    for cc in ccs:
        nodes = list(cc_df[cc_df['CC_ID']==cc]['node'])
        for id in nodes:
            id1_df = df_graph[df_graph['ID1']==id].reset_index(drop=True)
            for i in range(id1_df.shape[0]):
                if id1_df['Edge_PDB'][i]=='p':
                    cc_tot_pdb[cc]+=1
                elif id1_df['Edge_UP'][i]=='u':
                    cc_tot_up[cc]+=1
        id2_df = df_graph[df_graph['ID1']==id].reset_index(drop=True)
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
    
    srvs = []
    perc = []
    i = 0
    for cc in ccs:
        chains = cc_ids[cc].split(',')
        tot = 0
        for c in chains:
            i += 1
            if c in s_c:
                tot += s_c[c]
        srvs.append(tot)
        perc.append(tot/4169*100)
    
    cc_analysis_df = pd.DataFrame({'CC_ID': ccs, 'SRVs': srvs, 'SRVs %': perc, 'PDB_arches': tot_pdb, 'Uniprot_arches': tot_up, 'Tot_nodes': tot_nodes, 'Nodes': nodes})
    return cc_analysis_df

#######################################################################################
                                # IMPORTING DATA #
#######################################################################################

s4169 = make_df('s4169.tsv', 's4169', '\t')
s4169['Chain_ID'] = s4169['pdb'].to_numpy() + '_' + s4169['mutation'].str[1].to_numpy()
#print(s4169.head())

srvs_chain = s4169['Chain_ID'].value_counts().to_dict()

graph_df = make_df('G_tmp/tmp_20.tsv', 'graph_df', '\t')
del graph_df['Unnamed: 0'] # removing index column
#print(graph_df.head())

cc_df = make_df('G_tmp/CC_tmp/cc_20.tsv', 'cc_df', '\t')
del cc_df['Unnamed: 0'] # removing index column
#print(cc_df.head())

G_nodes = []
for i in range(len(cc_df['Nodes'].to_list())):
    G_nodes.extend(cc_df['Nodes'][i].split(','))
#print(len(G_nodes))

missing_nodes = []
for key in srvs_chain.keys():
    if key not in G_nodes:
        missing_nodes.append(key)
#print(len(missing_nodes))

#print(graph_df.columns)
#print(cc_df.columns)

id1 = graph_df['ID1'].to_list()
id2 = graph_df['ID2'].to_list()
Edge_PDB = graph_df['Edge_PDB'].to_list()
Edge_UP = graph_df['Edge_UP'].to_list()
tmscore = graph_df['TM-score'].to_list()

columns_G = [id1, id2, Edge_PDB, Edge_UP, tmscore]

for node in missing_nodes:
    id1.append(node)
    id2.append(node)
    Edge_PDB.append('p')
    Edge_UP.append('u')
    tmscore.append(1.00)

final_20 = pd.DataFrame({'ID1': id1, 'ID2': id2, 'Edge_PDB': Edge_PDB, 'Edge_UP': Edge_UP, 'TM-score': tmscore})
print(final_20.shape)

final_cc_df = cc_analysis(final_20, srvs_chain)
print(cc_df.shape)
print(final_cc_df.shape)

final_20.to_csv('final_20.tsv', sep = '\t')
final_cc_df.to_csv('final_20_cc.tsv', sep = '\t')

max = -1
for j in range(final_cc_df.shape[0]):
        r = 0
        if final_cc_df['SRVs'][j] > max:
            max = final_cc_df['SRVs'][j]
            r = j
summary = pd.DataFrame([{'Tot CCs': final_cc_df.shape[0],
                         'PDB arches': final_20[final_20['Edge_PDB'] == 'p'].shape[0],
                         'UP arches': final_20[final_20['Edge_UP']== 'u'].shape[0],
                         'TM arches': final_20[final_20['TM-score'].notna()].shape[0],
                         'Tot Arches': final_20.shape[0]}])
print('\n', summary, '\n')
print('\nThe CC with most SRVs is CC', final_cc_df['CC_ID'][r], 'counting', max, 'SRVs\n')

