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
                                # DEFINING FUNCTIONS #
#######################################################################################

def make_df(file, dfname, delimitor):
    '''Takes file path and makes pandas data frame'''
    try:
        dfname = pd.read_csv(file, sep = delimitor)
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
    return dfname

def tm_graph(df_graph, df_aln, final_name):
    '''Takes a graph and a foldseek dataframe and merges them into one graph
       with 3 types of edges'''
    
    # List of ID1 for chain graph
    chain_1 = list(df_graph['Chain_ID1'])
    # List of ID2 for chain graph
    chain_2 = list(df_graph['Chain_ID2'])
    # List of ID1 for FS tsv
    foldseek_1 =list(df_aln['query'])
    # List of ID2 for FS tsv
    foldseek_2 = list(df_aln['target'])   

    # List of pairs in chain graph
    pairs_gf = []
    # Dictionary = pair : edge_pdb
    pair_pdb = {}
    # Dictionary = pair : edge_up
    pair_up = {}

    for i in range(len(chain_1)):
        p = (chain_1[i], chain_2[i])
        pairs_gf.append(p)
        if graph_df['Edge_PDB'][i] != '' and df_graph['Edge_UP'][i] != '':
            pair_pdb[p]=df_graph['Edge_PDB'][i]
            pair_up [p]=df_graph['Edge_UP'][i]
        elif df_graph['Edge_PDB'][i] != '' and df_graph['Edge_UP'][i] == '':
            pair_pdb[p]=df_graph['Edge_PDB'][i]
            pair_up [p]= np.nan
        elif df_graph['Edge_PDB'][i] == '' and df_graph['Edge_UP'][i] != '':
            pair_pdb[p]=np.nan
            pair_up [p]= df_graph['Edge_UP'][i]    
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
        tm_scores = df_aln['alntmscore'][i]
        pair_tm[p] = df_aln['alntmscore'][i]
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
    
    final_name = pd.DataFrame({'ID1':id1, 'ID2':id2, 'Edge_PDB': pdb, 'Edge_UP': up, 'TM-score': sim})

    return final_name 

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
    i = 0
    for cc in ccs:
        chains = cc_ids[cc].split(',')
        tot = 0
        for c in chains:
            i += 1
            if c in s_c:
                tot += s_c[c]
        srvs.append(tot)
    
    cc_analysis_df = pd.DataFrame({'CC_ID': ccs, 'SRVs': srvs, 'PDB_arches': tot_pdb, 'Uniprot_arches': tot_up, 'Tot_nodes': tot_nodes, 'Nodes': nodes})
    return cc_analysis_df
 
#######################################################################################
                                # IMPORTING DATA #
#######################################################################################

foldseek = make_df('out.tsv', 'foldseek', '\t')

                    ############################################
                    ### GRAPH DATAFRAME: remove index column ###
                    ############################################

graph_df = make_df('Chain_ID_graph.tsv', 'graph_df', '\t')
del graph_df['Unnamed: 0'] # removing index column

s4169 = make_df('s4169.tsv', 's4169', '\t')

s4169['Chain_ID'] = s4169['pdb'].to_numpy() + '_' + s4169['mutation'].str[1].to_numpy()

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

both_checks = tm_graph(graph_df, foldseek_and_parsed, 'both_checks')
or_checks = tm_graph(graph_df, foldseek_or_parsed, 'or_checks')

print('\n\n')
print('Q and T have TM > 0.6\n')
print(both_checks.shape[0])
print('\n\n')
print('Q or T have TM > 0.6\n')
print(or_checks.shape[0])
print('\n\n')

srvs_chain = s4169['Chain_ID'].value_counts().to_dict()

both_checks_or_chains = both_checks[(both_checks['ID1'].isin(srvs_chain.keys())) | (both_checks['ID2'].isin(srvs_chain.keys()))]
print('Q and T have TM > 0.6\n\nID1 or ID2 carry SRV\n')
print(both_checks_or_chains.shape[0])
print('\n\n')

or_checks_or_chains = or_checks[(or_checks['ID1'].isin(srvs_chain.keys())) | (or_checks['ID2'].isin(srvs_chain.keys()))]
print('Q or T have TM > 0.6\n\nID1 or ID2 carry SRV\n')
print(or_checks_or_chains.shape[0])
print('\n\n')

both_checks_both_chains = both_checks[(both_checks['ID1'].isin(srvs_chain.keys())) & (both_checks['ID2'].isin(srvs_chain.keys()))]
print('Q and T have TM > 0.6\n\nID1 and ID2 carry SRV\n')
print(both_checks_both_chains.shape[0])
print('\n\n')

or_checks_both_chains = or_checks[(or_checks['ID1'].isin(srvs_chain.keys())) & (or_checks['ID2'].isin(srvs_chain.keys()))]
print('Q or T have TM > 0.6\n\nID1 and ID2 carry SRV\n')
print(or_checks_both_chains.shape[0])
print('\n\n')

six_G = [both_checks, or_checks,
         both_checks_or_chains, or_checks_or_chains,
         both_checks_both_chains, or_checks_both_chains]

six_names = ['tmp_00.tsv', 'tmp_01.tsv',
             'tmp_10.tsv', 'tmp_11.tsv',
             'tmp_20.tsv', 'tmp_21.tsv']

six_CC = ['cc_00.tsv', 'cc_01.tsv',
          'cc_10.tsv', 'cc_11.tsv',
          'cc_20.tsv', 'cc_21.tsv']

for i in range(len(six_G)):
    six_G[i].to_csv('G_tmp/'+six_names[i], sep = '\t')
    cc_an_df = cc_analysis(six_G[i], srvs_chain)

    max = -1
    for j in range(cc_an_df.shape[0]):
        r = 0
        if cc_an_df['SRVs'][j] > max:
            max = cc_an_df['SRVs'][j]
            r = j
    summary = pd.DataFrame([{'Tot CCs': cc_an_df.shape[0],
                             'PDB arches': six_G[i][six_G[i]['Edge_PDB'] == 'p'].shape[0],
                             'UP arches': six_G[i][six_G[i]['Edge_UP']== 'u'].shape[0],
                             'TM arches': six_G[i][six_G[i]['TM-score'].notna()].shape[0],
                             'Tot Arches': six_G[i].shape[0]}])
    print('\n', summary, '\n')
    print('\n'+six_CC[i][:-4]+'\n')
    print('\nThe CC with most SRVs is CC', cc_an_df['CC_ID'][r], 'counting', max, 'SRVs\n')

    cc_an_df.to_csv('G_tmp/CC_tmp/'+six_CC[i], sep = '\t')




