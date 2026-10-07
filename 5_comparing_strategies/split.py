#######################################################################################
                                # IMPORTING LIBRARIES #
#######################################################################################

import pandas as pd
import numpy as np

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

def parse_lists_df(series):
    final_list = []
    for el in series:
        el = el[1:-1].split(', ')
        toadd = []
        for e in el:
            toadd.append(e)
        final_list.append(','.join(toadd))
    if len(final_list)==4169:
        return final_list

#######################################################################################
                                # IMPORTING DATA #
#######################################################################################

metadata = make_df('metadata.tsv', 'metadata', '\t')
s4169 = make_df('s4169.tsv', 's4169', '\t')
s4943_compact = make_df('s4943_collapsed.tsv', 's4943_compact', '\t')
del s4943_compact['Unnamed: 0']
cc_info = make_df('../connected_components/final_20_cc.tsv', 'cc_info', '\t')
del cc_info['Unnamed: 0'] # removing index column

cc_sorted_d = cc_info.sort_values('SRVs', ascending = False).reset_index()

S4169 = make_df('S4169_all_info.tsv', 'S4169', '\t')

#######################################################################################
                            # CCs of blind test set (BTS) #
#######################################################################################

bts_srvs = 417

counter = 0 # counts SRVs
cc_ids = [] # stores CC_IDs with smaller amounts of SRVs
i = cc_sorted_d.shape[0]-1 # The n. of rows is +1 wrt last index
while counter <= bts_srvs:
    counter += cc_sorted_d['SRVs'][i]
    cc_ids.append(cc_sorted_d['CC_ID'][i])
    i -= 1 

ccs_bts = cc_sorted_d[cc_sorted_d.index > i]
del ccs_bts['index']

#######################################################################################
                            # CCs of training set (TR) #
#######################################################################################

ccs_tr = cc_sorted_d[cc_sorted_d.index <= i]
del ccs_tr['index']

#######################################################################################
                            # Saving TR and BTS tsv files #
#######################################################################################

#ccs_bts.to_csv('BTS.tsv', sep = '\t')
#ccs_tr.to_csv('TR.tsv', sep = '\t')

#######################################################################################
                            # Data collection for BTS #
#######################################################################################

s4169['unique'] = s4169['pdb'].to_numpy()+'_'+s4169['mutation'].str[1]+'_'+s4169['mutation'].to_numpy()

'''ddgs_averaged = []
j = 0
for i in range(s4943_compact.shape[0]):
    unique_id = s4943_compact['unique'][i]
    ddg = s4943_compact['DDG'][i]
    if len(ddg.split(',')) > 1:
       for k in range(s4169.shape[0]):
        if unique_id == s4169['unique'][k]:
            ddgs_averaged.append(s4169['actual'][k])
            j += 1
    elif len(ddg.split(',')) == 1:    
        ddgs_averaged.append(float(s4943_compact['DDG'][i][1:-1]))

s4943_compact['DDG_avg'] = ddgs_averaged

unique = s4943_compact['unique']

method = s4943_compact['DDG_method']

exp_m = []
for el in method:
    el = el[1:-1].split(', ')
    toadd = []
    for e in el:
        e = e[1:-1]
        toadd.append(e)
    exp_m.append(','.join(set(toadd)))

s4169_complete = pd.DataFrame({'Unique': unique, 'PDB_mut': s4943_compact['PDB_mut'], 'Clean_mut': s4943_compact['Clean_mut'],
                               'PDB_ID': s4943_compact['PDB_ID'], 'Chain': s4943_compact['Chain_ID'], 'Uniprot_ID': s4943_compact['Uniprot_ID'],
                               'DDG_avg': ddgs_averaged,'DDG': parse_lists_df(s4943_compact['DDG']), 
                               'Affinity_WT': parse_lists_df(s4943_compact['Affinity_WT']), 'Affinity_M': parse_lists_df(s4943_compact['Affinity_M']),
                               'Location': s4943_compact['Location'], 'Experimental_method': exp_m,
                               'SKEMPI_version': s4943_compact['Skempi_v'], 'out_type': s4943_compact['Out_type'], 'out_protein': s4943_compact['Out_protein']})

s4169_complete.to_csv('S4169_all_info.tsv', sep = '\t')'''

bts_data = pd.DataFrame(columns=S4169.columns)
cc_ids = []
for i in range(ccs_bts.shape[0]):
    for node in ccs_bts['Nodes'].iloc[i].split(','):
        data = S4169[S4169['Chain']==node]
        bts_data = pd.concat([bts_data, data], ignore_index=True)
    for j in range(ccs_bts['SRVs'].iloc[i]):
        cc_ids.append(ccs_bts['CC_ID'].iloc[i])
bts_data['CC_ID'] = cc_ids
del bts_data['Unnamed: 0']

tr_data = pd.DataFrame(columns=S4169.columns)
cc_ids = []
for i in range(ccs_tr.shape[0]):
    for node in ccs_tr['Nodes'].iloc[i].split(','):
        data = S4169[S4169['Chain']==node] 
        tr_data = pd.concat([tr_data, data], ignore_index=True)
    for j in range(ccs_tr['SRVs'].iloc[i]):
        cc_ids.append(ccs_tr['CC_ID'].iloc[i])

tr_data['CC_ID'] = cc_ids
del tr_data['Unnamed: 0']

#bts_data.to_csv('BTS_data.tsv', sep = '\t')
#tr_data.to_csv('TR_data.tsv', sep = '\t')

fold_1 = [0] # 1659 srv
fold_2 = [9] # 779 srv
fold_3 = [51, 35, 31, 26, 54, 12, 39, 3, 14, 36, 20] # 437 srv
fold_4 = [6, 2, 53, 5, 4, 19, 45] # 414 srv
fold_5 = [15, 11, 8, 7, 52, 1] # 458 srv 

training_folds = [fold_1, fold_2, fold_3, fold_4, fold_5]
df_1 = pd.DataFrame(columns=tr_data.columns) 
df_2 = pd.DataFrame(columns=tr_data.columns) 
df_3 = pd.DataFrame(columns=tr_data.columns) 
df_4 = pd.DataFrame(columns=tr_data.columns) 
df_5 = pd.DataFrame(columns=tr_data.columns)
tr_dfs = [df_1, df_2, df_3, df_4, df_5]

for i in range(5):
    for fold in training_folds[i]:
        tr_dfs[i] = pd.concat([tr_dfs[i], tr_data[tr_data['CC_ID']==fold]], ignore_index=True)
        #tr_dfs[i].to_csv('training/fold_'+str(i+1)+'.tsv', sep = '\t')

'''
i = 0
for fold in training_folds:
    totnodes = 0
    tot_srvs = 0
    for cc_id in fold:
        nodes = len(set(tr_data[tr_data['CC_ID']==cc_id]['Chain'].to_list()))
        totnodes += nodes
        srvs = len(set(tr_data[tr_data['CC_ID']==cc_id]['Unique'].to_list()))
        tot_srvs += srvs

    print(totnodes == len(set(tr_dfs[i]['Chain'].to_list())))
    print(tr_dfs[i].shape[0] == tot_srvs)'''

cols = ['Fold', 'SRVs', 'Complexes', 'Chains', 'Uniprot_IDs', 'CCs','CC_IDs']
stat_1 = pd.DataFrame(columns=cols) 
stat_2 = pd.DataFrame(columns=cols) 
stat_3 = pd.DataFrame(columns=cols) 
stat_4 = pd.DataFrame(columns=cols) 
stat_5 = pd.DataFrame(columns=cols) 
statistics = [stat_1, stat_2, stat_3, stat_4, stat_5]

tot_stats = pd.DataFrame(columns=cols) 
for i in range(5):
    
    statistics[i]['Fold'] = [i+1]
    statistics[i]['SRVs'] = [tr_dfs[i].shape[0]]
    statistics[i]['Complexes'] = [len(set(tr_dfs[i]['PDB_ID'].to_list()))]
    statistics[i]['Chains'] = [len(set(tr_dfs[i]['Chain'].to_list()))]
    u = list(set(tr_dfs[i]['Uniprot_ID'].to_list()))
    up_ids = []
    for up in u:
        for el in up.split(','):
            if el not in up_ids and el!='Not Mapped':
                up_ids.append(el)
    statistics[i]['Uniprot_IDs'] = [len(up_ids)]
    statistics[i]['CCs'] = [len(training_folds[i])]
    statistics[i]['CC_IDs'] = str(training_folds[i])[1:-1]

    tot_stats = pd.concat([tot_stats, statistics[i]], ignore_index=True)

bts_stats = pd.DataFrame(columns=cols) 
for i in range(5):
    bts_stats['Fold'] = ['BTS']
    bts_stats['SRVs'] = [bts_data.shape[0]]
    bts_stats['Complexes'] = [len(set(bts_data['PDB_ID'].to_list()))]
    bts_stats['Chains'] = [len(set(bts_data['Chain'].to_list()))]
    u = list(set(bts_data['Uniprot_ID'].to_list()))
    up_ids = []
    for up in u:
        for el in up.split(','):
            if el not in up_ids and el!='Not Mapped':
                up_ids.append(el)
    bts_stats['Uniprot_IDs'] = [len(up_ids)]
    bts_stats['CCs'] = [len([55, 18, 44, 56, 38, 48, 57, 27, 58, 33, 41, 59, 61,
                           8, 10, 60, 16, 50, 17, 23, 65, 64, 63, 62, 34, 25, 
                           67, 66, 47, 68, 70, 37, 21, 69, 30, 24, 74, 40, 72, 
                           71, 42, 73, 22, 75, 77, 32, 76, 79, 82, 78, 46, 49, 
                           84, 83, 80, 81, 29, 13, 43, 85, 86, 87, 88, 89, 90, 
                           91, 92, 93])]
    bts_stats['CC_IDs'] = str([55, 18, 44, 56, 38, 48, 57, 27, 58, 33, 41, 59, 61,
                           8, 10, 60, 16, 50, 17, 23, 65, 64, 63, 62, 34, 25, 
                           67, 66, 47, 68, 70, 37, 21, 69, 30, 24, 74, 40, 72, 
                           71, 42, 73, 22, 75, 77, 32, 76, 79, 82, 78, 46, 49, 
                           84, 83, 80, 81, 29, 13, 43, 85, 86, 87, 88, 89, 90, 
                           91, 92, 93])[1:-1]

tot_stats = pd.concat([tot_stats, bts_stats], ignore_index=True)
#print(tot_stats.head(6))

tot_stats = pd.concat([tot_stats, pd.DataFrame({'Fold': ['TOT'], 'SRVs': [4169], 'Complexes': [319], 'Chains': [444], 'Uniprot_IDs': [218], 
                                                'CCs': [94], 'CC_IDs': ['0-93']})], ignore_index=True)
#print(tot_stats.head(7))
tot_stats.to_csv('split_stats.tsv', sep = '\t')



