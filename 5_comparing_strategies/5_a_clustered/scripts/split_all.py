import pandas as pd 

training = pd.read_csv('training.tsv', sep = '\t')
bts = pd.read_csv('blind_test_set.tsv', sep = '\t')

print(f'Training set counts {training.shape[0]} srv')
print('-'*105)
print(f'Blind test set counts {bts.shape[0]} srv')
print('-'*105)
print(f'Total srv: {training.shape[0]+bts.shape[0]}')
print('-'*105)

data = pd.concat([training, bts], axis=0, ignore_index = True)
#print(data.columns.to_list())
data.drop('Unnamed: 0', axis='columns', inplace=True)
#print(data.columns.to_list())
#print(data.head())
#print(data.tail())
print(f'Ideally, to split the dataset in 5 folds we should obtain 5 clusters of around {data.shape[0]/5} mutations.')
print('But he first 2 clusters are very numerous:')
print(data.CC_ID.value_counts()[:2])
print(f'We have to keep them as they are, and split the remaining data in 3 clusters of around {(data.shape[0]-1659-779)/3} mutations')
print('-'*105)

ccs = pd.read_csv('CC_info.tsv', sep = '\t').sort_values('SRVs', ascending = False)
ccs.drop('Unnamed: 0', axis='columns', inplace=True)
#ccs.to_csv('CC_info.tsv', sep = '\t')

"""print('Inspecting the CC infos, we will have:')
print('Connected component 0: \n1659 SRVs')
print('Connected component 9: \n779 SRVs')
print('Connected components 51, 6, 15: \n582 SRVs')
print('Connected components 11, 8, 7, 52, 1, 2, 53, 5, 4, 19, 45, 35, 31: \n577 SRVs')
print(f'Connected components {set(ccs.CC_ID.to_list()) - set([0, 9, 51, 6, 15, 11, 8, 7, 52, 1, 2, 53, 5, 4, 19, 45, 35, 31])}: \n572 SRVs')"""

fold_1 = [0]
fold_2 = [9]
fold_3 = [51, 6, 15]
fold_4 = [11, 8, 7, 52, 1, 2, 53, 5, 4, 19, 45, 35, 31]
fold_5 = list(set(ccs.CC_ID.to_list()) - set([0, 9, 51, 6, 15, 11, 8, 7, 52, 1, 2, 53, 5, 4, 19, 45, 35, 31]))

folds = [fold_1, fold_2, fold_3, fold_4, fold_5]
i = 0
for f in folds:
    counter = 0
    for cc in f:
        counter += data[data.CC_ID == cc].shape[0]
    i+=1
    print(f'Fold {i}: {counter} SRVs')
print('-'*105)

fold_ids = []
i = 0
for f in folds:
    ids = []
    for cc in f:
        ids.extend(data[data.CC_ID == cc].Unique.to_list())
    fold_ids.append(ids)
    df = data[data['Unique'].isin(ids)]
    i += 1
    df.to_csv(f'fold_{i}.tsv', sep = '\t', index = False)
        