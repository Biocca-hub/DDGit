import torch 
import pandas as pd 
import numpy as np

fw = pd.read_csv('../output/forward.tsv', sep = '\t')
rev = pd.read_csv('../output/reverse.tsv', sep = '\t')

fw_labels = list(fw['unique'])
rev_labels = list(rev['unique'])

torch.save(fw_labels, '../output/fw_labels.pt')
torch.save(rev_labels, '../output/rev_labels.pt')

all_labels = fw_labels.extend(rev_labels)

torch.save(all_labels, '../output/all_labels.pt')

Y_fw = torch.tensor(np.array(fw.ddg))
Y_rev = torch.tensor(np.array(rev.ddg))

Y_all = torch.cat((Y_fw, Y_rev), 0)

torch.save(Y_fw, '../output/fw_targets.pt')
torch.save(Y_rev, '../output/rev_targets.pt')
torch.save(Y_all, '../output/all_targets.pt')

s8338 = pd.concat([fw, rev], ignore_index = True)

s8338.to_csv('../output/S8338.tsv', sep = '\t', index = False)

fw_s = pd.read_csv('../output/fw_splits.tsv', sep = '\t')
rev_s = pd.read_csv('../output/rev_splits.tsv', sep = '\t')

splits = pd.concat([fw_s, rev_s], ignore_index = True)

splits.to_csv('../output/S8338_splits.tsv', sep = '\t', index = False)