import torch 
import pandas as pd 
import numpy as np 

X_fw = torch.load('../input/features.pt')

# baseline features:
#       np.array(X_fw[i][:61]) for i == i-th row
#       out of these, the 1st 20 positions are anti-symmetrical:
#               np.array(X_fw[i][:20])*-1

# positional deltas: 
#       np.array(X_fw[i][62:77] for i == i-th row
#       5 descriptors, for which we must switch 1 with 2, 2 with 1 and multiply 3 with -1:
#               X_fw[62] == X_rev [63] and X_fw[63] == X_rev[62] and X_rev[64] == -1 * X_fw[64]
#               X_fw[65] == X_rev [66] and X_fw[66] == X_rev[65] and X_rev[67] == -1 * X_fw[67]
#               X_fw[68] == X_rev [69] and X_fw[69] == X_rev[68] and X_rev[70] == -1 * X_fw[70]
#               X_fw[71] == X_rev [72] and X_fw[72] == X_rev[71] and X_rev[73] == -1 * X_fw[73]
#               X_fw[74] == X_rev [75] and X_fw[75] == X_rev[74] and X_rev[76] == -1 * X_fw[76]

# environment average: 
#       np.array(X_fw[i][77:] for i == i-th row stays the same, no way of doing surrounding for mutant

X_rev = X_fw.clone() # copia vera, X_fw resta intatto

# one-hot / baseline antisimmetrica
X_rev[:, :20] = -X_fw[:, :20]

# positional deltas: 5 descrittori a triplette (a, b, c) -> (b, a, -c)
for start in range(62, 77, 3):
    a, b, c = start, start + 1, start + 2
    X_rev[:, a] = X_fw[:, b]
    X_rev[:, b] = X_fw[:, a]
    X_rev[:, c] = -X_fw[:, c]

# colonne 77: invariate (già copiate da clone)

'''X_rev = X_fw

for i in range(4169):

    # one hot encoding:
    X_rev[i][:20] = X_fw[i][:20] * -1

    # positional delta:

    X_rev[i][62] = X_fw[i][63]
    X_rev[i][63] = X_fw[i][62]
    X_rev[i][64] = X_fw[i][64] * -1

    X_rev[i][62 + 3] = X_fw[i][63 + 3]
    X_rev[i][63 + 3] = X_fw[i][62 + 3]
    X_rev[i][64 + 3] = X_fw[i][64 + 3] * -1

    X_rev[i][62 + 6] = X_fw[i][63 + 6]
    X_rev[i][63 + 6] = X_fw[i][62 + 6]
    X_rev[i][64 + 6] = X_fw[i][64 + 6] * -1

    X_rev[i][62 + 9] = X_fw[i][63 + 9]
    X_rev[i][63 + 9] = X_fw[i][62 + 9]
    X_rev[i][64 + 9] = X_fw[i][64 + 9] * -1

    X_rev[i][62 + 12] = X_fw[i][63 + 12]
    X_rev[i][63 + 12] = X_fw[i][62 + 12]
    X_rev[i][64 + 12] = X_fw[i][64 + 12] * -1'''

torch.save(X_rev, '../output/rev_features.pt')
torch.save(X_fw, '../output/fw_features.pt')

X_all = torch.cat((X_fw, X_rev), 0)

#print(X_all.shape) 
#      torch.Size([8338, 127])

torch.save(X_all, '../output/all_features.pt')


