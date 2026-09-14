import torch

xs = ['../input/fold_1_X_tensor.pt',
        '../input/fold_2_X_tensor.pt',
        '../input/fold_3_X_tensor.pt',
        '../input/fold_4_X_tensor.pt',
        '../input/fold_5_X_tensor.pt']

for i in xs:
    x = torch.load(i)
    print(x.shape)
    print(x[0][20:60])
    print(x[1][20:60])
    print(x[2][20:60])
    print(x[50][20:60])
    print()