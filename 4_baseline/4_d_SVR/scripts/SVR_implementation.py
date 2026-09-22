import torch
import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# TRAINING FOLDS: 2, 3, 4
# VALIDATION FOLD: 1
# TEST FOLD: 5

print("Evaluation of SVR:\ntrained on folds: 2,3,4\nvalidation: 1\ntest: 5\n\n")
x_training = ['../4_b_mlp//input/fold_2_X_tensor.pt',
        '../4_b_mlp//input/fold_3_X_tensor.pt',
        '../4_b_mlp//input/fold_4_X_tensor.pt']

x_validation = '../4_b_mlp/input/fold_1_X_tensor.pt'

x_test = '../4_b_mlp//input/fold_5_X_tensor.pt'


y_training = ['../4_b_mlp//input/fold_2_Y_tensor.pt',
        '../4_b_mlp//input/fold_3_Y_tensor.pt',
        '../4_b_mlp//input/fold_4_Y_tensor.pt']

y_validation = '../4_b_mlp//input/fold_1_Y_tensor.pt'

y_test = '../4_b_mlp//input/fold_5_Y_tensor.pt'

X_train = torch.cat([torch.load(f) for f in x_training], dim=0).numpy()
X_val = torch.load(x_validation).numpy()
X_test = torch.load(x_test).numpy()

Y_train = torch.cat([torch.load(f) for f in y_training], dim=0).numpy()
Y_val = torch.load(y_validation).numpy()
Y_test = torch.load(y_test).numpy()

best_pcc = -1
best_params = None
best_model = None

for C in [0.1, 1, 10, 100]:
    for epsilon in [0.001, 0.01, 0.1]:
        for gamma in ['scale', 0.01, 0.1, 1]:

            model = SVR(
                kernel='rbf',
                C=C,
                epsilon=epsilon,
                gamma=gamma
            )

            model.fit(X_train, Y_train)

            pred_val = model.predict(X_val)
            val_pcc = pearsonr(Y_val, pred_val)[0]

            if val_pcc > best_pcc:
                best_pcc = val_pcc
                best_params = (C, epsilon, gamma)
                best_model = model

print("Best PCC:", best_pcc)
print("Best params:", best_params)

pred_test = best_model.predict(X_test)

test_pcc = pearsonr(Y_test, pred_test)[0]
test_mse = mean_squared_error(Y_test, pred_test)

print("Test PCC:", test_pcc)
print("Test MSE:", test_mse)

print('\n'+'*'*20+'\n')

print("Evaluation of SVR:\ntrained on folds: 2,3,4\nvalidation: 1\ntest: 5\n\n")

'''
Training set: [0, 1, 4]
Validation set: 3
Test set: 2
'''

x_training = ['../4_b_mlp//input/fold_1_X_tensor.pt',
        '../4_b_mlp//input/fold_2_X_tensor.pt',
        '../4_b_mlp//input/fold_5_X_tensor.pt']

x_validation = '../4_b_mlp/input/fold_4_X_tensor.pt'

x_test = '../4_b_mlp//input/fold_3_X_tensor.pt'


y_training = ['../4_b_mlp//input/fold_1_Y_tensor.pt',
        '../4_b_mlp//input/fold_2_Y_tensor.pt',
        '../4_b_mlp//input/fold_5_Y_tensor.pt']

y_validation = '../4_b_mlp//input/fold_4_Y_tensor.pt'

y_test = '../4_b_mlp//input/fold_3_Y_tensor.pt'

X_train = torch.cat([torch.load(f) for f in x_training], dim=0).numpy()
X_val = torch.load(x_validation).numpy()
X_test = torch.load(x_test).numpy()

Y_train = torch.cat([torch.load(f) for f in y_training], dim=0).numpy()
Y_val = torch.load(y_validation).numpy()
Y_test = torch.load(y_test).numpy()

best_pcc = -1
best_params = None
best_model = None

for C in [0.1, 1, 10, 100]:
    for epsilon in [0.001, 0.01, 0.1]:
        for gamma in ['scale', 0.01, 0.1, 1]:

            model = SVR(
                kernel='rbf',
                C=C,
                epsilon=epsilon,
                gamma=gamma
            )

            model.fit(X_train, Y_train)

            pred_val = model.predict(X_val)
            val_pcc = pearsonr(Y_val, pred_val)[0]

            if val_pcc > best_pcc:
                best_pcc = val_pcc
                best_params = (C, epsilon, gamma)
                best_model = model

print("Best PCC:", best_pcc)
print("Best params:", best_params)

pred_test = best_model.predict(X_test)

test_pcc = pearsonr(Y_test, pred_test)[0]
test_mse = mean_squared_error(Y_test, pred_test)

print("Test PCC:", test_pcc)
print("Test MSE:", test_mse)

df = pd.DataFrame({
    'predictions': pred_test,
    'targets': Y_test
})

# ======================================
# FIGURA
# ======================================

fig, ax = plt.subplots(figsize=(8, 8))

scatter = sns.scatterplot(
    data=df,
    x="targets",
    y="predictions",
    color="navy",
    s=15,
    edgecolor='black',
    linewidth=0.1,
    alpha=0.08,
    legend=False,
    ax=ax
)

# ======================================
# LIMITI ASSI
# ======================================

min_val = min(
    df['targets'].min(),
    df['predictions'].min()
)

max_val = max(
    df['targets'].max(),
    df['predictions'].max()
)

pad = 0.05 * (max_val - min_val)

ax.set_xlim(min_val - pad, max_val + pad)
ax.set_ylim(min_val - pad, max_val + pad)

# ======================================
# RETTA y = x
# ======================================

ax.plot(
    [min_val, max_val],
    [min_val, max_val],
    color='blue',
    linestyle='--',
    linewidth=1,
    label='y = x'
)

# ======================================
# x = 0 e y = 0
# ======================================

ax.axvline(
    x=0,
    color='blue',
    linestyle='--',
    linewidth=1
)

ax.axhline(
    y=0,
    color='blue',
    linestyle='--',
    linewidth=1
)

# ======================================
# MEDIE
# ======================================

mean_target = df['targets'].mean()
mean_prediction = df['predictions'].mean()

mean_diag = (mean_target + mean_prediction) / 2

# Media sulla diagonale
ax.scatter(
    mean_diag,
    mean_diag,
    marker='*',
    s=120,
    color='yellow',
    edgecolor='black',
    linewidth=1.2,
    zorder=10,
    label='Mean on y=x'
)

# Media reale
ax.scatter(
    mean_target,
    mean_prediction,
    marker='*',
    s=120,
    color='red',
    edgecolor='black',
    linewidth=0.5,
    zorder=20,
    label='Mean'
)

# ======================================
# STILE
# ======================================

ax.set_aspect('equal', adjustable='box')

ax.set_xlabel("Target")
ax.set_ylabel("Prediction")
ax.set_title("Predictions vs Targets (SVR run 6/20)")

ax.legend(frameon=True)

plt.tight_layout()

plt.savefig(
    'scatter.png',
    dpi=600,
    bbox_inches='tight'
)
'''
pcc = test_pcc
mse = test_mse

plt.figure(figsize=(6,6))

plt.scatter(Y_test, pred_test, alpha=0.6)

# diagonale ideale y=x
mn = min(Y_test.min(), pred_test.min())
mx = max(Y_test.max(), pred_test.max())

plt.plot([mn, mx], [mn, mx], 'r--')

plt.xlabel("Experimental ΔΔG")
plt.ylabel("Predicted ΔΔG")
plt.title(f"Test Set\nPCC={pcc:.3f}  MSE={mse:.3f}")

plt.tight_layout()
plt.savefig("scatter_test_set6.png", dpi=300)
plt.close()

''''''print(X_train.shape)
print(Y_train.shape)

print(X_val.shape)
print(Y_val.shape)

print(X_test.shape)
print(Y_test.shape)'''

'''svr = SVR(
    kernel='rbf',
    C=1.0,
    epsilon=0.1
)

svr.fit(X_train, Y_train)

pred_val = svr.predict(X_val)
pred_test = svr.predict(X_test)

print("Val PCC:", pearsonr(Y_val, pred_val)[0])
print("Val MSE:", mean_squared_error(Y_val, pred_val))

print("Test PCC:", pearsonr(Y_test, pred_test)[0])
print("Test MSE:", mean_squared_error(Y_test, pred_test))'''
