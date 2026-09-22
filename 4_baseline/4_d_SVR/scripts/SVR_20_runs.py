import torch
import numpy as np 
import pandas as pd
import seaborn as sns
from itertools import combinations
from tqdm import tqdm
from itertools import product
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def pearson_corr(y_hat, y): 
    vx = y_hat - torch.mean(y_hat) # Dati predetti
    vy = y - torch.mean(y) # Dati reali
    return torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-8) 

def tensor_list(paths): 
    '''Takes file paths list, 
       imports file as tensor, 
       returns list of tensors'''
    tensors = []
    for i in range(len(paths)):
        t = torch.load(paths[i])
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        tensors.append(t)
    return tensors

def grid_split(folds):
    '''Takes a list of indeces,
       splits them in training (3 idx), 
       validation (1 idx), 
       test (1 idx). 
       Keeping the training set combination twice, 
       inverts validation into test set 
       and test into validation, 
       storing tensor lists in 3 lists ready for cv'''

    tetra = list(combinations(folds, 4))

    training_cv = []
    val_cv = []
    for t in tetra:
        val_test = list(set(folds)-set(t))
        training = list(t)
        training_cv.append(training)
        val_cv.append(val_test[0])
        
    return training_cv, val_cv

def training_split(folds):
    '''Takes a list of indeces,
       splits them in training (3 idx), 
       validation (1 idx), 
       test (1 idx). 
       Keeping the training set combination twice, 
       inverts validation into test set 
       and test into validation, 
       storing tensor lists in 3 lists ready for cv'''

    tris = list(combinations(folds, 3))

    training_cv = []
    val_cv = []
    test_cv = []
    for t in tris:
        val_test = list(set(folds)-set(t))
        training = list(t)
        training_cv.append(training)
        training_cv.append(training)
        val_cv.append(val_test[0])
        val_cv.append(val_test[1])
        test_cv.append(val_test[1])
        test_cv.append(val_test[0])
        
    return training_cv, val_cv, test_cv

training_idx, val_idx = grid_split([0,1,2,3,4])


xs = ['../4_b_mlp/input/fold_1_X_tensor.pt',
        '../4_b_mlp/input/fold_2_X_tensor.pt',
        '../4_b_mlp/input/fold_3_X_tensor.pt',
        '../4_b_mlp/input/fold_4_X_tensor.pt',
        '../4_b_mlp/input/fold_5_X_tensor.pt']

ys = ['../4_b_mlp/input/fold_1_Y_tensor.pt',
        '../4_b_mlp/input/fold_2_Y_tensor.pt',
        '../4_b_mlp/input/fold_3_Y_tensor.pt',
        '../4_b_mlp/input/fold_4_Y_tensor.pt',
        '../4_b_mlp/input/fold_5_Y_tensor.pt']

srvs = ['../4_b_mlp/input/fold_1_SRVs_tensor.pt',
        '../4_b_mlp/input/fold_2_SRVs_tensor.pt',
        '../4_b_mlp/input/fold_3_SRVs_tensor.pt',
        '../4_b_mlp/input/fold_4_SRVs_tensor.pt',
        '../4_b_mlp/input/fold_5_SRVs_tensor.pt']
    
#Xs = tensor_list(xs)
#Ys = tensor_list(ys)

#print(len(Xs))
#print(Ys)
"""
best_pcc = -1
best_params = None
best_model = None

total_runs = 4 * 3 * 4 * 5 # C * epsilon * gamma * CV 
with tqdm(total=total_runs, desc="Grid Search") as pbar:

    for C in [0.1, 1, 10, 100]:
        for epsilon in [0.001, 0.01, 0.1]:
            for gamma in ['scale', 0.01, 0.1, 1]:
                pcc_avg = []
                for i in range(5):
                        
                    X_train = torch.cat([torch.load(xs[j]) for j in training_idx[i]]).numpy()
                    
                    X_val = torch.load(xs[val_idx[i]]).numpy()
                    
                    Y_train = torch.cat([torch.load(ys[j]) for j in training_idx[i]]).numpy()
                    Y_val = torch.load(ys[val_idx[i]]).numpy()
                    
                    # Prova pesando i targets

                    #weights = np.ones(len(Y_train))

                    #weights[Y_train > 0] = 5.0

                    model = SVR(
                                kernel='rbf',
                                C=C,
                                epsilon=epsilon,
                                gamma=gamma
                            )

                    # Prova pesando i targets
                    model.fit(
                               X_train,
                               Y_train,
                               #sample_weight=weights
                                )

                    pred_val = model.predict(X_val)
                    val_pcc = pearsonr(Y_val, pred_val)[0]
                    pcc_avg.append(val_pcc)
                    
                    pbar.update(1)
                    

                mean_pcc = np.mean(pcc_avg)

                if mean_pcc > best_pcc:
                    best_pcc = mean_pcc
                    best_params = (C, epsilon, gamma)
                    best_model = model

print("Best PCC:", best_pcc)
print("Best params:", best_params)
"""
best_params= [10, 0.1, 0.01]
training_idx, val_idx, test_idx = training_split([0,1,2,3,4])

with tqdm(total=20, desc="Cross validation") as pbar:
    pccs = []
    losses = []
    predictions = []
    targets = []
    residui = []
    for i in range(20):
                            
        X_train = torch.cat([torch.load(xs[j]) for j in training_idx[i]]).numpy()
                        
        X_val = torch.load(xs[val_idx[i]]).numpy()
                        
        Y_train = torch.cat([torch.load(ys[j]) for j in training_idx[i]]).numpy()
        Y_val = torch.load(ys[val_idx[i]]).numpy()

        model = SVR(
                    kernel='rbf',
                    C=best_params[0],
                    epsilon=best_params[1],
                    gamma=best_params[2]
                    )

        model.fit(X_train, Y_train)

        pred_val = model.predict(X_val)
        val_pcc = pearsonr(Y_val, pred_val)[0]

        X_test = torch.load(xs[test_idx[i]]).numpy()

        Y_test = torch.load(ys[test_idx[i]]).numpy()
        
        pred_test = model.predict(X_test)

        test_pcc = pearsonr(Y_test, pred_test)[0]
        test_mse = mean_squared_error(Y_test, pred_test)
        losses.append(test_mse)
        pccs.append(test_pcc)
        predictions.append(pred_test)
        targets.append(Y_test)
        residui.append(np.mean(pred_test - Y_test))
                        
        pbar.update(1)

print("Test PCC:", np.mean(pccs), "±", np.std(pccs))
print("Test MSE:", np.mean(losses),"±", np.std(losses))
print("Average residual:", np.mean(residui))

"""
with tqdm(total=20, desc="Scatter plots") as pbar:
    for i in range(len(predictions)):
        df = pd.DataFrame({'targets': targets[i], 'predictions': predictions[i]})
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
            f'scatters/no_weights/scatter{i+1}.png',
            dpi=600,
            bbox_inches='tight'
        )
        plt.close()
        pbar.update(1)
"""    
all_res = np.concatenate(
    [pred - true for pred, true in zip(predictions, targets)]
)

plt.hist(all_res, bins=50)
plt.axvline(0, color="red")
plt.savefig(
            f'residuals.png',
            dpi=600,
            bbox_inches='tight'
        )
plt.close()

all_pred = np.concatenate(predictions)
all_true = np.concatenate(targets)

plt.scatter(all_true, all_pred, alpha=0.2)

m = min(all_true.min(), all_pred.min())
M = max(all_true.max(), all_pred.max())

plt.plot([m, M], [m, M], "r--")
plt.xlabel("True")
plt.ylabel("Pred")
plt.savefig(
            'scatters/no_weights/scatter.png',
            dpi = 600,
            bbox_inches = 'tight'
            )
plt.close()
