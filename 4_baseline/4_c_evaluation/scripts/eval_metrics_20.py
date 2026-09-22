import pandas as pd
import numpy as np
import ast
from sklearn.metrics import mean_squared_error

tr_file = '../../4_b_mlp/output/best_combination/all_predictions/train_predictions.tsv'
tr_df = pd.read_csv(tr_file, sep = '\t')
val_file = '../../4_b_mlp/output/best_combination/all_predictions/validation_predictions.tsv'
val_df = pd.read_csv(val_file, sep = '\t')
test_file = '../../4_b_mlp/output/best_combination/all_predictions/test_predictions.tsv'
test_df = pd.read_csv(test_file, sep = '\t')

final_df = pd.read_csv('../../4_b_mlp/output/best_combination/all_predictions/out.txt', sep = '\t')

#targets Y and predictions y
train_Y = []
train_y = []
validation_Y = []
validation_y = []
test_Y = []
test_y = []

best_tr_pccs = []
best_tr_mses = []
best_v_pccs = []
best_v_mses = []
best_test_pccs = []
best_test_mses = []

for i in range(20):

    tr_targ = ast.literal_eval(tr_df.loc[i, "best_targ"])
    train_Y.extend(tr_targ)
    tr_pred = ast.literal_eval(tr_df.loc[i, "best_pred"])
    train_y.extend(tr_pred)

    best_tr_pccs.append(np.corrcoef(tr_targ, tr_pred)[0, 1])
    best_tr_mses.append(mean_squared_error(tr_targ, tr_pred))

    v_targ = ast.literal_eval(val_df.loc[i, "best_targ"])
    validation_Y.extend(v_targ)
    v_pred = ast.literal_eval(val_df.loc[i, "best_pred"])
    validation_y.extend(v_pred)

    best_v_pccs.append(np.corrcoef(v_targ, v_pred)[0, 1])
    best_v_mses.append(mean_squared_error(v_targ, v_pred))

    test_targ = ast.literal_eval(test_df.loc[i, 'best_targ'])
    test_Y.extend(test_targ)
    test_pred = ast.literal_eval(test_df.loc[i, 'best_pred'])
    test_y.extend(test_pred)

    best_test_pccs.append(np.corrcoef(test_targ, test_pred)[0, 1])
    best_test_mses.append(mean_squared_error(test_targ, test_pred))

best_train_pcc = np.corrcoef(train_Y, train_y)[0, 1]
best_train_mse = mean_squared_error(train_Y, train_y)
best_validation_pcc = np.corrcoef(validation_Y, validation_y)[0, 1]
best_validation_mse = mean_squared_error(validation_Y, validation_y)
best_test_pcc = np.corrcoef(test_Y, test_y)[0, 1]
best_test_mse = mean_squared_error(test_Y, test_y)

train_Y = []
train_y = []
validation_Y = []
validation_y = []
test_Y = []
test_y = []

last_tr_pccs = []
last_tr_mses = []
last_v_pccs = []
last_v_mses =[]
last_test_pccs = []
last_test_mses = []

for i in range(20):

    tr_targ = ast.literal_eval(tr_df.loc[i, "last_target"])
    train_Y.extend(tr_targ)
    tr_pred = ast.literal_eval(tr_df.loc[i, "last_pred"])
    train_y.extend(tr_pred)

    last_tr_pccs.append(np.corrcoef(tr_targ, tr_pred)[0, 1])
    last_tr_mses.append(mean_squared_error(tr_targ, tr_pred))

    v_targ = ast.literal_eval(val_df.loc[i, "last_target"])
    validation_Y.extend(v_targ)
    v_pred = ast.literal_eval(val_df.loc[i, "last_pred"])
    validation_y.extend(v_pred)

    last_v_pccs.append(np.corrcoef(v_targ, v_pred)[0, 1])
    last_v_mses.append(mean_squared_error(v_targ, v_pred))

    test_targ = ast.literal_eval(test_df.loc[i, 'last_targ'])
    test_Y.extend(test_targ)
    test_pred = ast.literal_eval(test_df.loc[i, 'last_pred'])
    test_y.extend(test_pred)

    last_test_pccs.append(np.corrcoef(test_targ, test_pred)[0, 1])
    last_test_mses.append(mean_squared_error(test_targ, test_pred))

last_train_pcc = np.corrcoef(train_Y, train_y)[0, 1]
last_train_mse = mean_squared_error(train_Y, train_y)
last_validation_pcc = np.corrcoef(validation_Y, validation_y)[0, 1]
last_validation_mse = mean_squared_error(train_Y, train_y)
last_test_pcc = np.corrcoef(test_Y, test_y)[0, 1]
last_test_mse = mean_squared_error(test_Y, test_y)

epochs = pd.read_csv('../output/metrics.tsv', sep = '\t')
best_epoch = epochs['best_epoch']
last_epoch = epochs['last_epoch']

twenty = pd.DataFrame({'Run': [i for i in range(1,21)],
                         'best_epoch': best_epoch,
                         'PCC_tr_best': best_tr_pccs,
                         'MSE_tr_best': best_test_mses,
                         'PCC_v_best': best_v_pccs,
                         'MSE_v_best': best_v_mses,
                         'PCC_test_best': best_test_pccs,
                         'MSE_test_best': best_test_mses,
                         'last_epoch': last_epoch,
                         'PCC_tr_last': last_tr_pccs,
                         'MSE_tr_last': last_tr_mses,
                         'PCC_v_last': last_v_pccs,
                         'MSE_v_last': last_v_mses,
                         'PCC_test_last': last_test_pccs,
                         'MSE_test_last': last_test_mses})

print(twenty.head())

all_runs = pd.DataFrame({
    # Training metrics for best epoch
    'PCC_all_best_tr': [best_train_pcc],
    'PCC_mean_best_tr': [np.mean(best_tr_pccs)],
    'PCC_std_best_tr': [np.std(best_tr_pccs)],

    'MSE_all_best_tr': [best_train_mse],
    'MSE_mean_best_tr': [np.mean(best_tr_mses)],
    'MSE_std_best_tr': [np.std(best_tr_mses)],

    # Validation metrics for best epoch
    'PCC_all_best_v': [best_validation_pcc],
    'PCC_mean_best_v': [np.mean(best_v_pccs)],
    'PCC_std_best_v': [np.std(best_v_pccs)],

    'MSE_all_best_v': [best_validation_mse],
    'MSE_mean_best_v': [np.mean(best_v_mses)],
    'MSE_std_best_v': [np.std(best_v_mses)],

    # Test metrics for best epoch
    'PCC_all_best_test': [best_test_pcc],
    'PCC_mean_best_test': [np.mean(best_test_pccs)],
    'PCC_std_best_test': [np.std(best_test_pccs)],
    
    'MSE_all_best_test': [best_test_mse],
    'MSE_mean_best_test': [np.mean(best_test_mses)],
    'MSE_std_best_test': [np.std(best_test_mses)],

    # Training metrics for last epoch
    'PCC_all_last_tr': [last_train_pcc],
    'PCC_mean_last_tr': [np.mean(last_tr_pccs)],
    'PCC_std_last_tr': [np.std(last_tr_pccs)],

    'MSE_all_last_tr': [last_train_mse],
    'MSE_mean_last_tr': [np.mean(last_tr_mses)],
    'MSE_std_last_tr': [np.std(last_tr_mses)],

    # Validation metrics for last epoch
    'PCC_all_last_v': [last_validation_pcc],
    'PCC_mean_last_v': [np.mean(last_v_pccs)],
    'PCC_std_last_v': [np.std(last_v_pccs)],

    'MSE_all_last_v': [last_validation_mse],
    'MSE_mean_last_v': [np.mean(last_v_mses)],
    'MSE_std_last_v': [np.std(last_v_mses)],

    # Test metrics for last epoch
    'PCC_all_last_test': [last_test_pcc],
    'PCC_mean_last_test': [np.mean(last_test_pccs)],
    'PCC_std_last_test': [np.std(last_test_pccs)],

    'MSE_all_last_test': [last_test_mse],
    'MSE_mean_last_test': [np.mean(last_test_mses)],
    'MSE_std_last_test': [np.std(last_test_mses)]
})

print(all_runs)

twenty.to_csv('../output/metrics_20_runs.tsv', sep = '\t')
all_runs.to_csv('../output/overview_20_runs.tsv', sep = '\t')