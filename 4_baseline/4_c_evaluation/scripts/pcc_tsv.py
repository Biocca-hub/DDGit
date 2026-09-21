import pandas as pd
import numpy as np
import ast

tr_file = '../../4_b_mlp/output/best_combination/all_predictions/train_predictions.tsv'
tr_df = pd.read_csv(tr_file, sep = '\t')

val_file = '../../4_b_mlp/output/best_combination/all_predictions/validation_predictions.tsv'
val_df = pd.read_csv(val_file, sep = '\t')

test_file = '../../4_b_mlp/output/best_combination/all_predictions/test_predictions.tsv'
test_df = pd.read_csv(test_file, sep = '\t')

train_pcc_best = []
train_pcc_last = []

val_pcc_best = []
val_pcc_last = []

test_pcc_best = []
test_pcc_last = []

for i in range(20):

    train_targets = ast.literal_eval(tr_df.loc[i, "best_targ"])
    train_predictions = ast.literal_eval(tr_df.loc[i, "best_pred"])
    
    train_pcc_best.append(np.corrcoef(train_targets, train_predictions)[0, 1])

    val_targets = ast.literal_eval(val_df.loc[i, "best_targ"])
    val_predictions = ast.literal_eval(val_df.loc[i, "best_pred"])

    val_pcc_best.append(np.corrcoef(val_targets, val_predictions)[0, 1])

    test_targets = ast.literal_eval(test_df.loc[i, 'best_targ'])
    test_predictions = ast.literal_eval(test_df.loc[i, 'best_pred'])
    
    test_pcc_best.append(np.corrcoef(test_targets, test_predictions)[0, 1])

    train_targets = ast.literal_eval(tr_df.loc[i, "last_target"])
    train_predictions = ast.literal_eval(tr_df.loc[i, "last_pred"])
    
    train_pcc_last.append(np.corrcoef(train_targets, train_predictions)[0, 1])

    val_targets = ast.literal_eval(val_df.loc[i, "last_target"])
    val_predictions = ast.literal_eval(val_df.loc[i, "last_pred"])

    val_pcc_last.append(np.corrcoef(val_targets, val_predictions)[0, 1])

    test_targets = ast.literal_eval(test_df.loc[i, 'last_targ'])
    test_predictions = ast.literal_eval(test_df.loc[i, 'last_pred'])
    
    test_pcc_last.append(np.corrcoef(test_targets, test_predictions)[0, 1])

on_plots = pd.DataFrame({'run': [i for i in range(1,21)],
                         'train_pcc_best_epoch': train_pcc_best,
                         'train_pcc_last_epoch': train_pcc_last,
                         'val_pcc_best_epoch': val_pcc_best,
                         'val_pcc_last_epoch': val_pcc_last,
                         'test_pcc_best_epoch': test_pcc_best,
                         'test_pcc_last_epoch': test_pcc_last})

on_plots.to_csv('../output/pcc_summary.tsv', sep = '\t', index = False)