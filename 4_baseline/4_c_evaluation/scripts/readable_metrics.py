import pandas as pd 
'''
df = pd.read_csv('../output/metrics_20_runs.tsv', sep = '\t')
df = df.drop(['Unnamed: 0'], axis = 1)
splits = pd.read_csv('../output/splits.tsv', sep = '\t')
splits = splits.drop(['Unnamed: 0'], axis = 1)

pd.DataFrame({'run': df.Run, 
              'training': splits.Training,
              'validation': splits.Validation,
              'test': splits.Test,
              'best_epoch': df.best_epoch, 
              'PCC_tr_best': df.PCC_tr_best.apply(lambda x: round(x, 3)), 
              'MSE_tr_best': df.PCC_tr_best.apply(lambda x: round(x, 3)), 
              'PCC_v_best': df.PCC_v_best.apply(lambda x: round(x, 3)),
              'MSE_v_best': df.MSE_v_best.apply(lambda x: round(x, 3)), 
              'PCC_test_best': df.PCC_test_best.apply(lambda x: round(x, 3)), 
              'MSE_test_best': df.MSE_test_best.apply(lambda x: round(x, 3)), 
              'last_epoch': df.last_epoch,
              'PCC_tr_last': df.PCC_tr_last.apply(lambda x: round(x, 3)), 
              'MSE_tr_last': df.MSE_tr_last.apply(lambda x: round(x, 3)), 
              'PCC_v_last': df.PCC_v_last.apply(lambda x: round(x, 3)), 
              'MSE_v_last': df.MSE_v_last.apply(lambda x: round(x, 3)),
              'PCC_test_last':df.PCC_test_last.apply(lambda x: round(x, 3)), 
              'MSE_test_last': df.MSE_test_last.apply(lambda x: round(x, 3))}).to_csv('../output/readable_metrics.tsv', sep = '\t', index = False)
'''

df = pd.read_csv('../output/overview_20_runs.tsv', sep = '\t')

pd.DataFrame({'PCC_all_best_tr': df.PCC_all_best_tr.apply(lambda x: round(x, 3)), 
              'PCC_mean_best_tr': df.PCC_mean_best_tr.apply(lambda x: round(x, 3)), 
              'PCC_std_best_tr': df.PCC_std_best_tr.apply(lambda x: round(x, 3)),
              'MSE_all_best_tr': df.MSE_all_best_tr.apply(lambda x: round(x, 3)), 
              'MSE_mean_best_tr': df.MSE_mean_best_tr.apply(lambda x: round(x, 3)), 
              'MSE_std_best_tr': df.MSE_std_best_tr.apply(lambda x: round(x, 3)),
              'PCC_all_best_v': df.PCC_all_best_v.apply(lambda x: round(x, 3)), 
              'PCC_mean_best_v': df.PCC_mean_best_v.apply(lambda x: round(x, 3)), 
              'PCC_std_best_v': df.PCC_std_best_v.apply(lambda x: round(x, 3)), 
              'MSE_all_best_v': df.MSE_all_best_v.apply(lambda x: round(x, 3)),
              'MSE_mean_best_v': df.MSE_mean_best_v.apply(lambda x: round(x, 3)), 
              'MSE_std_best_v': df.MSE_std_best_v.apply(lambda x: round(x, 3)), 
              'PCC_all_best_test': df.PCC_all_best_test.apply(lambda x: round(x, 3)),
              'PCC_mean_best_test': df.PCC_mean_best_test.apply(lambda x: round(x, 3)), 
              'PCC_std_best_test': df.PCC_std_best_test.apply(lambda x: round(x, 3)), 
              'MSE_all_best_test': df.MSE_all_best_test.apply(lambda x: round(x, 3)),
              'MSE_mean_best_test': df.MSE_mean_best_test.apply(lambda x: round(x, 3)), 
              'MSE_std_best_test': df.MSE_std_best_test.apply(lambda x: round(x, 3)), 
              'PCC_all_last_tr': df.PCC_all_last_tr.apply(lambda x: round(x, 3)),
              'PCC_mean_last_tr': df.PCC_mean_last_tr.apply(lambda x: round(x, 3)), 
              'PCC_std_last_tr': df.PCC_std_last_tr.apply(lambda x: round(x, 3)), 
              'MSE_all_last_tr': df.MSE_all_last_tr.apply(lambda x: round(x, 3)),
              'MSE_mean_last_tr': df.MSE_mean_last_tr.apply(lambda x: round(x, 3)), 
              'MSE_std_last_tr': df.MSE_std_last_tr.apply(lambda x: round(x, 3)), 
              'PCC_all_last_v': df.PCC_all_last_v.apply(lambda x: round(x, 3)),
              'PCC_mean_last_v': df.PCC_mean_last_v.apply(lambda x: round(x, 3)), 
              'PCC_std_last_v': df.PCC_std_last_v.apply(lambda x: round(x, 3)), 
              'MSE_all_last_v': df.MSE_all_last_v.apply(lambda x: round(x, 3)),
              'MSE_mean_last_v': df.MSE_mean_last_v.apply(lambda x: round(x, 3)), 
              'MSE_std_last_v': df.MSE_std_last_v.apply(lambda x: round(x, 3)), 
              'PCC_all_last_test': df.PCC_all_last_test.apply(lambda x: round(x, 3)),
              'PCC_mean_last_test': df.PCC_mean_last_test.apply(lambda x: round(x, 3)), 
              'PCC_std_last_test': df.PCC_std_last_test.apply(lambda x: round(x, 3)), 
              'MSE_all_last_test': df.MSE_all_last_test.apply(lambda x: round(x, 3)),
              'MSE_mean_last_test': df.MSE_mean_last_test.apply(lambda x: round(x, 3)), 
              'MSE_std_last_test': df.MSE_std_last_test.apply(lambda x: round(x, 3))}).T.to_csv('../output/readable_overview.tsv', 
                                                                                                sep = '\t',
                                                                                                index = True, # column names are now indeces
                                                                                                header = False # row index is now column name
                                                                                                )