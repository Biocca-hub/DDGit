import pandas as pd

splits = '../output/20_split_pcc.tsv'
epoch = '../input/epoch_track.tsv'

df_s = pd.read_csv(splits, sep = '\t')
df_e = pd.read_csv(epoch, sep = '\t')

pd.DataFrame({'run': df_s.Run,
              'best_epoch': df_e.best_epoch,
              'PCC_training_best': df_s.PCC_tr_best,
              'PCC_validation_best': df_s.PCC_v_best,
              'Loss_validation_best': df_e.val_loss_best,
              'PCC_test_best': df_s.PCC_test_best, 
              'last_epoch': df_e.last_epoch,
              'PCC_training_last': df_s.PCC_tr_last,
              'PCC_validation_last': df_s.PCC_v_last,
              'Loss_validation_last': df_e.val_loss_last,
              'PCC_test_last': df_s.PCC_test_last}).to_csv('../output/metrics.tsv', sep = '\t', index = False)