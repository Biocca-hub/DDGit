import pandas as pd 

with open('../output/best_combination/all_predictions/best_epochs.txt', 'r') as reader:
    run = []
    best_epoch = [] 
    val_loss_best = []
    for line in reader:
        r = line.split(',')
    
        run.append(int(r[0].split(' ')[-1]))

        best_epoch.append(int(r[1].split(': ')[-1]))

        val_loss_best.append(float(r[2].split(': ')[-1]))

history = pd.read_csv('../output/best_combination/all_predictions/complete_history.tsv', sep = '\t')
last_epoch = []
val_loss_last = []
for i in range(1,21):
    df = history[history['run'] == i]
    j = df.shape[0]
    last_epoch.append(j)
    val_loss_last.append(df['val_loss'].to_list()[-1])

pd.DataFrame({'run': run,
              'best_epoch': best_epoch,
              'val_loss_best': val_loss_best,
              'last_epoch': last_epoch,
              'val_loss_last': val_loss_last}).to_csv('../output/best_combination/all_predictions/epoch_track.tsv', sep = '\t')