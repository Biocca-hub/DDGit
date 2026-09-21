import pandas as pd
from tqdm import tqdm


# For loss_plot and pcc_plot, input are:
#   df extracted from History_run_i.csv files with columns: 'epoch', 'train_loss', 'val_loss', 'train_pcc', 'val_pcc'
#   figname (name of output png)
#   single (specifies if df refers to a single run or not)

from loss_pcc_plots import loss_plot 
from loss_pcc_plots import pcc_plot 

# Inputs are df from scat.txt file and figname
from scatter_plot import scatter_plot

# Collective history_run dataframe
epoch = []
train_loss = []
val_loss = []
train_pcc = []
val_pcc = []
runs = []
with tqdm(total=500, desc="Files") as pbar:

    for i in range(500):
        file = f'../../4_b_mlp/output/best_combination/random_split/history_run_{i+1}.csv'
        df = pd.read_csv(file)
        epoch.extend(df['epoch'])
        train_loss.extend(df['train_loss'])
        val_loss.extend(df['val_loss'])
        train_pcc.extend(df['train_pcc'])
        val_pcc.extend(df['val_pcc'])
        runs.extend([i+1]*df.shape[0])
        pbar.update(1)

df = pd.DataFrame({
                   'run': runs,
                   'epoch': epoch, 
                   'train_loss': train_loss, 
                   'val_loss': val_loss, 
                   'train_pcc': train_pcc, 
                   'val_pcc': val_pcc
                })

with tqdm(total=2, desc="Plots") as pbar:

    loss_plot(df, 'global_loss', single = False)
    pbar.update(1)
    pcc_plot(df, 'global_pcc', single = False)
    pbar.update(1)

file = '../../4_b_mlp/output/best_combination/random_split/scat.txt'

with open(file, 'r') as reader:
    ps = []
    ts = []
    for line in reader:
        if line.startswith('Predictions'):
            predictions = line.split(' ')[1:]
            ps.append([float(val[:-1]) for val in predictions])
            
        elif line.startswith('Targets'):
            targets = line.split(' ')[1:]
            ts.append([float(val[:-1]) for val in targets])
with tqdm(total=len(ps), desc="preds") as pbar:
    all_ps = []
    all_tgs = []
    for i in range(len(ps)):
        df = pd.DataFrame({'predictions': ps[i], 'targets': ts[i]})
        #figname = f'../output/best_plots/scatter_run_{i+1}'
        #scatter_plot(df, figname)
        all_ps.extend(ps[i])
        all_tgs.extend(ts[i])
        pbar.update(1)

df = pd.DataFrame({'predictions': all_ps, 'targets': all_tgs})
scatter_plot(df, f'../output/best_plots/scatter_500_runs')





