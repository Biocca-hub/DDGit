##########################################################################################################################
############################################## IMPORTING NECESSARY LIBRARIES #############################################
##########################################################################################################################
from tqdm import tqdm
from itertools import combinations
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Sampler
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping 
import numpy as np

##########################################################################################################################
###################################################### DEFINING PCC ######################################################
##########################################################################################################################

def pearson_corr(y_hat, y): 
    vx = y_hat - torch.mean(y_hat) # Dati predetti
    vy = y - torch.mean(y) # Dati reali
    return torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-8) 

##########################################################################################################################
################################################ DEFINING CUSTOM SAMPLER #################################################
##########################################################################################################################

class MinSizeSampler(Sampler):
    def __init__(self, split_ranges: list[tuple[int, int]], min_size: int) -> None:
        self.split_ranges = split_ranges
        self.min_size = min_size
 
    def __iter__(self):
        indices = []
        for start_idx, end_idx in self.split_ranges:
            split_size = end_idx - start_idx
            if self.min_size > split_size:
                raise ValueError(f"min_size {self.min_size} is larger than the size of the split {split_size}")
            sampled_indices = torch.randperm(split_size)[:self.min_size] + start_idx
            indices.append(sampled_indices)
        indices = torch.cat(indices)
        indices = indices[torch.randperm(len(indices))]
        return iter(indices.tolist())
 
    def __len__(self):
        return self.min_size * len(self.split_ranges)

class EarlyStoppingWithWarmup(EarlyStopping):
    def __init__(self, warmup=10, **kwargs):
        super().__init__(**kwargs)
        self.warmup = warmup
    
    def on_validation_end(self, trainer, pl_module):
        '''print(
            f"epoch={trainer.current_epoch}, "
            f"warmup={self.warmup}, "
            f"check_on_train={self._check_on_train_epoch_end}"
        )'''

        if (
            self._check_on_train_epoch_end
            or self._should_skip_check(trainer)
            or trainer.current_epoch < self.warmup
        ):
            return

        '''print("RUNNING EARLY STOP CHECK")'''
        self._run_early_stopping_check(trainer)

##########################################################################################################################
##################################################### DEFINING MODEL #####################################################
##########################################################################################################################

class MLP(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()

        self.train_loss_history = []
        self.val_loss_history = []
        self.train_pcc_history = []
        self.val_pcc_history = []
        self.current_epoch_train_losses = []
        self.current_epoch_train_pccs = []

        layers = []
        prev_dim = input_dim

        for h in hidden_dims: 
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU()) # Sigmoid PPC = 0.2810 loss = 1.2403, con ReLu è avg.PCC = 0.3470, invece avg loss = 1.1788
            layers.append(nn.Dropout(0.3))  # Ridurre overfitting
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers) 
        self.loss_fn = nn.L1Loss() # invece di MSE che penalizza errori grandi di più, mettiamo questo

    def forward(self, x): 
        return self.model(x).view(-1)

    # TRAIN 

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        pcc = pearson_corr(y_hat, y)

        self.current_epoch_train_losses.append(loss.detach())
        self.current_epoch_train_pccs.append(pcc.detach())

        return loss

    def on_train_epoch_start(self):
        self.current_epoch_train_losses = []
        self.current_epoch_train_pccs = []

    def on_train_epoch_end(self):

        epoch_loss = torch.stack(
            self.current_epoch_train_losses
        ).mean()

        epoch_pcc = torch.stack(
            self.current_epoch_train_pccs
        ).mean()

        self.train_loss_history.append(
            epoch_loss.item()
        )

        self.train_pcc_history.append(
            epoch_pcc.item()
        )

        self.log("train_loss_epoch", epoch_loss)
        self.log("train_pcc_epoch", epoch_pcc)

    # VALIDATION

    def on_validation_epoch_start(self): # Inizializzazione di liste per memorizzare predizioni e target durante la validazione
        self.val_preds = []
        self.val_targets = []

    def validation_step(self, batch, batch_idx): # Calcolo delle predizioni e memorizzazione di predizioni e target per il calcolo finale alla fine dell'epoca
        x, y = batch
        y_hat = self(x)

        self.val_preds.append(y_hat.detach())
        self.val_targets.append(y.detach())

    def on_validation_epoch_end(self): # Concatenazione di tutte le predizioni e target per calcolare loss e PCC su tutto il fold
        y_hat = torch.cat(self.val_preds)
        y = torch.cat(self.val_targets)

        loss = self.loss_fn(y_hat, y) 
        pcc = pearson_corr(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_pcc", pcc, prog_bar=True)

        self.val_loss_history.append(loss.item())
        self.val_pcc_history.append(pcc.item())


        '''print(
            f"Epoch {self.current_epoch} | "
            f"val_loss: {loss:.4f} | val_pcc: {pcc:.4f}"
        )'''

    # TEST

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_targets = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self.test_preds.append(y_hat.detach())
        self.test_targets.append(y.detach())


    def on_test_epoch_end(self):
        y_hat = torch.cat(self.test_preds)
        y = torch.cat(self.test_targets)

        loss = self.loss_fn(y_hat, y)
        pcc = pearson_corr(y_hat, y)

        self.log("test_loss", loss)
        self.log("test_pcc", pcc)
        self.test_outputs = {"predictions": y_hat.cpu(), "targets": y.cpu()}

        '''print(
            f"TEST | loss: {loss:.4f} | pcc: {pcc:.4f}"
        )'''

    # OPTIMIZER

    def configure_optimizers(self): 
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

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

##########################################################################################################################
############################################### DEFINING SPLITTING FOR CV RUNS ###########################################
##########################################################################################################################

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
        
        '''print(f'Training set: {training}\nValidation set: {val_test[0]}\nTest set: {val_test[1]}')
        print(f'Training set: {training}\nValidation set: {val_test[1]}\nTest set: {val_test[0]}')'''
        
    return training_cv, val_cv, test_cv
 
##########################################################################################################################
##################################################### DEFINING DATA LOADING ##############################################
##########################################################################################################################

training_idx, val_idx, test_idx = training_split([0,1,2,3,4])

xs = ['../features/tensors1/fold_1_X_tensor.pt',
        '../features/tensors1/fold_2_X_tensor.pt',
        '../features/tensors1/fold_3_X_tensor.pt',
        '../features/tensors1/fold_4_X_tensor.pt',
        '../features/tensors1/fold_5_X_tensor.pt']

ys = ['../features/tensors/fold_1_Y_tensor.pt',
        '../features/tensors/fold_2_Y_tensor.pt',
        '../features/tensors/fold_3_Y_tensor.pt',
        '../features/tensors/fold_4_Y_tensor.pt',
        '../features/tensors/fold_5_Y_tensor.pt']

srvs = ['../features/tensors/fold_1_SRVs_tensor.pt',
        '../features/tensors/fold_2_SRVs_tensor.pt',
        '../features/tensors/fold_3_SRVs_tensor.pt',
        '../features/tensors/fold_4_SRVs_tensor.pt',
        '../features/tensors/fold_5_SRVs_tensor.pt']
    
Xs = tensor_list(xs)
Ys = tensor_list(ys)

print(type(Xs))
print(len(Xs))

print("Xs[0] shape:", Xs[0].shape)
print("Ys[0] shape:", Ys[0].shape)

print("Xs[0][:5]:", Xs[0][:5])

if Ys[0].ndim > 0:
    print("Ys[0][:5]:", Ys[0][:5])
else:
    print("Ys[0] =", Ys[0])

def cv_run(hidden_dims, pat, max_ep, warm, output, scatter, output_folder):

    print("Xs[0].shape =", Xs[0].shape)
    print("Ys[0].shape =", Ys[0].shape)

    datasets = []
    for i, (X, Y) in enumerate(zip(Xs, Ys)):
        print(f"Creating dataset {i}")
        datasets.append(TensorDataset(X, Y))
        print(f"Created dataset {i}")
    print("All datasets created")

    test_pcc_list = [] # STORES PCCs
    test_loss_list = [] # STORES LOSSs
    preds = []
    targs = []

    for i in tqdm(range(20), desc = 'Training progress:'):

        tr_data = ConcatDataset([datasets[j] for j in training_idx[i]])
        
        tr_split_ranges = []
        offset = 0

        for j in training_idx[i]:
            dataset_size = len(datasets[j])
            tr_split_ranges.append((offset, offset + dataset_size))
            offset += dataset_size

        tr_min_size = min(end - start for start, end in tr_split_ranges)
        
        val_data = datasets[val_idx[i]]
        test_data = datasets[test_idx[i]] 
        
        #print(f'Training folds: {training_idx[i]}\nValidation fold: {val_idx[i]}\nTesting fold: {test_idx[i]}\n\n')
        
        train_loader = DataLoader(tr_data, batch_size=32, sampler=MinSizeSampler(tr_split_ranges, tr_min_size))
        val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32)    

        model = MLP(input_dim=Xs[0].shape[1],
                    hidden_dims=hidden_dims) 
        
        early_stop = EarlyStoppingWithWarmup(
            warmup=warm, 
            monitor="val_loss",
            patience= pat,
            min_delta=1e-4,
            mode="min",
            check_on_train_epoch_end=False
        )
        
        checkpoint = ModelCheckpoint(
                                    monitor="val_loss",
                                    mode="min",
                                    save_top_k=1
                                )
        print(early_stop._check_on_train_epoch_end)

        trainer = pl.Trainer(
                            max_epochs=max_ep,
                            callbacks=[early_stop, checkpoint],
                            enable_progress_bar=True,
                            num_sanity_val_steps=0,
                            log_every_n_steps=1
                        )
        
        trainer.fit(model, train_loader, val_loader)

        best_model = MLP.load_from_checkpoint(
                                            checkpoint.best_model_path,
                                            input_dim=Xs[0].shape[1],
                                            hidden_dims=hidden_dims
                                        )

        ckpt = torch.load(checkpoint.best_model_path)

        best_epoch = ckpt["epoch"]
        best_val_loss = checkpoint.best_model_score.item()

        with open(f"{output_folder}/best_epochs.txt", "a") as f:
            f.write(
                f"Run {i+1}, "
                f"Best epoch: {best_epoch}, "
                f"Best val_loss: {best_val_loss}\n"
                )

        results = trainer.test(
                            best_model,
                            dataloaders=test_loader
                        )

        test_loss_list.append(results[0]['test_loss'])
        test_pcc_list.append(results[0]['test_pcc'])

        preds.append(best_model.test_outputs["predictions"])
        targs.append(best_model.test_outputs["targets"])

        print("train_loss:", len(model.train_loss_history))
        print("val_loss:", len(model.val_loss_history))
        print("train_pcc:", len(model.train_pcc_history))
        print("val_pcc:", len(model.val_pcc_history))

        with open(f"{output_folder}/history_run_{i+1}.csv", "w") as f:

            f.write(
                "epoch,train_loss,val_loss,train_pcc,val_pcc\n"
            )

            for e in range(len(model.val_loss_history)):

                f.write(
                    f"{e+1},"
                    f"{model.train_loss_history[e]},"
                    f"{model.val_loss_history[e]},"
                    f"{model.train_pcc_history[e]},"
                    f"{model.val_pcc_history[e]}\n"
                )
        
    folds = list(range(1, len(test_pcc_list) + 1))

    to_out = [f'PCC: {",".join(f"{p}" for p in test_pcc_list)}',
              f'Loss: {",".join(f"{l}" for l in test_loss_list)}']

    with open(output, 'w') as writer:
        writer.write('\n'.join(to_out))
    
    
    with open(scatter, 'w') as writer:
        to_write = []
        for i in range(len(preds)):
            p = preds[i]
            t = targs[i]
            to_write.append(f'Predictions_run_{i+1}: {str(p.tolist())[1:-1]}')
            to_write.append(f'Targets_run_{i+1}: {str(t.tolist())[1:-1]}')
        writer.write('\n'.join(to_write))

all_Xs = torch.cat(tensor_list(xs))
all_Ys = torch.cat(tensor_list(ys))

def random_run(hidden_dims, pat, max_ep, warm,
               output, scatter, runs, output_folder):

    test_loss_list = []
    test_pcc_list = []
    preds = []
    targs = []

    for i in tqdm(range(runs), desc='Training progress:'):

        groups = torch.load(f'r_splits/split_{i+1}.pt')

        tot = all_Xs.shape[0]

        tr_size = 414 * 3
        val_size = int((tot - tr_size) / 2)
        test_size = val_size

        tr_i = []
        val_i = []
        test_i = []

        for g in groups:
            if len(tr_i) < tr_size:
                tr_i.extend(g)
            elif len(val_i) < val_size:
                val_i.extend(g)
            else:
                test_i.extend(g)

        tr_i = torch.tensor(tr_i)
        val_i = torch.tensor(val_i)
        test_i = torch.tensor(test_i)

        x_tr = all_Xs[tr_i]
        y_tr = all_Ys[tr_i]

        x_val = all_Xs[val_i]
        y_val = all_Ys[val_i]

        x_test = all_Xs[test_i]
        y_test = all_Ys[test_i]

        train_loader = DataLoader(
            TensorDataset(x_tr, y_tr),
            batch_size=32
        )

        val_loader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=32
        )

        test_loader = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=32
        )

        model = MLP(
            input_dim=all_Xs.shape[1],
            hidden_dims=hidden_dims
        )

        early_stop = EarlyStoppingWithWarmup(
            warmup=warm,
            monitor="val_loss",
            patience=pat,
            min_delta=1e-4,
            mode="min",
            check_on_train_epoch_end=False
        )

        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )

        trainer = pl.Trainer(
            max_epochs=max_ep,
            callbacks=[early_stop, checkpoint],
            enable_progress_bar=True,
            num_sanity_val_steps=0,
            log_every_n_steps=1
        )

        trainer.fit(model, train_loader, val_loader)

        best_model = MLP.load_from_checkpoint(
            checkpoint.best_model_path,
            input_dim=all_Xs.shape[1],
            hidden_dims=hidden_dims
        )

        ckpt = torch.load(checkpoint.best_model_path)

        best_epoch = ckpt["epoch"]
        best_val_loss = checkpoint.best_model_score.item()

        with open(f"{output_folder}/best_epochs.txt", "a") as f:
            f.write(
                f"Run {i+1}, "
                f"Best epoch: {best_epoch}, "
                f"Best val_loss: {best_val_loss}\n"
            )

        results = trainer.test(
            best_model,
            dataloaders=test_loader
        )

        test_loss_list.append(results[0]["test_loss"])
        test_pcc_list.append(results[0]["test_pcc"])

        preds.append(best_model.test_outputs["predictions"])
        targs.append(best_model.test_outputs["targets"])

        with open(f"{output_folder}/history_run_{i+1}.csv", "w") as f:

            f.write(
                "epoch,train_loss,val_loss,train_pcc,val_pcc\n"
            )

            for e in range(len(model.val_loss_history)):

                f.write(
                    f"{e+1},"
                    f"{model.train_loss_history[e]},"
                    f"{model.val_loss_history[e]},"
                    f"{model.train_pcc_history[e]},"
                    f"{model.val_pcc_history[e]}\n"
                )

    to_out = [
        f'PCC: {",".join(str(p) for p in test_pcc_list)}',
        f'Loss: {",".join(str(l) for l in test_loss_list)}'
    ]

    with open(output, "w") as writer:
        writer.write("\n".join(to_out))

    with open(scatter, "w") as writer:
        to_write = []

        for i in range(len(preds)):
            p = preds[i]
            t = targs[i]

            to_write.append(
                f'Predictions_run_{i+1}: {str(p.tolist())[1:-1]}'
            )

            to_write.append(
                f'Targets_run_{i+1}: {str(t.tolist())[1:-1]}'
            )

        writer.write("\n".join(to_write))