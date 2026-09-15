# IMPORTING NECESSARY LIBRARIES 

from tqdm import tqdm
from itertools import combinations
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Sampler
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping 
import numpy as np

# DEFINING PCC 

def pearson_corr(y_hat, y): 
    vx = y_hat - torch.mean(y_hat) # predictions
    vy = y - torch.mean(y) # targets
    return torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-8) 

# DEFINING CUSTOM SAMPLER 

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

        if (
            self._check_on_train_epoch_end
            or self._should_skip_check(trainer)
            or trainer.current_epoch < self.warmup
        ):
            return

        self._run_early_stopping_check(trainer)
        
# DEFINING MODEL 

class MLP(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims, activation, dpout, loss_f):
        super().__init__()

        self.train_loss_history = []
        self.val_loss_history = []
        self.train_pcc_history = []
        self.val_pcc_history = []
        self.current_epoch_train_losses = []
        self.current_epoch_train_pccs = []
        self.best_val_loss = float("inf")
        self.best_train_preds = None
        self.best_train_targets = None
        self.best_val_preds = None
        self.best_val_targets = None
        self.last_train_preds = None
        self.last_train_targets = None
        
        layers = []

        # Starting: input layer == input size
        prev_dim = input_dim

        for h in hidden_dims: 
            layers.append(nn.Linear(prev_dim, h))
            # activation function is parametrized ==> <activation>
            layers.append(nn.ReLU() if activation == "relu" else nn.Sigmoid())
            # Dropout is parametrized ==> <dpout>
            layers.append(nn.Dropout(dpout))
            prev_dim = h

        # Output layer = 1 dimension
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers) 
        # Loss function is parametrized ==> <loss_f>
        self.loss_fn = nn.L1Loss() if loss_f == 'L1' else nn.MSELoss() 

    def forward(self, x): 
        return self.model(x).view(-1)

    # TRAIN 

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        pcc = pearson_corr(y_hat, y)
        self.train_preds_epoch.append(y_hat.detach())
        self.train_targets_epoch.append(y.detach())
        self.current_epoch_train_losses.append(loss.detach())
        self.current_epoch_train_pccs.append(pcc.detach())

        return loss

    def on_train_epoch_start(self):
        self.current_epoch_train_losses = []
        self.current_epoch_train_pccs = []

        self.train_preds_epoch = []
        self.train_targets_epoch = []

    def on_train_epoch_end(self):

        epoch_loss = torch.stack(
                                self.current_epoch_train_losses
                                ).mean()

        epoch_pcc = torch.stack(
                                self.current_epoch_train_pccs
                                ).mean()

        self.train_loss_history.append(epoch_loss.item())
        self.train_pcc_history.append(epoch_pcc.item())

        self.last_train_preds = torch.cat(self.train_preds_epoch)
        self.last_train_targets = torch.cat(self.train_targets_epoch)

        self.log("train_loss_epoch", epoch_loss)
        self.log("train_pcc_epoch", epoch_pcc)
        
    
    # VALIDATION

    def on_validation_epoch_start(self): 
        self.val_preds = []
        self.val_targets = []

    def validation_step(self, batch, batch_idx): 
        x, y = batch
        y_hat = self(x)

        self.val_preds.append(y_hat.detach())
        self.val_targets.append(y.detach())

    def on_validation_epoch_end(self): 
        y_hat = torch.cat(self.val_preds)
        y = torch.cat(self.val_targets)

        loss = self.loss_fn(y_hat, y) 
        pcc = pearson_corr(y_hat, y)

        if loss.item() < self.best_val_loss:
            self.best_val_loss = loss.item()

            self.best_val_preds = y_hat.detach()
            self.best_val_targets = y.detach()

            self.best_train_preds = self.last_train_preds.detach()
            self.best_train_targets = self.last_train_targets.detach()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_pcc", pcc, prog_bar=True)

        self.val_loss_history.append(loss.item())
        self.val_pcc_history.append(pcc.item())

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
        self.test_outputs = {"predictions": y_hat, "targets": y}

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

xs = ['../input/fold_1_X_tensor.pt',
        '../input/fold_2_X_tensor.pt',
        '../input/fold_3_X_tensor.pt',
        '../input/fold_4_X_tensor.pt',
        '../input/fold_5_X_tensor.pt']

ys = ['../input/fold_1_Y_tensor.pt',
        '../input/fold_2_Y_tensor.pt',
        '../input/fold_3_Y_tensor.pt',
        '../input/fold_4_Y_tensor.pt',
        '../input/fold_5_Y_tensor.pt']

srvs = ['../input/fold_1_SRVs_tensor.pt',
        '../input/fold_2_SRVs_tensor.pt',
        '../input/fold_3_SRVs_tensor.pt',
        '../input/fold_4_SRVs_tensor.pt',
        '../input/fold_5_SRVs_tensor.pt']
    
Xs = tensor_list(xs)
Ys = tensor_list(ys)

'''print(type(Xs))
print(len(Xs))

print("Xs[0] shape:", Xs[0].shape)
print("Ys[0] shape:", Ys[0].shape)

print("Xs[0][:5]:", Xs[0][:5])

if Ys[0].ndim > 0:
    print("Ys[0][:5]:", Ys[0][:5])
else:
    print("Ys[0] =", Ys[0])'''

def cv_run(hidden_dims, pat, max_ep, warm, output, scatter, output_folder, activation, dpout, loss_f):

    #print("Xs[0].shape =", Xs[0].shape)
    #print("Ys[0].shape =", Ys[0].shape)

    datasets = []
    for i, (X, Y) in enumerate(zip(Xs, Ys)):
        #print(f"Creating dataset {i}")
        datasets.append(TensorDataset(X, Y))
        #print(f"Created dataset {i}")
    #print("All datasets created")

    test_pcc_list = [] # STORES PCCs
    test_loss_list = [] # STORES LOSSs
    test_preds = []
    test_targs = []

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
                    hidden_dims=hidden_dims,
                    activation=activation,
                    dpout=dpout,
                    loss_f=loss_f,
                    ) 
        
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

        with open(f"{output_folder}/train_val_run_{i+1}.txt", "w") as f:

            f.write(
                f"BestTrainPredictions: "
                f"{str(model.best_train_preds.tolist())[1:-1]}\n"
            )

            f.write(
                f"BestTrainTargets: "
                f"{str(model.best_train_targets.tolist())[1:-1]}\n"
            )

            f.write(
                f"BestValidationPredictions: "
                f"{str(model.best_val_preds.tolist())[1:-1]}\n"
            )

            f.write(
                f"BestValidationTargets: "
                f"{str(model.best_val_targets.tolist())[1:-1]}\n"
            )

        best_model = MLP.load_from_checkpoint(
                                            checkpoint.best_model_path,
                                            input_dim=Xs[0].shape[1],
                                            hidden_dims=hidden_dims,
                                            activation=activation,
                                            dpout=dpout,
                                            loss_f=loss_f

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

        test_preds.append(best_model.test_outputs["predictions"])
        test_targs.append(best_model.test_outputs["targets"])

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

    with open(output, "w") as f:
        f.write("PCC\tLoss\n")
        for pcc, loss in zip(test_pcc_list, test_loss_list):
            f.write(f"{pcc}\t{loss}\n")

    #with open(output, 'w') as writer:
    #    writer.write('\n'.join(to_out))
    
    
    with open(scatter, 'w') as writer:
        to_write = []
        for i in range(len(test_preds)):
            p = test_preds[i]
            t = test_targs[i]
            to_write.append(f'Predictions_run_{i+1}: {str(p.tolist())[1:-1]}')
            to_write.append(f'Targets_run_{i+1}: {str(t.tolist())[1:-1]}')
        writer.write('\n'.join(to_write))
    
    pcc_max = np.max(test_pcc_list)
    pcc_min = np.min(test_pcc_list)
    pcc_mean = np.mean(test_pcc_list)
    pcc_std = np.std(test_pcc_list)

    return pcc_max, pcc_min, pcc_mean, pcc_std 