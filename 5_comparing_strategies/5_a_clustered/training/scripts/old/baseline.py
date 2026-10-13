# ============================================================
# IMPORTS
# ============================================================

import torch
import torch.nn as nn
import pytorch_lightning as pl
from lightning.pytorch.callbacks import EarlyStopping

# ============================================================
# PEARSON CORRELATION COEFFICIENT
# ============================================================

def pearson_corr(y_hat, y):
    """
    Compute the Pearson Correlation Coefficient (PCC) between
    model predictions and target values.

    Parameters:

    y_hat : torch.Tensor
        Predicted values.
    y : torch.Tensor
        Ground-truth target values.

    Returns

    torch.Tensor
        Pearson correlation coefficient in the range [-1, 1].

    A small constant (1e-8) is added to the denominator to avoid
    division by zero when one of the two vectors has zero variance.
    """

    vx = y_hat - torch.mean(y_hat)
    vy = y - torch.mean(y)

    return torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx**2))
        * torch.sqrt(torch.sum(vy**2))
        + 1e-8
    )

# ============================================================
# EARLY STOPPING WITH WARMUP
# ============================================================

class EarlyStoppingWithWarmup(EarlyStopping):
    """
    Early stopping callback with an initial warmup period.
    During the first `warmup` epochs, early stopping is disabled,
    allowing the model enough time to start learning before the
    monitored validation metric is used to stop training.
    """

    def __init__(self, warmup=10, **kwargs):
        """
        Initialize the early stopping callback and store the
        number of warmup epochs.
        """
        super().__init__(**kwargs)
        self.warmup = warmup

    def on_validation_end(self, trainer, pl_module):
        """
        Run the early stopping check after validation.

        The check is skipped if:
        - early stopping is configured to run at train epoch end;
        - Lightning determines that the check should be skipped;
        - the current epoch is still within the warmup period.

        After the warmup period, standard early stopping is applied.
        """

        if (
            self._check_on_train_epoch_end
            or self._should_skip_check(trainer)
            or trainer.current_epoch < self.warmup
        ):
            return

        self._run_early_stopping_check(trainer)

# ============================================================
# MULTILAYER PERCEPTRON
# ============================================================

class MLP(pl.LightningModule):
    """
    Multilayer Perceptron for a regression task.

    The network architecture is dynamically constructed according
    to the specified hidden dimensions, activation function and
    dropout probability.

    The model tracks loss and Pearson correlation for training and
    validation and stores predictions corresponding to the epoch
    with the best validation loss.

    Parameters
    
    input_dim : int
        Number of input features.

    hidden_dims : list[int]
        Number of neurons in each hidden layer.
        Example: [128, 64, 32].

    activation : str
        Activation function. Supported values:
        "relu" or "sigmoid".

    dpout : float
        Dropout probability applied after each hidden layer.

    loss_f : str
        Regression loss function.
        "L1" selects MAE (L1Loss), otherwise MSELoss is used.
    """

    def __init__(self,input_dim,hidden_dims,activation,dpout,loss_f):
        """
        Initialize metric histories, prediction storage,
        neural network architecture and loss function.
        """

        super().__init__()

        # --------------------------------------------------------
        # METRIC HISTORIES
        # --------------------------------------------------------

        # Store one loss value per epoch.
        self.train_loss_history = []
        self.val_loss_history = []

        # Store one PCC value per epoch.
        self.train_pcc_history = []
        self.val_pcc_history = []

        # Best validation loss observed so far.
        self.best_val_loss = float("inf")

        # --------------------------------------------------------
        # PREDICTION STORAGE
        # --------------------------------------------------------

        # Predictions/targets corresponding to the epoch with
        # the lowest validation loss.
        self.best_train_preds = None
        self.best_train_targets = None

        self.best_val_preds = None
        self.best_val_targets = None

        # Training predictions from the most recently completed epoch.
        self.last_train_preds = None
        self.last_train_targets = None

        # --------------------------------------------------------
        # MODEL ARCHITECTURE
        # --------------------------------------------------------

        layers = []

        # Number of features entering the first hidden layer.
        prev_dim = input_dim

        for h in hidden_dims:

            # Fully connected layer.
            layers.append(nn.Linear(prev_dim, h))

            # Activation function.
            layers.append(nn.ReLU() if activation == "relu" else nn.Sigmoid())

            # Dropout regularization.
            layers.append(nn.Dropout(dpout))

            # Output size of the current layer becomes the
            # input size of the following layer.
            prev_dim = h

        # Final regression layer: one continuous output.
        layers.append(nn.Linear(prev_dim, 1))

        # Combine all layers into a sequential model.
        self.model = nn.Sequential(*layers)

        # --------------------------------------------------------
        # LOSS FUNCTION
        # --------------------------------------------------------

        # L1 = Mean Absolute Error.
        # Otherwise use Mean Squared Error.
        self.loss_fn = (nn.L1Loss() if loss_f == "L1" else nn.MSELoss())

    # ============================================================
    # FORWARD PASS
    # ============================================================

    def forward(self, x):
        """
        Perform the forward pass through the neural network.

        Parameters
        
        x : torch.Tensor
            Input feature matrix with shape [N, input_dim].

        Returns
        
        torch.Tensor
            Predictions with shape [N].
        
        The final linear layer returns shape [N, 1].
        `.view(-1)` converts it to [N] to match the target shape.
        """

        return self.model(x).view(-1)

    # ============================================================
    # TRAINING
    # ============================================================

    def on_train_epoch_start(self):
        """
        Initialize containers at the beginning of each training epoch.

        Predictions and targets from individual batches are collected
        and later concatenated so that training loss and PCC can be
        computed over the entire training set.

        Computing PCC globally is preferable to averaging PCC values
        calculated independently for individual batches.
        """

        self.train_preds_epoch = []
        self.train_targets_epoch = []

    def training_step(self, batch, batch_idx):
        """
        Perform one training step on a single batch.

        The method:
        1. extracts input features and targets;
        2. performs the forward pass;
        3. computes the optimization loss;
        4. stores detached predictions and targets for epoch-level
           metric calculation;
        5. returns the loss used by Lightning for backpropagation.

        Parameters
        
        batch : tuple
            Tuple containing input features (x) and targets (y).

        batch_idx : int
            Index of the current batch.

        Returns
        
        torch.Tensor
            Loss used for gradient computation and parameter updates.
        """

        x, y = batch

        # Forward pass.
        y_hat = self(x)

        # Loss used for optimization.
        loss = self.loss_fn(y_hat, y)

        # Store predictions without keeping the computational graph.
        self.train_preds_epoch.append(y_hat.detach())

        self.train_targets_epoch.append(y.detach())

        return loss

    def on_train_epoch_end(self):
        """
        Compute training metrics over the entire training set.

        Predictions and targets collected from all batches are
        concatenated before computing loss and Pearson correlation.

        This ensures that train PCC is calculated in the same way
        as validation and test PCC.

        The method also stores predictions from the current epoch
        so they can later be associated with the best validation epoch.
        """

        # Concatenate all training batches.
        train_preds = torch.cat(self.train_preds_epoch)

        train_targets = torch.cat(self.train_targets_epoch)

        # Compute metrics over the entire training set.
        epoch_loss = self.loss_fn(train_preds, train_targets)

        epoch_pcc = pearson_corr(train_preds, train_targets)

        # Store metric history.
        self.train_loss_history.append(epoch_loss.item())

        self.train_pcc_history.append(epoch_pcc.item())

        # Store predictions from the current epoch.
        self.last_train_preds = train_preds
        self.last_train_targets = train_targets

        # Make metrics available to Lightning.
        self.log("train_loss_epoch", epoch_loss)

        self.log("train_pcc_epoch", epoch_pcc, prog_bar=True)

    # ============================================================
    # VALIDATION
    # ============================================================

    def on_validation_epoch_start(self):
        """
        Initialize containers for validation predictions and targets.

        They are reset at the beginning of every validation epoch.
        """

        self.val_preds = []
        self.val_targets = []

    def validation_step(self, batch, batch_idx):
        """
        Perform inference on one validation batch.

        Predictions and targets are stored and later concatenated
        to compute validation metrics over the complete validation set.

        Parameters
        
        batch : tuple
            Input features and target values.

        batch_idx : int
            Index of the current validation batch.
        """

        x, y = batch

        y_hat = self(x)

        self.val_preds.append(y_hat.detach())

        self.val_targets.append(y.detach())

    def on_validation_epoch_end(self):
        """
        Compute validation metrics over the complete validation set.

        The method calculates validation loss and Pearson correlation.

        If the current validation loss is lower than the best loss
        observed so far, predictions and targets from both training
        and validation are stored as the best epoch results.

        Validation metrics are also logged and added to their
        corresponding history lists.
        """

        # Concatenate validation batches.
        y_hat = torch.cat(self.val_preds)

        y = torch.cat(self.val_targets)

        # Global validation metrics.
        loss = self.loss_fn(y_hat, y)

        pcc = pearson_corr(y_hat, y)

        # Save results if validation loss improved.
        if loss.item() < self.best_val_loss:

            self.best_val_loss = loss.item()

            self.best_val_preds = y_hat.detach()
            self.best_val_targets = y.detach()

            # Training predictions corresponding to
            # the same epoch.
            self.best_train_preds = (self.last_train_preds.detach())

            self.best_train_targets = (self.last_train_targets.detach())

        # Lightning logging.
        self.log("val_loss", loss, prog_bar=True)

        self.log("val_pcc", pcc, prog_bar=True)

        # Save metric history.
        self.val_loss_history.append(loss.item())

        self.val_pcc_history.append(pcc.item())

    # ============================================================
    # TESTING
    # ============================================================

    def on_test_epoch_start(self):
        """
        Initialize containers for test predictions and targets.

        They are reset before each test epoch.
        """

        self.test_preds = []
        self.test_targets = []

    def test_step(self, batch, batch_idx):
        """
        Perform inference on one test batch.

        Predictions and targets are collected and later used to
        compute metrics over the complete test set.

        Parameters
        
        batch : tuple
            Input features and target values.

        batch_idx : int
            Index of the current test batch.
        """

        x, y = batch

        y_hat = self(x)

        self.test_preds.append(y_hat.detach())

        self.test_targets.append(y.detach())

    def on_test_epoch_end(self):
        """
        Compute final metrics over the complete test set.

        All test batches are concatenated before computing loss
        and Pearson correlation.

        Predictions and targets are stored in `self.test_outputs`
        for subsequent analysis.
        """

        y_hat = torch.cat(self.test_preds)

        y = torch.cat(self.test_targets)

        # Test metrics.
        loss = self.loss_fn(y_hat, y)

        pcc = pearson_corr(y_hat, y)

        # Log test performance.
        self.log("test_loss", loss)

        self.log("test_pcc", pcc)

        # Store outputs for later analysis.
        self.test_outputs = {"predictions": y_hat, "targets": y}

    # ============================================================
    # OPTIMIZER
    # ============================================================

    def configure_optimizers(self):
        """
        Configure the optimizer used for model training.

        AdamW is used with:
        - learning rate = 1e-3
        - weight decay = 1e-4

        Weight decay provides L2-like regularization and can help
        reduce overfitting.

        Returns
        
        torch.optim.AdamW
            Optimizer used by Lightning during training.
        """

        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)