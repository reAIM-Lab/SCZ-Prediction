import os
import numpy as np
import pandas as pd
from collections import Counter
import torch
from sklearn.metrics import *
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import sys

from models import *
from losses import *

def downsize_batches(loader, new_batchsize):
    list_pids = []
    list_xs = []
    list_padding_mask = []
    list_ymask = []
    list_labels = []
    
    for pids, signals, padding_mask, ymask, labels in loader:
        list_pids.append(pids)
        list_xs.append(signals)
        list_padding_mask.append(padding_mask)
        list_ymask.append(ymask)
        list_labels.append(labels)
        
    dataset_pids = np.concatenate(list_pids)
    dataset_x = np.concatenate(list_xs)
    dataset_padding = np.concatenate(list_padding_mask)
    dataset_ymask = np.concatenate(list_ymask)
    dataset_labels = np.concatenate(list_labels)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(dataset_pids), torch.from_numpy(dataset_x), torch.from_numpy(dataset_padding), torch.from_numpy(dataset_ymask), torch.from_numpy(dataset_labels))
    smaller_test_loader = torch.utils.data.DataLoader(dataset, batch_size=new_batchsize)
    return smaller_test_loader

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

def train(train_loader, model, device, optimizer, criterion, loss_regularization_dict, skip_train):
    model.to(device)
    model.train()

    epoch_loss = 0
    list_training_loss = []
    
    true_ys = []
    pred_ys = []
    for i, batch in enumerate(train_loader):
        model.train()
        pids, signals, padding_mask, ymask, labels = batch
        ymask = get_skiptrain(ymask, skip_train)
        
        optimizer.zero_grad()
        signals, padding_mask, ymask, labels = signals.to(device, non_blocking=True), padding_mask.to(device, non_blocking=True), ymask.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(signals, padding_mask, ymask)
        
        labels_long = make_labels_long(labels, ymask)# make labels repeat for seq len
        loss = criterion(outputs.squeeze(), labels_long.squeeze())

        if loss_regularization_dict is not None:
            model.eval()
            regularization_term = calculate_regularization(batch, outputs, model, loss_regularization_dict)
            model.train()
            loss += loss_regularization_dict['lmbda']*regularization_term

        epoch_loss += loss.item()
        loss.backward()
        list_training_loss.append(loss.item())
        optimizer.step()
        
        torch.cuda.synchronize()
        
        true_ys.append(labels_long.detach().cpu().numpy())
        pred_ys.append(outputs.detach().cpu().numpy())


    true_ys_flattened = np.concatenate(true_ys).ravel()
    pred_ys_flattened = np.concatenate(pred_ys).ravel()
    pred_labels = (pred_ys_flattened>0.5)*1
    
    auc_train = roc_auc_score(true_ys_flattened, pred_ys_flattened)
    f1_train = f1_score(true_ys_flattened, pred_labels)
    auprc_train = average_precision_score(true_ys_flattened, pred_ys_flattened)
    correct_label = accuracy_score(true_ys_flattened, pred_labels)
    
    training_output_dict = {'auroc': auc_train, 'f1': f1_train, 'auprc': auprc_train, 'accuracy': correct_label, 'loss': np.mean(list_training_loss)}

    return training_output_dict

def test(test_loader, model, device, criterion, loss_regularization_dict, skip_train):
    model = model.to(device)
    model.eval()

    total_loss = 0
    list_testing_loss = []
    true_ys = []
    pred_ys = []
    list_pids = []
    list_ymask = []

    for i, batch in enumerate(test_loader):    
        pids, signals, padding_mask, ymask, labels = batch
        signals, padding_mask, ymask, labels = signals.to(device, non_blocking=True), padding_mask.to(device, non_blocking=True), ymask.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        ymask = get_skiptrain(ymask, skip_train)
        
        outputs = model(signals, padding_mask, ymask)
        labels_long = make_labels_long(labels, ymask)
        pids_long = make_labels_long(pids, ymask.to('cpu'))
        ymask_long = ymask.reshape(-1)[ymask.reshape(-1) > 0]
        loss = criterion(outputs.squeeze(), labels_long.squeeze())
        
        if loss_regularization_dict is not None:
            regularization_term = calculate_regularization(batch, outputs, model, loss_regularization_dict)
            loss += loss_regularization_dict['lmbda'] * regularization_term

        total_loss += loss.item()
        list_testing_loss.append(loss.item())

        torch.cuda.synchronize()
        
        true_ys.append(labels_long.detach().cpu().numpy())
        pred_ys.append(outputs.detach().cpu().numpy())
        list_pids.append(pids_long.numpy())
        list_ymask.append(ymask_long.detach().cpu().numpy())


    true_ys_flattened = np.concatenate(true_ys).ravel()
    pred_ys_flattened = np.concatenate(pred_ys).ravel()
    pids_flattened = np.concatenate(list_pids).ravel()
    ymask_flattened = np.concatenate(list_ymask, axis=0)
    pred_labels = (pred_ys_flattened>0.5)*1
    

    auc_test = roc_auc_score(true_ys_flattened, pred_ys_flattened)
    f1_test = f1_score(true_ys_flattened, pred_labels)
    auprc_test = average_precision_score(true_ys_flattened, pred_ys_flattened)
    correct_label = accuracy_score(true_ys_flattened, pred_labels)

    testing_output_dict = {'auroc': auc_test, 'f1': f1_test, 'auprc': auprc_test, 'accuracy': correct_label, 'loss': np.mean(list_testing_loss)}
    outputs_arr = np.vstack((pids_flattened, ymask_flattened, true_ys_flattened, pred_ys_flattened)).T
    return testing_output_dict, outputs_arr

def get_optimizer(params, optimizer_config):
    optimizer_cls = getattr(optim, optimizer_config["type"], None)
    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")
    return optimizer_cls(params, **optimizer_config.get("params", {}))

def get_scheduler(optimizer, scheduler_config):
    scheduler_cls = getattr(lr_scheduler, scheduler_config["type"], None)
    if scheduler_cls is None:
        raise ValueError(f"Unsupported scheduler: {scheduler_config['type']}")
    return scheduler_cls(optimizer, **scheduler_config.get("params", {}))

def get_loss_function(loss_config):
    if loss_config["type"] == "WeightedBCELoss":
        return Weighted_BCELoss(weights=loss_config["weights"])
    else:
        raise ValueError(f"Unsupported loss type: {loss_config['type']}")

def make_labels_long(labels, ymask):
    """
    This function returns the "long" version of pids or labels based on the masking in tte -- only "post-psychosis" 
    iterations
    """
    labels_long = labels.unsqueeze(dim = 1).expand(-1, ymask.shape[1])   # (batch, seq_len)
    labels_long = labels_long.reshape(-1, 1)   # (batch*seq_len, 1)
    ymask_reshaped = ymask.reshape(-1) # (batch*seq_len,)
    labels_long = labels_long[ymask_reshaped > 0]      # (num_valid,)
    return labels_long



def get_skiptrain(y_mask, skip_train):
    """
    Args:
        y_mask (torch.Tensor): Tensor of shape (batch, seq).
        skip_train (int): Interval to keep elements.

    Returns:
        torch.Tensor: Same shape as y_mask with only every skip_train-th value kept.
    """
    batch, seq = y_mask.shape
    device = y_mask.device

    nonzero_mask = (y_mask != 0)
    # get first nonzero index per row
    has_nonzero = nonzero_mask.any(dim=1)
    first_idx = torch.where(
        has_nonzero,
        nonzero_mask.float().argmax(dim=1),
        torch.full((batch,), seq, device=device)  # seq → means "no nonzero"
    )

    # 2. Create a matrix of sequence indices
    seq_idx = torch.arange(seq, device=device).unsqueeze(0).expand(batch, seq)

    # 3. Build a mask: True only where we keep values
    # keep if (index - first_idx) % skip_train == 0 and index >= first_idx
    keep_mask = (seq_idx >= first_idx.unsqueeze(1)) & \
                (((seq_idx - first_idx.unsqueeze(1)) % skip_train) == 0)

    # 4. Zero out everything else but preserve original values
    y_new = y_mask * keep_mask
    return y_new
