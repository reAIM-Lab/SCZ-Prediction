import numpy as np
import os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
import sys
import gc
from scipy.sparse import *
import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle 
import random
import math
from joblib import dump, load
from dl_eval_utils import *
from models import *
import torch.optim.lr_scheduler as lr_scheduler


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
            
def downsize_batches(loader, new_batchsize):
    list_xs = []
    list_ys = []
    for x, y in loader:
        list_xs.append(x)
        list_ys.append(y)

    dataset_x = np.concatenate(list_xs)
    dataset_y = np.concatenate(list_ys)

    dataset = torch.utils.data.TensorDataset(torch.Tensor(dataset_x), torch.Tensor(dataset_y))
    smaller_test_loader = torch.utils.data.DataLoader(dataset, batch_size=new_batchsize)
    return smaller_test_loader

def train(train_loader, model, device, optimizer, criterion):
    model.train()

    epoch_loss = 0
    list_training_loss = []
    
    true_ys = []
    pred_ys = []
    for i, (signals, labels) in enumerate(train_loader):
        
        optimizer.zero_grad()
        signals, labels = signals.to(device), labels.to(device)
        # Check if they're on the correct device
        assert signals.device == device, f"Signals not on device: {signals.device}"
        assert labels.device == device, f"Labels not on device: {labels.device}"

        outputs = model(signals)
        loss = criterion(outputs.squeeze(), labels.squeeze())
        epoch_loss += loss.item()
        loss.backward()
        list_training_loss.append(loss.item())
        optimizer.step()
        
        torch.cuda.synchronize()
        
        true_ys.append(labels.detach().cpu().numpy())
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

def test(test_loader, model, device, criteria, verbose=True):
    model = model.to(device)
    model.eval()

    total_loss = 0
    list_testing_loss = []
    true_ys = []
    pred_ys = []
    for i, (x, y) in enumerate(test_loader):
        x, y = torch.Tensor(x.float()).to(device), torch.Tensor(y.float()).to(device)
        out = model(x)
        
        true_ys.append(y.detach().cpu().numpy())
        pred_ys.append(out.detach().cpu().numpy())

        loss = criteria(out.squeeze(), y.squeeze())
        total_loss += loss.item()
        list_testing_loss.append(loss.item())
    
    true_ys_flattened = np.concatenate(true_ys).ravel()
    pred_ys_flattened = np.concatenate(pred_ys).ravel()
    pred_labels = (pred_ys_flattened>0.5)*1
    
    auc_test = roc_auc_score(true_ys_flattened, pred_ys_flattened)
    f1_test = f1_score(true_ys_flattened, pred_labels)
    auprc_test = average_precision_score(true_ys_flattened, pred_ys_flattened)
    correct_label = accuracy_score(true_ys_flattened, pred_labels)
    
    testing_output_dict = {'auroc': auc_test, 'f1': f1_test, 'auprc': auprc_test, 'accuracy': correct_label, 'loss': np.mean(list_testing_loss)}


    return testing_output_dict


def train_model(model, train_loader, valid_loader, optimizer, n_epochs, device, criterion, cv=0, str_save = None, scheduler = None):
    train_loss_trend = []
    test_loss_trend = []
    early_stopping = EarlyStopping(patience=5, verbose=False)
    best_val_loss = float('inf')  # Initialize best validation loss
    best_model_state = None       # To store the state of the best model



    model = model.to(device)
    for epoch in range(n_epochs + 1):
        training_output_dict = train(train_loader, model, device, optimizer, criterion)
        testing_output_dict = test(valid_loader, model, device, criterion)
        
        train_loss_trend.append(training_output_dict['loss'])
        test_loss_trend.append(testing_output_dict['loss'])
        
        current_val_loss = testing_output_dict['loss']
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_state = model.state_dict()  # Save the best model state

        if scheduler:
            scheduler.step()
            
        if epoch % 5 == 0:
            print('\nEpoch %d' % (epoch))
            print('Training ===>loss: ', training_output_dict['loss'],
                  ' Accuracy: %.2f percent' % (100*training_output_dict['accuracy']),
                  ' AUC: %.2f' % (training_output_dict['auroc']),
                 ' F1-Score: %.2f' % (training_output_dict['f1']))
            print('Test ===>loss: ', testing_output_dict['loss'],
                  ' Accuracy: %.2f percent' % (100*testing_output_dict['accuracy']),
                  ' AUC: %.2f' % (testing_output_dict['auroc']),
                 ' F1-Score: %.2f' % (testing_output_dict['f1']))
        
        early_stopping(testing_output_dict['loss'], model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Save model and results
    #torch.save(model.state_dict(), 'psychosis_prediction/models/transformer1.pt')
        

    plt.figure()
    plt.plot(train_loss_trend, label='Train loss')
    plt.plot(test_loss_trend, label='Validation loss')
    plt.legend()
    if best_model_state is not None and str_save is not None: # save model and plot
        torch.save(best_model_state, str_save)
        print(f"Best model saved with validation loss: {best_val_loss}")

        str_save = str_save[0:-3] + '.png'
        plt.savefig(str_save)
        
        model.load_state_dict(best_model_state)
    
# Grid search -- hyperparameter tuning
def get_model_performance(model, train_loader, val_loader, loss_weights, dict_metrics, lr = 1e-5, n_epochs=100, str_save = None, scheduler = None, wd = 0, device = torch.device("cuda:0")):
    model.to(device)
    parameters = model.parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay = wd)
    
    if scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


    # train model
    train_model(model, train_loader, val_loader, optimizer, n_epochs, device,cv=0, criterion = Weighted_BCELoss(weights=loss_weights), str_save = str_save, scheduler = scheduler)
    
    # predict on validation data
    model.eval()
    with torch.no_grad():
        pred_ys = []
        true_ys = []
        model.to(device)
        for i, (x, y) in enumerate(val_loader):
            x, y = torch.Tensor(x.float()).to(device), torch.Tensor(y.float()).to(device)
            out = model(x)
            pred_ys.append(out.detach().cpu().numpy())
            true_ys.append(y.detach().cpu().numpy())

        pred_ys_flattened = np.concatenate(pred_ys).ravel()
        true_ys_flattened = np.concatenate(true_ys).ravel()
    model.train()

    # create validation dataset
    check_data = pd.DataFrame(true_ys_flattened, columns = ['sz_flag'])
    check_data['pred_prob'] = pred_ys_flattened
    
    #if str_save is not None:
    #    torch.save(model.state_dict(), str_save)

    fpr, tpr, thresholds = roc_curve(check_data['sz_flag'], check_data['pred_prob'])
    idx = np.argmax(tpr - fpr)
    cutoff_prob = thresholds[idx]
    check_data['y_pred'] = 1*(check_data['pred_prob'] > cutoff_prob)
    
    perf = bootstrap_evaluation(check_data['sz_flag'], check_data['pred_prob'], cutoff_prob, dict_metrics, binary_funcs=['Sensitivity', 'Specificity', 'PPV'], n_bootstraps = 300, alpha = 0.05)

    torch.cuda.empty_cache()
    del model
    gc.collect()
    
    return list(perf.values)
    