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
    model = model.to(device)
    model.train()
    epoch_loss = 0
    list_training_loss = []
    
    true_ys = []
    pred_ys = []
    for i, (signals, labels) in enumerate(train_loader):
        
        optimizer.zero_grad()
        signals, labels = signals.to(device), labels.to(device)
        
        outputs = model(signals)
        loss = criterion(outputs.squeeze(), labels.squeeze())
        epoch_loss += loss.item()
        loss.backward()
        list_training_loss.append(loss.item())
        optimizer.step()

        
        true_ys.append(labels.detach().cpu().numpy())
        pred_ys.append(outputs.detach().cpu().numpy())


    true_ys_flattened = np.concatenate(true_ys).ravel()
    pred_ys_flattened = np.concatenate(pred_ys).ravel()
    pred_labels = (pred_ys_flattened>0.5)*1
    
    auc_train = roc_auc_score(true_ys_flattened, pred_ys_flattened)
    f1_train = f1_score(true_ys_flattened, pred_labels)
    correct_label = accuracy_score(true_ys_flattened, pred_labels)

    return auc_train, f1_train, correct_label, np.mean(list_training_loss)

def test(test_loader, model, device, criteria, verbose=True):
    model.to(device)

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
    correct_label = accuracy_score(true_ys_flattened, pred_labels)

    return auc_test, f1_test, correct_label, np.mean(list_testing_loss)


def train_model(model, train_loader, valid_loader, optimizer, n_epochs, device, criterion, cv=0):
    train_loss_trend = []
    test_loss_trend = []
    early_stopping = EarlyStopping(patience=5, verbose=False)


    for epoch in range(n_epochs + 1):
        auc_train, f1_train, accuracy_train, mean_loss_train = train(train_loader, model, device, optimizer, criterion)
        auc_test, f1_test, accuracy_test, mean_loss_test = test(valid_loader, model, device, criterion)
        
        train_loss_trend.append(mean_loss_train)
        test_loss_trend.append(mean_loss_test)
        #lr_scheduler.step()
        if epoch % 5 == 0:
            print('\nEpoch %d' % (epoch))
            print('Training ===>loss: ', mean_loss_train,
                  ' Accuracy: %.2f percent' % (100*accuracy_train),
                  ' AUC: %.2f' % (auc_train),
                 ' F1-Score: %.2f' % (f1_train))
            print('Test ===>loss: ', mean_loss_test,
                  ' Accuracy: %.2f percent' % (100*accuracy_test),
                  ' AUC: %.2f' % (auc_test),
                 ' F1-Score: %.2f' % (f1_test))
        
        early_stopping(mean_loss_test, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Save model and results
    #torch.save(model.state_dict(), 'psychosis_prediction/models/transformer1.pt')
    plt.plot(train_loss_trend, label='Train loss')
    plt.plot(test_loss_trend, label='Validation loss')
    plt.legend()
    