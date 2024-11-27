import numpy as np
import os
import pandas as pd
import pyodbc
import time
import scipy.stats as stats
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from datetime import datetime
import sys
import gc
import torch
from sklearn.metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle 
import random
import math
from joblib import dump, load
import itertools
import warnings
warnings.filterwarnings('ignore')

from models import *
sys.path.append('../utils')
from train_utils import *
from eval_utils import *

path = 'PATH'
raw_path = path + 'RAW DATA PATH'
int_path = path + 'INTERMEDIATE DATA PATH'
result_path = 'RESULTS PATH'
result_prefix = 'CCAE_11_26'

train_loader = torch.load(int_path + 'CCAE_11_26_dl_individual_snomed_train_loader.pth')
val_loader_shuffled = torch.load(int_path + 'CCAE_11_26_dl_individual_snomed_val_loader_shuffled.pth')

with open(int_path + "CCAE_11_26_dl_individualfeats_colnames_snomed", "rb") as fp:   # Unpickling
    save_cols = pickle.load(fp)

train_labels = [np.asarray(y) for _, (_, y) in enumerate(train_loader)]
train_labels = pd.Series(np.concatenate(train_labels).reshape(-1))

val_loader_shuffled = downsize_batches(val_loader_shuffled, 2048)
train_loader = downsize_batches(train_loader, 2048)

device = torch.device("cuda:0") 

# Define Weights
unweighted_weights = torch.tensor(np.asarray([1, 1])).to(device)
standard_weights = torch.tensor(np.asarray([sum(train_labels.values==0)/sum(train_labels.values==0), sum(train_labels.values==0)/sum(train_labels.values)])).to(device)
half_weights = torch.tensor(np.asarray([sum(train_labels.values==0)/sum(train_labels.values==0), 0.5*sum(train_labels.values==0)/sum(train_labels.values)])).to(device)
double_weights = torch.tensor(np.asarray([sum(train_labels.values==0)/sum(train_labels.values==0), 2*sum(train_labels.values==0)/sum(train_labels.values)])).to(device)

dict_metrics = {'AUROC': roc_auc_score, 'AUPRC': average_precision_score,
               'Sensitivity': sensitivity, 'Specificity': specificity, 
               'PPV': precision_score}

# Grid Search: LSTM
n_epochs = 60 
n_features = len(save_cols)
print('Number of features', n_features)
num_heads = 4 
list_performances = []

search_space = {
    "embedding_size": [128, 256, 512, 1024],
    "hidden_size": [128, 256, 512],
    "num_layers": [2, 4],
    "weights": [standard_weights, half_weights, double_weights, unweighted_weights],
    "dropout": [0.1, 0.3, 0.5],
    "learning_rate": [1e-3, 1e-4],
    "weight_decay": [1e-2, 1e-3]
}

keys, values = zip(*search_space.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
random.seed(26)
random_grid = random.sample(permutations_dicts, 250)

for config in random_grid:
    print(config)
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    loss_weights = config["weights"]
    wd = config['weight_decay']
    dropout = config['dropout']
    lr = config['learning_rate']
    embedding_size = config['embedding_size']
    
    str_model_name = 'lstm_es'+str(embedding_size)+'_hs'+str(hidden_size)+'_nl'+str(num_layers)+'_wd'+str(wd)+'_lr'+str(lr)+'_dropout'+str(dropout)+'_weights'+str(int(loss_weights[1].cpu().numpy()))+'.pt'
    model = LSTMModel(n_features, embedding_size, hidden_size, num_layers, dropout = dropout, output_size=1)
    list_perf = (get_model_performance(model, train_loader, val_loader_shuffled, 
                                       loss_weights = loss_weights, wd = wd, lr = lr, n_epochs = n_epochs, dict_metrics = dict_metrics, str_save = result_path+str_model_name, scheduler = True))
    list_performances.append([config]+list(np.asarray(list_perf).reshape(-1)))
    
    df_performance_lstm = pd.DataFrame(data = list_performances, columns = ['config', 'AUROC', 'AUROC CI low', 'AUROC CI high','AUPRC', 'AUPRC CI low', 'AUPRC CI high','Sensitivity', 'Sensitivity CI low', 'Sensitivity CI high','Specificity', 'Specificity CI low', 'Specificity CI high','PPV', 'PPV CI low', 'PPV CI high'])

    df_performance_lstm.to_csv(result_path + result_prefix + 'lstm_gridsearch.csv')

device = torch.device("cuda:0") 
n_epochs = 60 
n_features = len(save_cols)
num_heads = 4 
list_performances = []

# TRANSFORMERS
search_space = {
    "hidden_size": [128, 256, 512],
    "dim_feedforward": [128, 256, 512, 1024],
    "num_layers": [2, 4],
    "weights": [standard_weights, half_weights, double_weights, unweighted_weights],
    "emb_first": [True, False],
    "dropout": [0.1, 0.3, 0.5],
    "learning_rate": [1e-3, 1e-4],
    "weight_decay": [1e-2, 1e-3]
}

keys, values = zip(*search_space.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

random.seed(26)
random_grid = random.sample(permutations_dicts, 250)

for config in random_grid:
    print(config)
    dim_feedforward = config["dim_feedforward"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    dropout = config["dropout"]
    emb_first = config["emb_first"]
    loss_weights = config["weights"]
    wd = config['weight_decay']
    lr = config["learning_rate"]
    
    str_model_name = 'transformer_dimFF'+str(dim_feedforward)+'_hs'+str(hidden_size)+'_nl'+str(num_layers)+'_lr'+str(lr)+'_dropout'+str(dropout)+'_wd'+str(wd)+'_weights'+str(int(loss_weights[1].cpu().numpy()))
    
    if emb_first == True: 
        model = TransformerModelEmbPE(hidden_size, dim_feedforward, num_layers, num_heads, dropout, n_features)
        str_model_name += '_EmbPE.pt'
    
    else: 
        model = TransformerModelPEEmb(hidden_size, dim_feedforward, num_layers, num_heads, dropout, n_features)
        str_model_name += '_PEEmb.pt'

    list_perf = (get_model_performance(model, train_loader, val_loader_shuffled, 
                           loss_weights = loss_weights, lr = lr, n_epochs = n_epochs, wd = wd, str_save = result_path+str_model_name, scheduler = True, dict_metrics = dict_metrics))
    list_performances.append([config]+list(np.asarray(list_perf).reshape(-1)))
    
    df_performance_trans = pd.DataFrame(data = list_performances, columns = ['config','AUROC', 'AUROC CI low', 'AUROC CI high','AUPRC', 'AUPRC CI low', 'AUPRC CI high','Sensitivity', 'Sensitivity CI low', 'Sensitivity CI high',
    'Specificity', 'Specificity CI low', 'Specificity CI high','PPV', 'PPV CI low', 'PPV CI high'])

    df_performance_trans.to_csv(result_path + result_prefix + 'transformer_gridsearch.csv')
