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

path = '/data2/processed_datasets/ak4885/psychosis_schizophrenia_prediction/'
raw_path = path + 'raw_data_3yrs/raw_data_ccae/'
int_path = path + 'raw_data_3yrs/intermediate_data_ccae/'
result_path = 'model_runs/11_26_ccae_grud_adam/'
result_prefix = 'CCAE_11_26_'

train_loader = torch.load(int_path + 'CCAE_11_26_grud_dl_individual_snomed_train_loader.pth')
val_loader_shuffled = torch.load(int_path + 'CCAE_11_26_grud_dl_individual_snomed_val_loader_shuffled.pth')

X_mean = torch.load(int_path + 'CCAE_11_26_grud_individual_means.pt')


with open(int_path + "CCAE_11_26_grud_dl_individualfeats_colnames_snomed", "rb") as fp:   # Unpickling
    save_cols = pickle.load(fp)

train_labels = [np.asarray(y) for _, (_, y) in enumerate(train_loader)]
train_labels = pd.Series(np.concatenate(train_labels).reshape(-1))

device = torch.device("cuda:0") 

# Define Weights
unweighted_weights = torch.tensor(np.asarray([1, 1])).to(device)
standard_weights = torch.tensor(np.asarray([sum(train_labels.values==0)/sum(train_labels.values==0), sum(train_labels.values==0)/sum(train_labels.values)])).to(device)
half_weights = torch.tensor(np.asarray([sum(train_labels.values==0)/sum(train_labels.values==0), 0.5*sum(train_labels.values==0)/sum(train_labels.values)])).to(device)
double_weights = torch.tensor(np.asarray([sum(train_labels.values==0)/sum(train_labels.values==0), 2*sum(train_labels.values==0)/sum(train_labels.values)])).to(device)

dict_metrics = {'AUROC': roc_auc_score, 'AUPRC': average_precision_score,
               'Sensitivity': sensitivity, 'Specificity': specificity, 
               'PPV': precision_score}



# Grid Search: GRU-D
n_epochs = 60 
n_features = len(save_cols)
print('Number of features', n_features)
input_dim = n_features 
hidden_dim = n_features 
output_dim = n_features

list_performances = []

search_space = {
    "weights": [standard_weights, half_weights, double_weights, unweighted_weights],
    "learning_rate": [1e-3, 1e-4, 1e-5],
    "weight_decay": [1e-2, 1e-3]
}

keys, values = zip(*search_space.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

for config in permutations_dicts:
    print(config)
    loss_weights = config["weights"]
    wd = config['weight_decay']
    lr = config['learning_rate']
    
    str_model_name = 'lstm_wd'+str(wd)+'_lr'+str(lr)+'_weights'+str(int(loss_weights[1].cpu().numpy()))+'.pt'
    model = GRUD(input_dim, hidden_dim, output_dim, X_mean, output_last = True)

    list_perf = (get_model_performance(model, train_loader, val_loader_shuffled, 
                                       loss_weights = loss_weights, wd = wd, lr = lr, n_epochs = n_epochs, dict_metrics = dict_metrics, str_save = result_path+str_model_name, scheduler = True))
    list_performances.append([config]+list(np.asarray(list_perf).reshape(-1)))
    
    df_performance_grud = pd.DataFrame(data = list_performances, columns = ['config', 'AUROC', 'AUROC CI low', 'AUROC CI high','AUPRC', 'AUPRC CI low', 'AUPRC CI high','Sensitivity', 'Sensitivity CI low', 'Sensitivity CI high','Specificity', 'Specificity CI low', 'Specificity CI high','PPV', 'PPV CI low', 'PPV CI high'])

    df_performance_grud.to_csv(result_path + result_prefix + 'gru_gridsearch.csv')

