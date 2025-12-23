import random
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.mixture import GaussianMixture
import pandas as pd
import pickle
import numpy as np
import sys
import re
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tnrange, tqdm_notebook
import torch.nn as nn
import time

from explainers import *
sys.path.append('../../long_model_training/utils')
from models import *
from train_utils import downsize_batches

seed_value = 35
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)

print(torch.get_float32_matmul_precision())

# set up dataset

device = torch.device("cuda:2")

with open(f'{int_path}/9_26_mdcd_2dx_fullhistory_du_snomed_colnames', "rb") as fp:
    data_columns = pickle.load(fp)

# load model
device = torch.device("cuda:0") 
model_config = torch.load(f"{model_directory}/best_model_config.pt", weights_only=False, map_location=device)
testing_clf = model_config['name'](**model_config['params']).to(device)
testing_clf.load_state_dict(torch.load(f"{model_directory}/best_model.pt", weights_only = False, map_location=device))
testing_clf.to(device)
# testing_clf = torch.compile(testing_clf)
testing_clf.eval()

# import data
data_dist = torch.load(f'{save_directory}/data_points_distribution.pt', weights_only=False)
test_loader_small = torch.load(f'{int_path}/{dataset_prefix}test_loader.pth', weights_only=False)
test_loader = downsize_batches(test_loader_small, 4096)

# AFO Explainer
explainer = AFOExplainer(testing_clf, data_dist, activation = None) 

# Run AFO
testing_ts = np.arange(10, 73, 5)# np.arange(40, 73, 10) 

for ind in testing_ts:
    results_dict = {'pids':[], 'tte':[], 'time_iter': [], 'importance_scores': [], 'ranked_feats': []}
    for i, batch in enumerate(test_loader):
        print(ind, i)
        pids, signals, padding_mask, tte, labels = batch
        
        # limit to only people with something to change in that location
        batch_mask = padding_mask[:, ind] == 1 # get all the people who have a point at this time
        pids = pids[batch_mask]
        signals = signals[batch_mask]
        padding_mask = padding_mask[batch_mask]
        tte = tte[batch_mask]
        print(pids.shape, signals.shape, padding_mask.shape, tte.shape)

        signals, padding_mask = signals.to(device, non_blocking=True), padding_mask.to(device, non_blocking=True)

        signals = signals.permute(0,2,1) # Transformer: pid, features, time
        score = explainer.attribute(signals, padding_mask, ind)
        ranked_features = np.array([((-(score[n])).argsort(0).argsort(0) + 1) for n in range(score.shape[0])])

        # get "long" pids so it should be the same length as score
        pids_long = pids.unsqueeze(dim = 1).expand(-1, tte.shape[1]).reshape(-1, 1)   # (batch * seq_len, 1)
        tte_mask = tte.reshape(-1) # (batch*seq_len,)
        pids_long = pids_long[tte_mask > 0] # (num_valid,)
        tte_long = tte_mask[tte_mask > 0]
        print(pids_long.shape, tte_long.shape, score.shape, ranked_features.shape)

        results_dict['pids'].append(pids_long)
        results_dict['tte'].append(tte_long)
        results_dict['time_iter'].append([ind])
        results_dict['importance_scores'].append(score)
        results_dict['ranked_feats'].append(ranked_features)

        torch.save(results_dict, f'{save_directory}/AFO_outputs_time{ind}.pt')
