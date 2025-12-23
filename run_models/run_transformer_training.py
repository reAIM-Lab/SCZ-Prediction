import torch
import torch.nn as nn
import torch.optim as optim
import copy
import json
from pathlib import Path
from collections import defaultdict
import itertools
import pandas as pd
import sys
import pickle

sys.path.append('utils')
import models
from losses import *
from train_utils import *
from training import *

seed_value = 35
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)

# set up paths

os.makedirs(save_directory, exist_ok=True)
device = torch.device("cuda:0")


# import data
train_loader = torch.load(f'{int_path}/{dataset_prefix}train_loader.pth', weights_only=False)
val_loader = torch.load(f'{int_path}/{dataset_prefix}val_loader.pth', weights_only=False)
test_loader = torch.load(f'{int_path}/{dataset_prefix}test_loader.pth', weights_only=False)

with open(f'{int_path}/9_26_ccae_2dx_fullhistory_du_snomed_colnames', "rb") as fp:   # Unpickling
    data_columns = pickle.load(fp)

# Define weights for loss function 
unweighted_weights = torch.tensor(np.asarray([1, 1])).to(device)

grid_search_all = {'learning_rate': [1e-3, 1e-4, 1e-5],
               'weight_decay': [1e-2, 1e-3],
               'optimizer':['AdamW', 'Adam', 'RMSprop'],
               'hidden_size': [128, 256, 512],
               'dim_feedforward': [128, 256, 512, 1024],
               'num_layers': [2, 4, 6],
               'dropout': [0.1, 0.3, 0.5],
               'emb_first': [True, False],
               'loss_weights': [unweighted_weights]
}

# baseline models: create config list and RANDOMLY SAMPLE
keys, values = zip(*grid_search_all.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
permutations_dicts = np.random.choice(permutations_dicts, size=50, replace=False).tolist()

list_configs_baseline = []
list_model_configs = []
for params in permutations_dicts:
    config = {
    "loss": {
        "type": "WeightedBCELoss",
        "weights": params['loss_weights']
    },
    "optimizer": {
        "type": params['optimizer'],  
        "params": {"lr": params['learning_rate'], "weight_decay": params['weight_decay']}
    },
    "scheduler": {
        "type": "StepLR",
        "params": {"step_size": 5, "gamma": 0.5}
    },
    "early_stopping_patience": 5,
    "epochs": 30,
    "device":device,
    'skip_train': 8,
    "loss_regularization_dict": None,
    "model_selection": 'loss', # auroc or loss
    "init_model": False # False if model architecture is part of hyperparam search
}
    list_configs_baseline.append(config)

    if params['emb_first'] == True:
        model_config = {'name': models.TransformerModelEmbPE}
    else: 
        model_config = {'name': models.TransformerModelPEEmb}    
    model_config['params'] = {'hidden_size': params['hidden_size'], 'dim_feedforward': params['dim_feedforward'],
                              'num_layers': params['num_layers'], 'num_heads': 4, 'dropout': params['dropout'],
                              'n_features': len(data_columns), 'zero_iter': 59}
    list_model_configs.append(model_config)

torch.save(list_configs_baseline, f"{save_directory}/hparam_configs_list.pt")
torch.save(list_model_configs, f"{save_directory}/model_configs_list.pt")

search_hyperparams(list_configs_baseline, list_model_configs, train_loader, val_loader, test_loader, save_directory, device)

# save final output
model_config = torch.load(f"{save_directory}/best_model_config.pt", weights_only=False)
model = model_config['name'](**model_config['params']).to(device)
model.load_state_dict(torch.load(f"{save_directory}/best_model.pt", weights_only = False))

config = torch.load(f"{save_directory}/best_hparam_config.pt", weights_only=False) 
config['skip_train'] = 1
unweighted_loss_config = config["loss"]
unweighted_loss_config['weights'] = torch.Tensor([1,1])
unweighted_criterion = get_loss_function(unweighted_loss_config)

_, val_output = test(val_loader, model, device, unweighted_criterion, 
                        config.get("loss_regularization_dict"), 1)
_, test_output = test(test_loader, model, device, unweighted_criterion, 
                        config.get("loss_regularization_dict"), 1)
pd.DataFrame(val_output, columns = ['person_id', 'tte', 'y_true', 'y_pred']).to_csv(os.path.join(save_directory, "val_outputs.csv"))
pd.DataFrame(test_output, columns = ['person_id', 'tte', 'y_true', 'y_pred']).to_csv(os.path.join(save_directory, "test_outputs.csv"))
