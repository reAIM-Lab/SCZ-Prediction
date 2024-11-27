import numpy as np
import os
import pandas as pd
import time
from datetime import datetime
from collections import Counter
import sys
from sklearn.preprocessing import StandardScaler
import gc
import torch
import pickle 
import random
import math
from joblib import dump, load

sys.path.append('../utils')
from data_utils import *

shared_cols = True
make_scaler = False

path = 'PATH'
raw_path = path + 'RAW DATA PATH'
int_path = path + 'INTERMEDIATE DATA PATH'

df_all_iters = pd.read_csv(int_path + 'CCAE_11_15_dl_data_snomed_individualfeats.csv')
num_days_prediction = 90
df_pop = pd.read_csv(raw_path + 'population.csv')
df_pop.rename({'psychosis_dx_date':'psychosis_diagnosis_date'}, axis=1, inplace=True)
df_pop['psychosis_diagnosis_date'] = pd.to_datetime(df_pop['psychosis_diagnosis_date'], format="mixed", dayfirst = False)
df_pop['cohort_start_date'] = pd.to_datetime(df_pop['cohort_start_date'], format="mixed", dayfirst = False)
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction]

df_split = pd.read_csv(int_path + 'CCAE_11_20_tvt_split.csv')
df_split = df_split.loc[df_split['person_id'].isin(df_all_iters['person_id'])]
train_pids = list(df_split.loc[df_split['split']=='train', 'person_id'])
val_pids = list(df_split.loc[df_split['split']=='val', 'person_id'])
test_pids = list(df_split.loc[df_split['split']=='test', 'person_id'])
print(len(train_pids)/len(df_split), len(val_pids)/len(df_split), len(test_pids)/len(df_split))

overall_max = df_all_iters['ranked_iteration'].max()
print(overall_max)
df_all_iters.set_index(['person_id','ranked_iteration'], inplace=True)
df_all_iters.drop('iteration', axis=1, inplace=True)
df_all_iters.sort_index(inplace=True)

if shared_cols == False:
    save_cols = list(df_all_iters.columns)

    with open(int_path + "CCAE_11_26_dl_individualfeats_colnames_snomed", "wb") as fp:   #Pickling
        pickle.dump(save_cols, fp)
else: 
    save_cols = load(path + 'MDCD_11_15_dl_colnames_snomed')
    df_all_iters = df_all_iters[save_cols]

train_data = df_all_iters.loc[train_pids]
val_data = df_all_iters.loc[val_pids]
test_data = df_all_iters.loc[test_pids]

if make_scaler:
    scaler = StandardScaler()
    train_data_mat = scaler.fit_transform(train_data)
    print('done with fit/first transform')
    val_data_mat = scaler.transform(val_data)
    test_data_mat = scaler.transform(test_data)

    # save the standard scaler
    dump(scaler, int_path + 'CCAE_11_26_dl_sharedfeats_vt_scaler_snomed.bin', compress=True)    
else:
    scaler = load(path + 'MDCD_11_15_dl_scaler_snomed.bin')
    train_data_mat = scaler.transform(train_data)
    test_data_mat = scaler.transform(test_data)

# pad the data and create loader objects
labels = df_pop[['person_id', 'sz_flag']].set_index('person_id')
train_data = get_full_df(train_data, train_data_mat, train_pids, overall_max = overall_max)
train_data = train_data.reshape(len(train_pids), int(len(train_data)/len(train_pids)), train_data_mat.shape[1])
train_labels = labels.loc[train_pids]
print(train_data.shape, train_labels.shape)

train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_labels.values))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4096, shuffle = True)
torch.save(train_loader, int_path + 'CCAE_11_26_dl_individual_snomed_train_loader.pth')


val_data = get_full_df(val_data, val_data_mat, val_pids, overall_max = overall_max)
val_data = val_data.reshape(len(val_pids), int(len(val_data)/len(val_pids)), val_data_mat.shape[1])
val_labels = labels.loc[val_pids]
print(val_data.shape, val_labels.shape)

val_dataset = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_labels.values))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4096, shuffle=True)
torch.save(val_loader, int_path+'CCAE_11_26_dl_individual_snomed_val_loader_shuffled.pth')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4096, shuffle=False)
torch.save(val_loader, int_path+'CCAE_11_26_dl_individual_snomed_val_loader_unshuffled.pth')

test_data = get_full_df(test_data, test_data_mat, test_pids, overall_max = overall_max)
test_data = test_data.reshape(len(test_pids), int(len(test_data)/len(test_pids)), test_data_mat.shape[1])
test_labels = labels.loc[test_pids]
print(test_data.shape, test_labels.shape)

test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_labels.values))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4096, shuffle = False)
torch.save(test_loader, int_path+'CCAE_11_26_dl_shared_mdcdscale_snomed_test_loader_unshuffled.pth')