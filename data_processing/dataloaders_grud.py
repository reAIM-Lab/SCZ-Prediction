import numpy as np
import os
import pandas as pd
import time
import scipy.stats as stats
from datetime import datetime
from tqdm import tqdm
from collections import Counter
import sys
import gc
import torch
from sklearn.preprocessing import StandardScaler
import pickle 
from joblib import dump, load

sys.path.append('../utils')
from data_utils import *

shared_cols = True
path = 'PATH'
raw_path = path + 'RAW DATA PATH'
int_path = path + 'INTERMEDIATE DATA PATH'

df_all_iters = pd.read_csv(int_path + 'CCAE_11_15_dl_data_snomed_sharedfeats.csv')
num_days_prediction = 90
df_pop = pd.read_csv(raw_path + 'population.csv')
df_pop.rename({'psychosis_dx_date':'psychosis_diagnosis_date'}, axis=1, inplace=True)
df_pop['psychosis_diagnosis_date'] = pd.to_datetime(df_pop['psychosis_diagnosis_date'], format="mixed", dayfirst = False)
df_pop['cohort_start_date'] = pd.to_datetime(df_pop['cohort_start_date'], format="mixed", dayfirst = False)
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction]

df_split = pd.read_csv(int_path + 'CCAE_11_26_vt_split.csv')
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
    save_cols = load(path + 'MDCD_11_15_grud_colnames_snomed')
    df_all_iters = df_all_iters[save_cols]
print(len(save_cols))

# get the actual data points
train_data = df_all_iters.loc[train_pids]
val_data = df_all_iters.loc[val_pids]
test_data = df_all_iters.loc[test_pids]

# get the masks of the data points
train_data_mask = (train_data>0)*1
val_data_mask = (val_data>0)*1
test_data_mask = (test_data>0)*1

make_scaler = False
if make_scaler:
    scaler = StandardScaler()
    train_data_mat = scaler.fit_transform(train_data)
    print('done with fit/first transform')
    val_data_mat = scaler.transform(val_data)
    test_data_mat = scaler.transform(test_data)

    # save the standard scaler
    dump(scaler, int_path + 'CCAE_11_26_grud_dl_sharedfeats_vt_scaler_snomed.bin', compress=True)
    
else:
    scaler = load(path + 'raw_data_3yrs/intermediate_data_mdcd/MDCD_11_15_dl_scaler_snomed.bin')
    train_data_mat = scaler.transform(train_data)
    val_data_mat = scaler.transform(val_data)
    test_data_mat = scaler.transform(test_data)


# load in and add on the column for the number of days since psychosis
df_timedeltas = pd.read_csv(int_path + 'CCAE_11_15_grud_timedeltas_snomed_sharedfeats.csv')
df_timedeltas = df_timedeltas.merge(df_all_iters.reset_index()[['person_id','ranked_iteration','time_since_psychosis']], how='inner', on=['person_id','ranked_iteration'])
df_timedeltas = df_timedeltas.sort_values(['person_id', 'ranked_iteration'])

for i in set(save_cols).difference(df_timedeltas.columns):
    if '42898160' in i:
        df_timedeltas[i] = df_timedeltas['42898160']
    elif '9201' in i:
        df_timedeltas[i] = df_timedeltas['9201']
    elif '9202' in i:
        df_timedeltas[i] = df_timedeltas['9202']
    elif '9203' in i:
        df_timedeltas[i] = df_timedeltas['9203']
    elif 'psych' in i:
        df_timedeltas[i] = df_timedeltas['psych_visits']

        
df_timedeltas.drop(['42898160','9201','9202','9203','Unnamed: 0', 'psych_visits'], axis=1, inplace=True)
print(df_timedeltas.shape, len(save_cols))

# divide timedeltas into train/test/split
overall_max = df_timedeltas['ranked_iteration'].max()
print(overall_max)
df_timedeltas.set_index(['person_id','ranked_iteration'], inplace=True)
df_timedeltas.drop('iteration', axis=1, inplace=True)
df_timedeltas.sort_index(inplace=True)
df_timedeltas = df_timedeltas[save_cols]
# get the actual data points
train_tds = df_timedeltas.loc[train_pids]
val_tds = df_timedeltas.loc[val_pids]
test_tds = df_timedeltas.loc[test_pids]

labels = df_pop[['person_id', 'sz_flag']].set_index('person_id')

# Get dataloaders
# VALIDATION
val_data_data = get_full_df(val_data, val_data_mat, val_pids, overall_max = overall_max)
val_data_data = val_data_data.reshape(len(val_pids), int(len(val_data_data)/len(val_pids)), val_data_mat.shape[1])

val_data_mask = get_full_df(val_data_mask, val_data_mask.values, val_pids, overall_max = overall_max)
val_data_mask = val_data_mask.reshape(len(val_pids), int(len(val_data_mask)/len(val_pids)), val_data_mask.shape[1])

val_last_obs = torch.Tensor(val_data_data * val_data_mask)
val_last_obs_mat = torch.cat((torch.zeros((val_last_obs.shape[0], 1, val_last_obs.shape[2])), val_last_obs), 1)
val_last_obs_mat = val_last_obs_mat[:,0:-1,:]
print(val_last_obs_mat.shape)

val_data_deltas = get_full_df(val_tds, val_tds.values, val_pids, overall_max = overall_max)
val_data_deltas = val_data_deltas.reshape(len(val_pids), int(len(val_data_deltas)/len(val_pids)), val_tds.shape[1])

val_labels = labels.loc[val_pids]
stacked_val_data = np.stack([val_data_data, val_last_obs_mat, val_data_mask, val_data_deltas], axis=1)
print(stacked_val_data.shape, val_labels.shape)

val_dataset = torch.utils.data.TensorDataset(torch.Tensor(stacked_val_data), torch.Tensor(val_labels.values))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2048, shuffle=True)
torch.save(val_loader, int_path+'CCAE_11_26_grud_dl_individual_snomed_val_loader_shuffled.pth')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2048, shuffle=False)
torch.save(val_loader, int_path+'CCAE_11_26_grud_dl_individual_snomed_val_loader_unshuffled.pth')

# TEST
test_data_data = get_full_df(test_data, test_data_mat, test_pids, overall_max = overall_max)
test_data_data = test_data_data.reshape(len(test_pids), int(len(test_data_data)/len(test_pids)), test_data_mat.shape[1])

test_data_mask = get_full_df(test_data_mask, test_data_mask.values, test_pids, overall_max = overall_max)
test_data_mask = test_data_mask.reshape(len(test_pids), int(len(test_data_mask)/len(test_pids)), test_data_mask.shape[1])

test_last_obs = torch.Tensor(test_data_data * test_data_mask)
test_last_obs_mat = torch.cat((torch.zeros((test_last_obs.shape[0], 1, test_last_obs.shape[2])), test_last_obs), 1)
test_last_obs_mat = test_last_obs_mat[:,0:-1,:]

test_data_deltas = get_full_df(test_tds, test_tds.values, test_pids, overall_max = overall_max)
test_data_deltas = test_data_deltas.reshape(len(test_pids), int(len(test_data_deltas)/len(test_pids)), test_data_deltas.shape[1])

test_labels = labels.loc[test_pids]
stacked_test_data = np.stack([test_data_data, test_last_obs_mat, test_data_mask, test_data_deltas], axis=1)
print(stacked_test_data.shape, test_labels.shape)

test_dataset = torch.utils.data.TensorDataset(torch.Tensor(stacked_test_data), torch.Tensor(test_labels.values))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle = False)
torch.save(test_loader, int_path+'CCAE_11_26_grud_dl_shared_mdcdscale_snomed_test_loader_unshuffled.pth')

# TRAIN
train_data_data = get_full_df(train_data, train_data_mat, train_pids, overall_max = overall_max)
train_data_data = train_data_data.reshape(len(train_pids), int(len(train_data_data)/len(train_pids)), train_data_mat.shape[1])

train_data_mask = get_full_df(train_data_mask, train_data_mask.values, train_pids, overall_max = overall_max)
train_data_mask = train_data_mask.reshape(len(train_pids), int(len(train_data_mask)/len(train_pids)), train_data_mask.shape[1])

train_last_obs = torch.Tensor(train_data_data * train_data_mask)
train_last_obs_mat = torch.cat((torch.zeros((train_last_obs.shape[0], 1, train_last_obs.shape[2])), train_last_obs), 1)
train_last_obs_mat = train_last_obs_mat[:,0:-1,:]
print(train_last_obs_mat.shape)

train_data_deltas = get_full_df(train_tds, train_tds.values, train_pids, overall_max = overall_max)
train_data_deltas = train_data_deltas.reshape(len(train_pids), int(len(train_data_deltas)/len(train_pids)), train_data_deltas.shape[1])

train_labels = labels.loc[train_pids]
stacked_train_data = np.stack([train_data_data, train_last_obs_mat, train_data_mask, train_data_deltas], axis=1)
print(stacked_train_data.shape, train_labels.shape)

train_dataset = torch.utils.data.TensorDataset(torch.Tensor(stacked_train_data), torch.Tensor(train_labels.values))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2048, shuffle = True)
torch.save(train_loader, int_path + 'CCAE_11_26_grud_dl_shared_mdcdscale_snomed_train_loader.pth')

# Get X_mean
nonzero_mask = torch.Tensor(train_data_data) != 0

nonzero_counts = nonzero_mask.sum(dim=0) # Count nonzero entries along dim 0
nonzero_sum = torch.where(nonzero_mask, torch.Tensor(train_data_data), torch.tensor(0.0)).sum(dim=0) # sum the values
safe_nonzero_counts = torch.where(nonzero_counts == 0, torch.tensor(1), nonzero_counts) # avoid division by 0
mean_matrix = nonzero_sum / safe_nonzero_counts

# Replace positions with zero count back to 0
mean_matrix = torch.where(nonzero_counts == 0, torch.tensor(0.0), mean_matrix)
mean_matrix = mean_matrix.unsqueeze(0)
print(train_data_data.shape, mean_matrix.shape)

torch.save(mean_matrix, int_path + 'CCAE_11_26_grud_mdcdscale_shared_means.pt')