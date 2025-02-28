import numpy as np
import os
import pandas as pd
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from collections import Counter
import sys
import gc
from scipy.sparse import *
import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import pickle 
import random
import math
from joblib import dump, load
import json

shared_cols = True

bw_only = False
bm_only = False
wm_only = False
mf_only = False

make_scaler = True
demo_aware = True
keep_iters = None # [-10, 31]

df_all_iters = pd.read_csv(int_path + 'CUMC_1_27_dl_data_snomed.csv')
if 'Unnamed: 0' in df_all_iters.columns:
    print('removing unnamed column')
    df_all_iters.drop('Unnamed: 0', axis=1, inplace=True)

if keep_iters is not None:
    df_all_iters = df_all_iters.loc[df_all_iters['iteration']>=keep_iters[0]]
    df_all_iters = df_all_iters.loc[df_all_iters['iteration']<=keep_iters[1]]
    print('Check keep iters', df_all_iters['iteration'].min(), df_all_iters['iteration'].max())

num_days_prediction = 90
df_pop = pd.read_csv(raw_path + 'population.csv')
df_pop.rename({'psychosis_dx_date':'psychosis_diagnosis_date'}, axis=1, inplace=True)
df_pop['psychosis_diagnosis_date'] = pd.to_datetime(df_pop['psychosis_diagnosis_date'], format="mixed", dayfirst = False)
df_pop['cohort_start_date'] = pd.to_datetime(df_pop['cohort_start_date'], format="mixed", dayfirst = False)
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction]
    
print(len(df_all_iters))
if bw_only == True:
    # LIMIT TO Black and White patients ONLY
    df_pop = df_pop.loc[df_pop['race_concept_id'].isin([8516, 8527])]
    df_all_iters = df_all_iters.loc[df_all_iters['person_id'].isin(df_pop['person_id'])]
    
if bm_only == True:
    # LIMIT TO non-White patients ONLY
    df_pop = df_pop.loc[~(df_pop['race_concept_id'].isin([8527]))]
    df_all_iters = df_all_iters.loc[df_all_iters['person_id'].isin(df_pop['person_id'])]

if wm_only == True:
    # LIMIT TO non-Black patients ONLY
    df_pop = df_pop.loc[~(df_pop['race_concept_id'].isin([8516]))]
    df_all_iters = df_all_iters.loc[df_all_iters['person_id'].isin(df_pop['person_id'])]
    
if mf_only == True:
    df_pop = df_pop.loc[df_pop['gender_concept_id'].isin([8532, 8507])]
    df_all_iters = df_all_iters.loc[df_all_iters['person_id'].isin(df_pop['person_id'])]
print(len(df_all_iters))    
df_all_iters.head()

# add ranked iteration
ranked_vals = df_all_iters.reset_index().groupby('person_id')['iteration'].rank(method='first').values
df_all_iters['ranked_iteration'] = ranked_vals
print(df_all_iters['ranked_iteration'].max())

# check that there is at most a difference of 1 between each pid from one iteration to the next
def find_largest_diff(df):
    # Sort by pid and iteration
    df_sorted = df.sort_values(by=['person_id', 'iteration'])
    
    # Calculate the largest difference for each pid
    result = df_sorted.groupby('person_id')['iteration'].apply(
        lambda x: x.diff().max()
    ).reset_index(name='largest_diff')
    
    return result
find_largest_diff(df_all_iters)['largest_diff'].max() # should be 1

if demo_aware == True:
    df_pop['is_White'] = 0
    df_pop.loc[df_pop['race_concept_id']==8527, 'is_White'] = 1
    df_pop['is_Black'] = 0
    df_pop.loc[df_pop['race_concept_id']==8516, 'is_Black'] = 1
    df_pop['is_Male'] = 0
    df_pop.loc[df_pop['gender_concept_id']==8507, 'is_Male'] = 1
    
    print(len(df_all_iters))
    df_all_iters = df_all_iters.merge(df_pop[['person_id', 'is_White', 'is_Black', 'is_Male']], how='inner', left_on = 'person_id', right_on='person_id')
    print(len(df_all_iters))
    

df_split = pd.read_csv(int_path + 'CUMC_1_27_tvt_split.csv')
df_split = df_split.loc[df_split['person_id'].isin(df_all_iters['person_id'])]
train_pids = list(df_split.loc[df_split['split']=='train', 'person_id'])
val_pids = list(df_split.loc[df_split['split']=='val', 'person_id'])
test_pids = list(df_split.loc[df_split['split']=='test', 'person_id'])
print(len(train_pids)/len(df_split), len(val_pids)/len(df_split), len(test_pids)/len(df_split))

tvt_split = {
    "train_pids": tuple(train_pids),
    "val_pids": tuple(val_pids),
    "test_pids": tuple(test_pids)
}

with open(int_path + "CUMC_2_16_dl_da_tvt_order.json", "w") as f:
    json.dump(tvt_split, f)


overall_max = df_all_iters['ranked_iteration'].max()
print(overall_max)
df_all_iters.set_index(['person_id','ranked_iteration'], inplace=True)
df_all_iters.drop('iteration', axis=1, inplace=True)
df_all_iters.sort_index(inplace=True)


if shared_cols == False:
    save_cols = list(df_all_iters.columns)

    with open(int_path + "CUMC_1_27_dl_da_colnames", "wb") as fp:   #Pickling
        pickle.dump(save_cols, fp)
else: 
    save_cols = load(int_path + 'CUMC_1_27_dl_da_colnames')

print('Check for unnamed col (should be False):', 'Unnamed: 0' in save_cols)
df_all_iters = df_all_iters[save_cols]

print(df_all_iters.isna().sum().sum())

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
    dump(scaler, int_path + 'CUMC_1_27_dl_da_vanilla_order.bin', compress=True)    
else:
    scaler = load(path)
    train_data_mat = scaler.transform(train_data)
    val_data_mat = scaler.transform(val_data)
    test_data_mat = scaler.transform(test_data)
    

# pad the data: add 0s to the beginning
def get_full_df(original_df, scaled_df_mat, pids, overall_max=overall_max):
    # get the maximum iterations per patient and subtract from the number of timesteps in the matrix 
    # for psychosis SCZ, that is 41
    
    save_cols = original_df.columns
    original_df = original_df[original_df.columns[0:1]]
    max_iter = original_df.reset_index().groupby('person_id')['ranked_iteration'].max()
    max_iter.name = 'max_iter'
    max_iter = overall_max-max_iter
    
    # add the number of padding rows that need to happen per patient to the dataframe
    original_df = original_df.merge(max_iter, how='left', left_index=True, right_index=True)
    original_df.reset_index(inplace=True)
    original_df['ranked_iteration'] = original_df['ranked_iteration']+original_df['max_iter']
    
    # replace the data with the scaled data
    original_df.set_index(['person_id', 'ranked_iteration'], inplace=True)
    original_df.drop('max_iter', axis=1, inplace=True)
    
    # create a new dataframe that goes through each patient and each timestep
    new_df = pd.DataFrame(index=[np.repeat(pids, overall_max), np.tile(np.arange(1, overall_max+1), len(pids))], columns=save_cols)
    
    # then fill it in with the existing data
    new_df.loc[original_df.index] = scaled_df_mat
    
    # convert to matrix and fillna
    new_df = new_df.values.astype(float)
    new_df[np.isnan(new_df)] = 0
    return new_df

labels = df_pop[['person_id', 'sz_flag']].set_index('person_id')

train_data = get_full_df(train_data, train_data_mat, tvt_split['train_pids'])
train_data = train_data.reshape(len(train_pids), int(len(train_data)/len(train_pids)), train_data_mat.shape[1])
train_labels = labels.loc[train_pids]
print(train_data.shape, train_labels.shape)

train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_data), torch.Tensor(train_labels.values))
train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=1024, shuffle = True, worker_init_fn=np.random.seed(14))
torch.save(train_loader, int_path + 'CUMC_2_16_dl_da_vanilla_train_loader.pth')

print(all(train_labels == labels.loc[list(tvt_split['train_pids'])]))

val_data = get_full_df(val_data, val_data_mat, tvt_split['val_pids'])
val_data = val_data.reshape(len(val_pids), int(len(val_data)/len(val_pids)), val_data_mat.shape[1])
val_labels = labels.loc[val_pids]
print(val_data.shape, val_labels.shape)

test_data = get_full_df(test_data, test_data_mat, tvt_split['test_pids'])
test_data = test_data.reshape(len(test_pids), int(len(test_data)/len(test_pids)), test_data_mat.shape[1])
test_labels = labels.loc[test_pids]
print(test_data.shape, test_labels.shape)

val_dataset = torch.utils.data.TensorDataset(torch.Tensor(val_data), torch.Tensor(val_labels.values))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4096, shuffle=True)
torch.save(val_loader, int_path+'MDCD_2_10_dl_da_val_loader_shuffled.pth')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4096, shuffle=False)
torch.save(val_loader, int_path+'MDCD_2_10_dl_da_val_loader_unshuffled.pth')

test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_data), torch.Tensor(test_labels.values))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4096, shuffle = False)
torch.save(test_loader, int_path+'MDCD_2_10_dl_da_test_loader_unshuffled.pth')


print(all(val_labels == labels.loc[list(tvt_split['val_pids'])]))
print(all(test_labels == labels.loc[list(tvt_split['test_pids'])]))