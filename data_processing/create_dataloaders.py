import numpy as np
import os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import sys
import gc
import pickle 
import torch
from joblib import dump, load
import json
import polars as pl

sys.path.append('../')
from preprocessing_utils import *

# set up dataset
# PATHS


make_scaler = True
demo_aware = False

# censor date to cohort start date
num_days_prediction = 90
"""
with open(f'{int_path}/{dataset_prefix}du_snomed_colnames', "rb") as fp:   #Pickling
    data_columns = pickle.load(fp)
"""
with open(f'{path}/intermediate_data_mdcd_3yrs/9_26_mdcd_2dx_fullhistory_du_snomed_colnames', "rb") as fp:   #Pickling
    data_columns = pickle.load(fp)
print(len(data_columns))

# get split df
df_split = pd.read_csv(f'{int_path}/tvt_split_2dx.csv', index_col=0)
train_pids = list(df_split.loc[df_split['split']=='train', 'person_id'])
val_pids = list(df_split.loc[df_split['split']=='val', 'person_id'])
test_pids = list(df_split.loc[df_split['split']=='test', 'person_id'])
print(len(train_pids)/len(df_split), len(val_pids)/len(df_split), len(test_pids)/len(df_split))

tvt_split = {
    "train_pids": tuple(train_pids),
    "val_pids": tuple(val_pids),
    "test_pids": tuple(test_pids)
}

# load in actual data
df_pl = pl.read_csv(f'{int_path}/{dataset_prefix}snomed_data.csv')
print('Done loading in polars')
df_all_iters = df_pl.to_pandas()
df_all_iters = df_all_iters[['person_id', 'iteration'] + data_columns]
print('Done loading in data')

df_iter_dates = pd.read_csv(f'{int_path}/{dataset_prefix}iteration_dates.csv')
df_just_iters = pd.read_csv(f'{int_path}/{dataset_prefix}time_to_event.csv')

df_pop = pd.read_csv(f'{data_path}/population_2dx.csv', parse_dates = ['psychosis_diagnosis_date', 'scz_diagnosis_date', 'cohort_start_date'])
print('checking df_pop')
print(len(val_pids), len(df_pop.loc[df_pop['person_id'].isin(val_pids)]), len(df_all_iters.loc[df_all_iters['person_id'].isin(val_pids), 'person_id'].unique()))
print(len(train_pids), len(df_pop.loc[df_pop['person_id'].isin(train_pids)]), len(df_all_iters.loc[df_all_iters['person_id'].isin(train_pids), 'person_id'].unique()))
print(len(test_pids), len(df_pop.loc[df_pop['person_id'].isin(test_pids)]), len(df_all_iters.loc[df_all_iters['person_id'].isin(test_pids), 'person_id'].unique()))

df_pop.loc[df_pop['person_id'].isin(set(df_all_iters['person_id']))]

print(len(df_all_iters))

if demo_aware == True:
    df_pop['is_White'] = 0
    df_pop.loc[df_pop['race_concept_id']==8527, 'is_White'] = 1
    df_pop['is_Black'] = 0
    df_pop.loc[df_pop['race_concept_id']==8516, 'is_Black'] = 1
    df_pop['is_Male'] = 0
    df_pop.loc[df_pop['gender_concept_id']==8507, 'is_Male'] = 1
    
    df_all_iters = df_all_iters.merge(df_pop[['person_id', 'is_White', 'is_Black', 'is_Male']], how='inner', left_on = 'person_id', right_on='person_id')
    print(len(df_all_iters))

# checking the data to make sure all iterations are present
print(df_all_iters.isna().sum().sum())
min_iteration = df_all_iters['iteration'].min()
max_iteration = df_all_iters['iteration'].max()

ind_iterations = np.arange(min_iteration, max_iteration+1, 1)
print(min_iteration, max_iteration, len(ind_iterations))

df_all_iters['ranked_iteration'] = df_all_iters['iteration'] - min_iteration

df_all_iters.set_index(['person_id','ranked_iteration'], inplace=True)
df_all_iters.sort_index(inplace=True)
print('Check largest difference', find_largest_diff(df_all_iters)['largest_diff'].max()) # should be 1
df_all_iters.drop('iteration', axis=1, inplace=True)

# SCALE DATA
df_all_iters = df_all_iters[data_columns]

# get the actual data points
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
    dump(scaler, f'{int_path}/{save_prefix}{deeplearning_prefix}scaler.bin', compress=True)
    
else:
    scaler = load(f'{path}/intermediate_data_mdcd_3yrs/9_26_mdcd_2dx_medianhistory_top1000du_dl_scaler.bin')
    train_data_mat = scaler.transform(train_data)
    val_data_mat = scaler.transform(val_data)
    test_data_mat = scaler.transform(test_data)

"""
### Pad the data
- Array of patient IDs (pids x 1) 
- Padded array (X): pids x time sequence x features
- Mask: Binary pids x time sequence; 1 if that time is observed, 0 otherwise 
- Time to event: pids x time sequence, TTE o if it should be masked
- Y: Event, binary (pids x 1)
"""

del df_all_iters
del df_pl
gc.collect()


datasets = Dataset_Object(df_just_iters, df_pop, data_columns, min_iteration*-1, ind_iterations, 4)
val_dataset = datasets.create_dataset_object(val_pids, val_data_mat, val_data)
test_dataset = datasets.create_dataset_object(test_pids, test_data_mat, test_data)
train_dataset = datasets.create_dataset_object(train_pids, train_data_mat, train_data)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle = True, pin_memory = True, num_workers = 4, persistent_workers = True)
torch.save(train_loader, f'{int_path}/{save_prefix}{deeplearning_prefix}train_loader.pth')

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle = True, pin_memory = True, num_workers = 4, persistent_workers = True)
torch.save(val_loader, f'{int_path}/{save_prefix}{deeplearning_prefix}val_loader.pth')

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle = True, pin_memory = True, num_workers = 4, persistent_workers = True)
torch.save(test_loader, f'{int_path}/{save_prefix}{deeplearning_prefix}test_loader.pth')
