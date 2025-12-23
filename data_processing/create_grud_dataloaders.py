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

from preprocessing_utils import *

# set up dataset
demo_aware = False

# censor date to cohort start date
num_days_prediction = 90


with open(f'{int_path}/{dataset_prefix}du_snomed_colnames', "rb") as fp:   #Pickling
    data_columns = pickle.load(fp)

print(len(data_columns))

# get split df
df_split = pd.read_csv(f'{int_path}/tvt_split_2dx.csv', index_col=0)
train_pids = list(df_split.loc[df_split['split']=='train', 'person_id'])
val_pids = list(df_split.loc[df_split['split']=='val', 'person_id'])
test_pids = list(df_split.loc[df_split['split']=='test', 'person_id'])
print(len(train_pids)/len(df_split), len(val_pids)/len(df_split), len(test_pids)/len(df_split))

tvt_split = {
    "train_pids": list(train_pids),
    "val_pids": list(val_pids),
    "test_pids": list(test_pids)
}

for split in ['train', 'test', 'val']:
    pids = tvt_split[f'{split}_pids']
    print(type(pids), len(pids))

    # load in actual data
    df_all_iters = pl.read_csv(f'{int_path}/{dataset_prefix}snomed_data.csv')
    df_all_iters = df_all_iters.to_pandas()
    df_all_iters = df_all_iters.loc[df_all_iters['person_id'].isin(pids)]
    print('Done loading in data')

    df_iter_dates = pd.read_csv(f'{int_path}/{dataset_prefix}iteration_dates.csv')
    df_iter_dates = df_iter_dates.loc[df_iter_dates['person_id'].isin(pids)]
    df_just_iters = pd.read_csv(f'{int_path}/{dataset_prefix}time_to_event.csv')
    df_just_iters = df_just_iters.loc[df_just_iters['person_id'].isin(pids)]

    df_pop = pd.read_csv(f'{data_path}/population_2dx.csv', parse_dates = ['psychosis_diagnosis_date', 'scz_diagnosis_date', 'cohort_start_date'])
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

    # import and process timedeltas
    # load in and outer merge for both of them so they are the same length. Merge ON person_id and iteration
    df_timedeltas = pl.read_csv(f'{int_path}/{dataset_prefix}timedeltas.csv')
    df_timedeltas = df_timedeltas.to_pandas()
    df_timedeltas = df_timedeltas.loc[df_timedeltas['person_id'].isin(pids)]
    print('Done loading in timedeltas')

    print(len(df_timedeltas), len(df_all_iters))

    # checking the data to make sure all iterations are present
    print(df_all_iters.isna().sum().sum(), df_timedeltas.isna().sum().sum())
    min_iteration = df_all_iters['iteration'].min()
    max_iteration = df_all_iters['iteration'].max()

    ind_iterations = np.arange(min_iteration, max_iteration+1, 1)
    print(min_iteration, max_iteration, len(ind_iterations))

    df_all_iters['ranked_iteration'] = df_all_iters['iteration'] - min_iteration
    df_timedeltas['ranked_iteration'] = df_timedeltas['iteration'] - min_iteration

    df_all_iters.set_index(['person_id','ranked_iteration'], inplace=True)
    df_timedeltas.set_index(['person_id','ranked_iteration'], inplace=True)

    df_all_iters.sort_index(inplace=True)
    df_timedeltas.sort_index(inplace=True)

    df_all_iters.drop('iteration', axis=1, inplace=True)
    df_timedeltas.drop('iteration', axis=1, inplace=True)

    # fix visits columns in timedeltas
    list_visits = [262, 32036, 581458, 581476, 581478, 9201, 9202, 9203, 42898160, 722455]
    for visit_type in list_visits:
        visit_type = float(visit_type)
        df_timedeltas[f'{visit_type}_MH_num_visits'] = df_timedeltas[f'{visit_type}_MH'].copy()
        df_timedeltas[f'{visit_type}_ALL_num_visits'] = df_timedeltas[f'{visit_type}_ALL'].copy()
        df_timedeltas[f'{visit_type}_MH_los'] = df_timedeltas[f'{visit_type}_MH'].copy()
        df_timedeltas[f'{visit_type}_ALL_los'] = df_timedeltas[f'{visit_type}_ALL'].copy()

    if demo_aware == True:
        df_timedeltas['is_White'] = 0
        df_timedeltas['is_Black'] = 0
        df_timedeltas['is_Male'] = 0

    df_all_iters = df_all_iters[data_columns]
    df_timedeltas = df_timedeltas[data_columns]

    scaler = load(f'{int_path}/{save_prefix}du_dl_scaler.bin')
    data_mask = (df_all_iters > 0) * 1
    if demo_aware == True:
        data_mask[['is_White', 'is_Black', 'is_Male']] = 1
    df_all_iters = pd.DataFrame(scaler.transform(df_all_iters), columns = data_columns, index = df_all_iters.index)

    print('starting object creation')
    dataset_obj = GRUDDataset_Object(df_just_iters, df_pop, data_columns, min_iteration*-1, ind_iterations, 4)
    print('running create dataset')
    dataset = dataset_obj.create_dataset_object(pids, df_all_iters, data_mask, df_timedeltas)
    print('making loader')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle = True, pin_memory = True, persistent_workers = True, num_workers = 4)
    print('saving loader')
    torch.save(loader, f'{int_path}/{save_prefix}{dl_prefix}grud_{split}_loader.pth')

    if split == 'train':
        # get the means for X_data GRUD
        X_data = dataset.tensors[1][:,0,:,:]
        X_mask = dataset.tensors[2].unsqueeze(-1).expand(-1, -1, X_data.shape[-1])
        print(X_data.shape, X_mask.shape)
        masked_X = X_data * X_mask 
        sum_vals = masked_X.sum(dim=0) # Step 3: sum over batch only where mask == 1
        count_vals = X_mask.sum(dim=0).clamp(min=1)  # avoid divide by 0
        mean_vals = sum_vals / count_vals        # (sequence, features)
        mean_vals = mean_vals.unsqueeze(0)       # (1, sequence, features)
        print(mean_vals.shape)
        torch.save(mean_vals, f'{int_path}/{save_prefix}{dl_prefix}grud_means.pt')
