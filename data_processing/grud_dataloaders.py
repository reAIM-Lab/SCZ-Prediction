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
import json

shared_cols = True
make_scaler = False

bw_only = False
bm_only = False
wm_only = False
mf_only = False

demo_aware = True
keep_iters = None # [-10, 31]

"paths"

df_all_iters = pd.read_csv(int_path + 'MDCD_12_1_dl_data_snomed.csv')
df_all_iters.drop('Unnamed: 0', axis=1, inplace=True)
df_all_iters = df_all_iters.sort_values(['person_id', 'iteration'])

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

# load in timedeltas
# load in and outer merge for both of them so they are the same length. Merge ON person_id and iteration
df_timedeltas = pd.read_csv(int_path + 'MDCD_12_1_grud_timedeltas_snomed.csv')
df_timedeltas = df_timedeltas.sort_values(['person_id', 'iteration'])

if keep_iters is not None:
    df_timedeltas = df_timedeltas.loc[df_timedeltas['iteration']>=keep_iters[0]]
    df_timedeltas = df_timedeltas.loc[df_timedeltas['iteration']<=keep_iters[1]]
    print('Check keep iters', df_timedeltas['iteration'].min(), df_timedeltas['iteration'].max())

print(len(df_timedeltas), len(df_all_iters))
df_timedeltas = df_timedeltas.merge(df_all_iters[['person_id','iteration']], how='outer', on=['person_id','iteration'])
print(len(df_timedeltas), len(df_all_iters))
df_all_iters = df_all_iters.merge(df_timedeltas[['person_id','iteration']], how='outer', on=['person_id','iteration'])
print(len(df_timedeltas), len(df_all_iters))

df_timedeltas = df_timedeltas.loc[df_timedeltas['person_id'].isin(df_pop['person_id'])]
df_all_iters = df_all_iters.loc[df_all_iters['person_id'].isin(df_pop['person_id'])]
print(len(df_timedeltas), len(df_all_iters))

df_all_iters.fillna(0, inplace=True)
time_per_iter = 120
df_all_iters['time_since_psychosis'] = df_all_iters['iteration'] * time_per_iter/365
df_all_iters.loc[df_all_iters['iteration'] <= 0, 'time_since_psychosis'] = 0
# check that time_since_psychosis is correct
print(df_all_iters['time_since_psychosis'].unique())
print('\n\nPre-psychosis tsp', df_all_iters.loc[df_all_iters['iteration']<=0, 'time_since_psychosis'].unique())
print('\n\nPost-psychosis tsp', df_all_iters.loc[df_all_iters['iteration']>0, 'time_since_psychosis'].unique())

df_timedeltas.fillna(0, inplace=True)
df_timedeltas['time_since_psychosis'] = df_timedeltas['iteration'] * time_per_iter/365
df_timedeltas.loc[df_timedeltas['iteration'] <= 0, 'time_since_psychosis'] = 0
# check that time_since_psychosis is correct
print(df_timedeltas['time_since_psychosis'].unique())
print('\n\nPre-psychosis tsp', df_timedeltas.loc[df_timedeltas['iteration']<=0, 'time_since_psychosis'].unique())
print('\n\nPost-psychosis tsp', df_timedeltas.loc[df_timedeltas['iteration']>0, 'time_since_psychosis'].unique())


ranked_vals = df_timedeltas.reset_index().groupby('person_id')['iteration'].rank(method='first').values
df_timedeltas['ranked_iteration'] = ranked_vals
overall_max = df_timedeltas['ranked_iteration'].max()
print(overall_max)

df_timedeltas = df_timedeltas.sort_values(['person_id', 'ranked_iteration'])
df_timedeltas.set_index(['person_id','ranked_iteration'], inplace=True)
df_timedeltas.sort_index(inplace=True)


ranked_vals = df_all_iters.reset_index().groupby('person_id')['iteration'].rank(method='first').values
df_all_iters['ranked_iteration'] = ranked_vals
overall_max = df_all_iters['ranked_iteration'].max()
print(overall_max)

df_all_iters = df_all_iters.sort_values(['person_id', 'ranked_iteration'])
df_all_iters.set_index(['person_id','ranked_iteration'], inplace=True)
df_all_iters.sort_index(inplace=True)

# check that there is at most a difference of 1 between each pid from one iteration to the next
def find_largest_diff(df):
    # Sort by pid and iteration
    df_sorted = df.sort_values(by=['person_id', 'iteration'])
    
    # Calculate the largest difference for each pid
    result = df_sorted.groupby('person_id')['iteration'].apply(
        lambda x: x.diff().max()
    ).reset_index(name='largest_diff')
    
    return result
print('Check largest difference', find_largest_diff(df_all_iters)['largest_diff'].max()) # should be 1)

print('Checking Match:', (df_all_iters['iteration'].reset_index().values == df_timedeltas['iteration'].reset_index().values).all())
df_all_iters.drop('iteration', axis=1, inplace=True)
df_timedeltas.drop('iteration', axis=1, inplace=True)

print(df_all_iters.isna().sum().sum(), df_timedeltas.isna().sum().sum())

# split and scale data
df_split = pd.read_csv(int_path + 'MDCD_10_30_tvt_split.csv')
df_split = df_split.loc[df_split['person_id'].isin(df_all_iters.index.get_level_values(0))]
train_pids = list(df_split.loc[df_split['split']=='train', 'person_id'])
val_pids = list(df_split.loc[df_split['split']=='val', 'person_id'])
test_pids = list(df_split.loc[df_split['split']=='test', 'person_id'])
print(len(train_pids)/len(df_split), len(val_pids)/len(df_split), len(test_pids)/len(df_split))

tvt_split = {
    "train_pids": tuple(train_pids),
    "val_pids": tuple(val_pids),
    "test_pids": tuple(test_pids)
}
with open(int_path + "MDCD_2_10_grud_dl_da_tvt_order.json", "w") as f:
    json.dump(tvt_split, f)


if shared_cols == False:
    save_cols = list(df_all_iters.columns)

    with open(int_path + "CCAE_1_27_grud_dl_individualfeats_colnames", "wb") as fp:   #Pickling
        pickle.dump(save_cols, fp)
else: 
    save_cols = load(int_path + 'MDCD_2_10_dl_da_colnames')

df_all_iters = df_all_iters[save_cols]    
print('Check for unnamed col (should be False):', 'Unnamed: 0' in save_cols)
print(len(save_cols))

# get the actual data points
train_data = df_all_iters.loc[train_pids]
val_data = df_all_iters.loc[val_pids]
test_data = df_all_iters.loc[test_pids]

# get the masks of the data points
train_data_mask = (train_data>0)*1
val_data_mask = (val_data>0)*1
test_data_mask = (test_data>0)*1

if make_scaler:
    scaler = StandardScaler()
    train_data_mat = scaler.fit_transform(train_data)
    print('done with fit/first transform')
    val_data_mat = scaler.transform(val_data)
    test_data_mat = scaler.transform(test_data)

    # save the standard scaler
    dump(scaler, int_path + 'CCAE_1_27_grud_dl_individualfeats_scaler.bin', compress=True)
    
else:
    scaler = load(int_path + 'MDCD_2_10_dl_da_scaler.bin')
    train_data_mat = scaler.transform(train_data)
    val_data_mat = scaler.transform(val_data)
    test_data_mat = scaler.transform(test_data)

# fix up timedelta problems
for i in set(save_cols).difference(df_timedeltas.columns):
    if '42898160' in i:
        df_timedeltas[i] = df_timedeltas['42898160']
    elif 'num_visits_nonhospital' in i:
        df_timedeltas['num_visits_nonhospital'] = df_timedeltas['42898160']
    elif '9201' in i:
        df_timedeltas[i] = df_timedeltas['9201']
    elif 'num_visits_inpatient' in i:
        df_timedeltas['num_visits_inpatient'] = df_timedeltas['9201']
    elif '9202' in i:
        df_timedeltas[i] = df_timedeltas['9202']
    elif 'num_visits_outpatient' in i:
        df_timedeltas['num_visits_outpatient'] = df_timedeltas['9202']
    elif '9203' in i:
        df_timedeltas[i] = df_timedeltas['9203']
    elif 'num_visits_ED' in i:
        df_timedeltas['num_visits_ED'] = df_timedeltas['9203']
    elif 'psych' in i:
        df_timedeltas[i] = df_timedeltas['psych_visits']
    elif '262' in i:
        df_timedeltas[i] = df_timedeltas['262.0']
    elif '38004222' in i:
        df_timedeltas[i] = df_timedeltas['38004222.0']
    elif '38004228' in i:
        df_timedeltas[i] = df_timedeltas['38004228.0']
    elif '38004238' in i:
        df_timedeltas[i] = df_timedeltas['38004238.0']
    elif '38004250' in i:
        df_timedeltas[i] = df_timedeltas['38004250.0']
    elif '8883' in i:
        df_timedeltas[i] = df_timedeltas['8883.0']
    elif '8971' in i:
        df_timedeltas[i] = df_timedeltas['8971.0']
    elif '5083' in i:
        df_timedeltas[i] = df_timedeltas['5083.0']
    elif '581477' in i:
        df_timedeltas[i] = df_timedeltas['581477.0'] 
    else:
        print(i)

if demo_aware == True:
    df_timedeltas['is_Male'] = 0
    df_timedeltas['is_Black'] = 0
    df_timedeltas['is_White'] = 0

# divide timedeltas into train/test/split
df_timedeltas = df_timedeltas[save_cols]
print(df_timedeltas.shape, len(save_cols))
# get the actual data points
train_tds = df_timedeltas.loc[train_pids]
val_tds = df_timedeltas.loc[val_pids]
test_tds = df_timedeltas.loc[test_pids]

# pad the data
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
print(df_all_iters.isna().sum().sum(), df_timedeltas.isna().sum().sum())
print('Unnamed: 0' in df_timedeltas.columns) 

val_data_data = get_full_df(val_data, val_data_mat, tvt_split['val_pids'])
val_data_data = val_data_data.reshape(len(val_pids), int(len(val_data_data)/len(val_pids)), val_data_mat.shape[1])

val_data_mask = get_full_df(val_data_mask, val_data_mask.values, val_pids)
val_data_mask = val_data_mask.reshape(len(val_pids), int(len(val_data_mask)/len(val_pids)), val_data_mask.shape[1])

val_last_obs = torch.Tensor(val_data_data * val_data_mask)
val_last_obs_mat = torch.cat((torch.zeros((val_last_obs.shape[0], 1, val_last_obs.shape[2])), val_last_obs), 1)
val_last_obs_mat = val_last_obs_mat[:,0:-1,:]
print(val_last_obs_mat.shape)

val_data_deltas = get_full_df(val_tds, val_tds.values, tvt_split['val_pids'])
val_data_deltas = val_data_deltas.reshape(len(val_pids), int(len(val_data_deltas)/len(val_pids)), val_tds.shape[1])

val_labels = labels.loc[val_pids]
stacked_val_data = np.stack([val_data_data, val_last_obs_mat, val_data_mask, val_data_deltas], axis=1)
print(stacked_val_data.shape, val_labels.shape)

test_data_data = get_full_df(test_data, test_data_mat, tvt_split['test_pids'])
test_data_data = test_data_data.reshape(len(test_pids), int(len(test_data_data)/len(test_pids)), test_data_mat.shape[1])

test_data_mask = get_full_df(test_data_mask, test_data_mask.values, test_pids)
test_data_mask = test_data_mask.reshape(len(test_pids), int(len(test_data_mask)/len(test_pids)), test_data_mask.shape[1])

test_last_obs = torch.Tensor(test_data_data * test_data_mask)
test_last_obs_mat = torch.cat((torch.zeros((test_last_obs.shape[0], 1, test_last_obs.shape[2])), test_last_obs), 1)
test_last_obs_mat = test_last_obs_mat[:,0:-1,:]

test_data_deltas = get_full_df(test_tds, test_tds.values, tvt_split['test_pids'])
test_data_deltas = test_data_deltas.reshape(len(test_pids), int(len(test_data_deltas)/len(test_pids)), test_data_deltas.shape[1])

test_labels = labels.loc[test_pids]
stacked_test_data = np.stack([test_data_data, test_last_obs_mat, test_data_mask, test_data_deltas], axis=1)
print(stacked_test_data.shape, test_labels.shape)

val_dataset = torch.utils.data.TensorDataset(torch.Tensor(stacked_val_data), torch.Tensor(val_labels.values))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2048, shuffle=True)
torch.save(val_loader, int_path+'MDCD_2_10_grud_dl_da_val_loader_shuffled.pth')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2048, shuffle=False)
torch.save(val_loader, int_path+'MDCD_2_10_grud_dl_da_val_loader_unshuffled.pth')

test_dataset = torch.utils.data.TensorDataset(torch.Tensor(stacked_test_data), torch.Tensor(test_labels.values))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle = False)
torch.save(test_loader, int_path+'MDCD_2_10_grud_dl_da_test_loader_unshuffled.pth')

train_data_data = get_full_df(train_data, train_data_mat, tvt_split['train_pids'])
train_data_data = train_data_data.reshape(len(train_pids), int(len(train_data_data)/len(train_pids)), train_data_mat.shape[1])

train_data_mask = get_full_df(train_data_mask, train_data_mask.values, tvt_split['train_pids'])
train_data_mask = train_data_mask.reshape(len(train_pids), int(len(train_data_mask)/len(train_pids)), train_data_mask.shape[1])

train_last_obs = torch.Tensor(train_data_data * train_data_mask)
train_last_obs_mat = torch.cat((torch.zeros((train_last_obs.shape[0], 1, train_last_obs.shape[2])), train_last_obs), 1)
train_last_obs_mat = train_last_obs_mat[:,0:-1,:]
print(train_last_obs_mat.shape)

train_data_deltas = get_full_df(train_tds, train_tds.values, tvt_split['train_pids'])
train_data_deltas = train_data_deltas.reshape(len(train_pids), int(len(train_data_deltas)/len(train_pids)), train_data_deltas.shape[1])

train_labels = labels.loc[train_pids]
stacked_train_data = np.stack([train_data_data, train_last_obs_mat, train_data_mask, train_data_deltas], axis=1)
print(stacked_train_data.shape, train_labels.shape)

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

torch.save(mean_matrix, int_path + 'MDCD_2_10_grud_dl_da_means.pt')