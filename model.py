import numpy as np
import os
import pandas as pd
import pyodbc
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import sys
import gc
import pickle
import joblib
from itertools import product
import matplotlib

from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *



# Read in data and concatenate dataframe
list_files = []
list_filenames = os.listdir('stored_data/visit_iters_primary_psych')
for filename_ind in tqdm(range(len(list_filenames))):
    filename = list_filenames[filename_ind]
    list_files.append(pd.read_csv('stored_data/visit_iters_primary_psych/'+filename))


df_all_iters = pd.concat(list_files)
df_all_iters.fillna(0, inplace=True)
print('Unnamed: 0' in df_all_iters.columns)

# train test split
num_days_prediction = 90
df_pop = pd.read_csv(path+'population.csv')
df_pop.rename({'psychosis_dx_date':'psychosis_diagnosis_date'}, axis=1, inplace=True)
df_pop['psychosis_diagnosis_date'] = pd.to_datetime(df_pop['psychosis_diagnosis_date'], format="%Y-%m-%d")
df_pop['cohort_start_date'] = pd.to_datetime(df_pop['cohort_start_date'])
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction]

labels = df_all_iters[['person_id', 'iteration']].merge(df_pop[['person_id','sz_flag']], how='left', left_on = 'person_id', right_on='person_id')
labels.set_index('person_id', inplace=True)

df_all_iters.set_index('person_id', inplace=True)
df_all_iters.drop(['iteration'], inplace=True, axis=1)

pid_train, pid_test, y_train, y_test = train_test_split(df_pop['person_id'], df_pop['sz_flag'], stratify=df_pop['sz_flag'], test_size=0.3, random_state = 4)

X_train = df_all_iters.loc[pid_train.values]
X_test = df_all_iters.loc[pid_test.values]

y_train = labels.loc[pid_train.values, 'sz_flag']
y_test = labels.loc[pid_test.values, 'sz_flag']
save_cols = df_all_iters.columns

# standard scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# grid search to get model
clf = XGBClassifier(seed=3)
params = {max_depth: [3,4,5], n_estimators: [150,200,250,300]}

grid = GridSearchCV(estimator = clf,
    param_grid = params,
    scoring = 'roc_auc',
    n_jobs = -1,
    cv = 5,
    verbose = 3)

grid.fit(X_train, y_train)

with open('models/xgb_6_visits.pkl','wb') as f:
    pickle.dump(grid.best_estimator_,f)