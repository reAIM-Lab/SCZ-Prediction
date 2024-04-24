import numpy as np
import os
import pandas as pd
import pyodbc
import time
import scipy.stats as stats
from datetime import datetime
from tqdm import tqdm
import sys
import gc
import pickle
from joblib import dump, load
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

store_data = 'stored_data/'

# Read in data 
df_all_iters = pd.read_csv(store_data+'xgboost_all_iters_8_visits_4_17.csv')
df_all_iters.fillna(0, inplace=True)
if 'Unnamed: 0' in df_all_iters.columns:
    df_all_iters.drop(['Unnamed: 0'], axis=1, inplace=True)

# train test split
num_days_prediction = 90
df_pop = pd.read_csv('../prediction_data/population.csv')
df_pop['psychosis_diagnosis_date'] = pd.to_datetime(df_pop['psychosis_diagnosis_date'], format="%Y-%m-%d")
df_pop['cohort_start_date'] = pd.to_datetime(df_pop['cohort_start_date'])
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction]

labels = df_all_iters[['person_id', 'iteration']].merge(df_pop[['person_id','sz_flag']], how='left', left_on = 'person_id', right_on='person_id')
labels.set_index('person_id', inplace=True)

df_all_iters.set_index('person_id', inplace=True)
df_all_iters.drop(['iteration'], inplace=True, axis=1)

pid_trainval, pid_test, y_trainval, y_test = train_test_split(df_pop['person_id'], df_pop['sz_flag'], stratify=df_pop['sz_flag'], test_size=0.2, random_state = 4)
trainval_pop = df_pop.loc[df_pop['person_id'].isin(pid_trainval)]
pid_train, pid_val, y_train, y_val = train_test_split(trainval_pop['person_id'], trainval_pop['sz_flag'], stratify=trainval_pop['sz_flag'], test_size=1/7, random_state = 4)

X_train = df_all_iters.loc[pid_train.values]
X_val = df_all_iters.loc[pid_val.values]
X_test = df_all_iters.loc[pid_test.values]

y_train = labels.loc[pid_train.values, 'sz_flag']
y_val = labels.loc[pid_val.values, 'sz_flag']
y_test = labels.loc[pid_test.values, 'sz_flag']
save_cols = df_all_iters.columns

# save the train/test split
df_split = pd.concat([pd.DataFrame(index=pid_train, columns = ['split'], data = 'train'),
                      pd.DataFrame(index=pid_val, columns = ['split'], data='val'), 
           pd.DataFrame(index=pid_test, columns = ['split'], data = 'test')])
df_split.to_csv(store_data+'patient_split_4_19.csv')

# standard scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
dump(scaler, store_data+'scaler_4_19.bin', compress=True)

# grid search to get model
clf = XGBClassifier(seed=3)
params = {'max_depth': [3,4,5], 'n_estimators': [200,300]}

grid = GridSearchCV(estimator = clf,
    param_grid = params,
    scoring = 'roc_auc',
    n_jobs = 3,
    cv = 5,
    verbose = 3)

grid.fit(X_train, y_train)

with open('models/xgb_8_visits_4_19.pkl','wb') as f:
    pickle.dump(grid.best_estimator_,f)

### Decide threshold for positive classification
probs = grid.best_estimator_.predict_proba(X_val)
probs = probs[:,1]
fpr, tpr, thresholds = roc_curve(y_val, probs)
idx = np.argmax(tpr - fpr)
cutoff_prob = thresholds[idx]
print(cutoff_prob)
dump(cutoff_prob, store_data + 'xgb_8_visits_4_19_cutoff')

# save test labels and transformed test dataset for future use
test_labels = labels.loc[pid_test.values]
probs_test = grid.best_estimator_.predict_proba(X_test)
test_labels['prob_1'] = probs_test[:,1]
test_labels['y_pred'] = test_labels['prob_1'] >= cutoff_prob
print(list(y_test)==list(test_labels['sz_flag']))

test_labels.to_csv(store_data+'4_19_model_test_output.csv')

df_test = pd.DataFrame(X_test, columns=save_cols)
df_test[['person_id', 'iteration', 'sz_flag']] = test_labels.reset_index()[['person_id', 'iteration', 'sz_flag']].values
df_test.to_csv(store_data+'test_df_4_19.csv')
