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
from sklearn.linear_model import LogisticRegression


sys.path.append('../utils')
from eval_utils import *

# Read in data and concatenate dataframe
df_all_iters = pd.read_csv('stored_data/xgboost_all_iters_8_visits_5_21.csv')
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

df_split = pd.read_csv('stored_data/patient_split_5_21.csv')

pid_train = df_split.loc[df_split['split']=='train', 'person_id']
pid_val = df_split.loc[df_split['split']=='val', 'person_id']
pid_test = df_split.loc[df_split['split']=='test', 'person_id']

X_train = df_all_iters.loc[pid_train.values]
X_val = df_all_iters.loc[pid_val.values]
X_test = df_all_iters.loc[pid_test.values]

y_train = labels.loc[pid_train.values, 'sz_flag']
y_val = labels.loc[pid_val.values, 'sz_flag']
y_test = labels.loc[pid_test.values, 'sz_flag']


# Use premade scaler
scaler = load('stored_data/scaler_5_21.bin')
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# grid search to get model
clf = LogisticRegression()
params = {'penalty': ['l1', 'l2', 'elasticnet'], 'C': [0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(estimator = clf,
    param_grid = params,
    scoring = 'roc_auc',
    n_jobs = 3,
    cv = 5,
    verbose = 3)

grid.fit(X_train, y_train)

with open('models/logreg_5_21_fullbaseline.pkl','wb') as f:
    pickle.dump(grid.best_estimator_,f)

### Decide threshold for positive classification
probs = grid.best_estimator_.predict_proba(X_val)
probs = probs[:,1]
fpr, tpr, thresholds = roc_curve(y_val, probs)
idx = np.argmax(tpr - fpr)
cutoff_prob = thresholds[idx]
print(cutoff_prob)
dump(cutoff_prob, 'stored_data/logreg_5_21_fullbaseline_cutoff')
print_performance(X_val, y_val, cutoff_prob)

print_performance(X_test, y_test, cutoff_prob)

# save test labels and transformed test dataset for future use
test_labels = labels.loc[pid_test.values]
probs_test = grid.best_estimator_.predict_proba(X_test)
test_labels['prob_1'] = probs_test[:,1]
test_labels['y_pred'] = test_labels['prob_1'] >= cutoff_prob
print(list(y_test)==list(test_labels['sz_flag']))

# save test labels and transformed test dataset for future use
test_labels = labels.loc[pid_test.values]
probs_test = grid.best_estimator_.predict_proba(X_test)
test_labels['prob_1'] = probs_test[:,1]
test_labels['y_pred'] = test_labels['prob_1'] >= cutoff_prob
print(list(y_test)==list(test_labels['sz_flag']))

test_labels.to_csv('stored_data/5_21_model_test_baselineoutput.csv')
