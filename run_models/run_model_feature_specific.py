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

# GET FEATURE-SPECIFIC Columns
cond_cols = list(df_all_iters.columns[1:621])
med_cols = list(df_all_iters.columns[622:777])
df_labs = pd.read_csv('../prediction_data/temporal_labs.csv')
lab_names = list(df_labs['concept_name'].unique())
lab_names += ['Methadone_Lab']
lab_cols = list(set(df_all_iters.columns).intersection(lab_names))

df_procedues = pd.read_csv('../prediction_data/temporal_procedures.csv')
procedure_names = list(df_procedues['concept_name'].unique())
procedure_names += ['Methadone_Procedure']
procedure_cols = list(set(df_all_iters.columns).intersection(procedure_names))

visit_cols = list(df_all_iters.columns[-27:-1])

# remove columns --> replace cond_cols with whichever feature group you're interested in
df_all_iters = df_all_iters[cond_cols+['person_id','iteration']] 

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


# standard scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
print(X_train.shape)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
dump(scaler, 'stored_data/scaler_5_21_conditions.bin', compress=True)


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

with open('models/xgb_8_visits_5_21_conditions.pkl','wb') as f:
    pickle.dump(grid.best_estimator_,f)

### Decide threshold for positive classification
probs = grid.best_estimator_.predict_proba(X_val)
probs = probs[:,1]
fpr, tpr, thresholds = roc_curve(y_val, probs)
idx = np.argmax(tpr - fpr)
cutoff_prob = thresholds[idx]
print(cutoff_prob)
dump(cutoff_prob, 'stored_data/xgb_cutoff_conditions')
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

test_labels.to_csv('stored_data/5_21_model_test_output_conditions.csv')

df_test = pd.DataFrame(X_test, columns=save_cols)
df_test[['person_id', 'iteration', 'sz_flag']] = test_labels.reset_index()[['person_id', 'iteration', 'sz_flag']].values
df_test.to_csv('stored_data/test_df_5_21_conditions.csv')

table2 = pd.DataFrame(columns = ['AUROC', 'AUROC CI', 'Accuracy', 'Accuracy CI',
                                   'Sensitivity', 'Sensitivity CI', 'Specificity', 'Specificity CI',
                                'AUPRC', 'AUPRC_CI', 'PPV', 'PPV_CI'])

table2.loc['All'] = create_table2_row(test_labels)
table2