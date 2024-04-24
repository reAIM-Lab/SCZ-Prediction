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
import pickle
from joblib import load, dump
import matplotlib
from sklearn.metrics import *

data_path = '../prediction_data/'

# read in population dataframe
num_days_prediction = 90
df_pop = pd.read_csv(data_path+"population.csv")
df_pop['psychosis_diagnosis_date'] = pd.to_datetime(df_pop['psychosis_diagnosis_date'], format="%Y-%m-%d")
df_pop['cohort_start_date'] = pd.to_datetime(df_pop['cohort_start_date'])
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction]

test_labels = pd.read_csv('stored_data/4_19_model_test_output.csv')
df_test = pd.read_csv('stored_data/test_df_4_19.csv')
save_cols = load('stored_data/df_all_iters_columns_8_visits_4_18')
save_cols.remove('iteration')

# STABILITY ANALYSIS
eps = 1e-10

patient_loss = -1*((test_labels['sz_flag']*np.log(test_labels['prob_1']))+(1-test_labels['sz_flag'])*(np.log(1-test_labels['prob_1']+eps)))
test_labels['bce_loss'] = patient_loss

cond_cols = save_cols[0:620]
med_cols = save_cols[620:776]

lab_vals = pd.read_csv('../prediction_data/temporal_labs.csv')
lab_cols = list(lab_vals['concept_name'].unique()) + ['Methadone_Lab']
lab_cols = list(set(lab_cols).intersection(save_cols))

procedure_vals = pd.read_csv('../prediction_data/temporal_procedures.csv')
procedure_cols = list(procedure_vals['concept_name'].unique()) + ['Methadone_Procedure']
procedure_cols = list(set(procedure_cols).intersection(save_cols))
visit_cols = save_cols[1705:]

forward_visit_features = [i for i in visit_cols if 'recent' not in i]
inpatient_features = [i for i in forward_visit_features if 'inpatient' in i]
outpatient_features = [i for i in forward_visit_features if 'outpatient' in i]
ed_features = [i for i in forward_visit_features if 'ed' in i.lower()]
psych_inpatient_features = [i for i in forward_visit_features if 'nonhospitalization' in i]

df_index = ['Conditions', 'Medications', 'Labs', 'Procedures', 'Visits']
list_alphas = 1-np.logspace(-2.5, 0, 30)
results_ci95 = pd.DataFrame(index=df_index, columns=(1-list_alphas))
results_mean = pd.DataFrame(index=df_index, columns=(1-list_alphas))

results_correlation = pd.DataFrame(index=df_index, columns=(1-list_alphas))
results_correlation_ci95 = pd.DataFrame(index=df_index, columns=(1-list_alphas))

df_test = pd.DataFrame(df_test[save_cols])
df_test['sz_flag'] = test_labels['sz_flag']

for a_ind in tqdm(range(len(list_alphas))):
    alpha = list_alphas[a_ind]
    alpha_worst = test_labels['bce_loss'].sort_values(ascending=False).iloc[0:int(np.floor(len(test_labels)*(1-alpha)))]
    test_worst = df_test.loc[alpha_worst.index]
    for list_cols, name_list_cols in zip([cond_cols, med_cols, lab_cols, procedure_cols, forward_visit_features], results_ci95.index):
        
        # getting overall feature values
        grouped_mean = test_worst[list_cols].mean(axis=1)
        ci_low, ci_high = stats.t.interval(0.95, len(test_worst)-1, loc=grouped_mean.mean(), scale=stats.sem(grouped_mean))
        ci95 = ci_high-grouped_mean.mean()
        
        results_mean.loc[name_list_cols, 1-alpha] = grouped_mean.mean()
        results_ci95.loc[name_list_cols, 1-alpha] = ci95
        
        # getting correlation between schizophrenia onset and feature values
        corr_val = test_worst['sz_flag'].corr(grouped_mean)
        corr_ci_low, corr_ci_high = stats.pearsonr(test_worst['sz_flag'], grouped_mean).confidence_interval(confidence_level=0.95)
        corr_ci95 = corr_ci_high-corr_val

        results_correlation.loc[name_list_cols, 1-alpha] = corr_val
        results_correlation_ci95.loc[name_list_cols, 1-alpha] = corr_ci95
        
plt.figure(figsize=(24,4))
font = {'size':20,
       'weight':'bold'}
matplotlib.rc('font', **font)

for i in range(0, 5):
    plt.subplot(1,5,i+1)
    col = results_mean.index[i]
    plt.plot(1-list_alphas, results_mean.loc[col], 'o')
    ci_low = np.asarray(results_mean.loc[col]-results_ci95.loc[col]).astype(float)
    ci_high = np.asarray(results_mean.loc[col]+results_ci95.loc[col]).astype(float)
    plt.fill_between(1-list_alphas, ci_low, ci_high, color='red', alpha=0.5)
    
    plt.ylabel('Feature type mean')
    plt.xlabel('Subpopulation size (1-α)')
    plt.title(col)
    
plt.tight_layout()


plt.figure(figsize=(30,5))
font = {'size':20,
       'weight':'bold'}
matplotlib.rc('font', **font)

for i in range(0, 5):
    plt.subplot(1,5,i+1)
    col = results_mean.index[i]
    plt.plot(1-list_alphas, results_correlation.loc[col], 'o')
    ci_low = np.asarray(results_correlation.loc[col]-results_correlation_ci95.loc[col]).astype(float)
    ci_high = np.asarray(results_correlation.loc[col]+results_correlation_ci95.loc[col]).astype(float)
    plt.fill_between(1-list_alphas, ci_low, ci_high, color='red', alpha = 0.5)
    
    plt.ylabel('Pearson Corr. b/w \n feature group and SCZ')
    plt.xlabel('Subpopulation size (1-α)')
    plt.title(col)
    
plt.tight_layout()
plt.savefig('results/xgboost_stability_correlation.pdf', dpi=300)