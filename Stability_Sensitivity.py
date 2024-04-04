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

path = '../../'

# load testing data
test_pids = np.load('../stored_data/test_pids_4_2_24.npy')

list_files = []
list_filenames = os.listdir('../stored_data/visit_iters_6')
for filename_ind in tqdm(range(len(list_filenames))):
    filename = list_filenames[filename_ind]
    list_files.append(pd.read_csv('../stored_data/visit_iters_6/'+filename))
df_all_iters = pd.concat(list_files)
df_all_iters.fillna(0, inplace=True)

num_days_prediction = 90
df_pop = pd.read_csv(path+'population.csv')
df_pop.rename({'psychosis_dx_date':'psychosis_diagnosis_date'}, axis=1, inplace=True)
df_pop['psychosis_diagnosis_date'] = pd.to_datetime(df_pop['psychosis_diagnosis_date'], format="%Y-%m-%d")
df_pop['cohort_start_date'] = pd.to_datetime(df_pop['cohort_start_date'])
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction]

df_pop = df_pop.loc[df_pop['person_id'].isin(df_all_iters['person_id'])]

labels = df_all_iters[['person_id', 'iteration']].merge(df_pop[['person_id','sz_flag']], how='left', left_on = 'person_id', right_on='person_id')
labels.set_index('person_id', inplace=True)

df_all_iters.set_index('person_id', inplace=True)
df_all_iters.drop(['iteration'], inplace=True, axis=1)
save_cols = df_all_iters.columns

X_test = df_all_iters.loc[test_pids]
y_test = labels.loc[test_pids, 'sz_flag']

scaler = load('../stored_data/scaler_4_2_24.bin')
X_test = scaler.transform(X_test)

df_iter_pop = pd.read_csv('../stored_data/iterated_population_6_visits.csv')
iterated_testing_df = df_iter_pop.loc[df_iter_pop['person_id'].isin(test_pids)]

# Run model on test set
with open('../models/xgb_every_6_visits.pkl', 'rb') as f:
    testing_clf = pickle.load(f)

y_pred_proba = testing_clf.predict_proba(X_test)
test_labels = labels.loc[test_pids]

test_labels['prob_1'] = y_pred_proba[:,1]
test_labels['y_pred'] = np.round(test_labels['prob_1'])
test_labels.reset_index(inplace=True)
test_labels = test_labels.merge(iterated_testing_df[['person_id', 'iteration', 'years_obs', 'psychosis_diagnosis_date', 'cutoff_date', 'cohort_start_date','censor_date', 'first_visit']], how = 'left', left_on = ['person_id', 'iteration'], right_on = ['person_id', 'iteration'])

test_labels = test_labels.merge(df_pop[['person_id', 'race_concept_id', 'gender_concept_id']], how='left',
                 left_on='person_id', right_on='person_id')


# STABILITY ANALYSIS
eps = 1e-10

patient_loss = -1*((test_labels['sz_flag']*np.log(test_labels['prob_1']))+(1-test_labels['sz_flag'])*(np.log(1-test_labels['prob_1']+eps)))
test_labels['bce_loss'] = patient_loss

cond_cols = save_cols[0:598]
med_cols = save_cols[598:751]
lab_cols = save_cols[751:1481]
visit_cols = save_cols[1481:1507]
procedure_cols = save_cols[1507:]

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

df_test = pd.DataFrame(X_test, columns=save_cols)
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
plt.savefig('xgboost_stability_correlation.pdf', dpi=300)

# Sensitivity Analysis
mean_diffs = []
adj_pval = []
cis = []
for col in save_cols:
    tp = test_labels.loc[(test_labels['sz_flag']==1)&(test_labels['y_pred']==1)]
    tp = tp.merge(df_test[col], how='left', left_index=True, right_index=True)

    fn = test_labels.loc[(test_labels['sz_flag']==1)&(test_labels['y_pred']==0)]
    fn = fn.merge(df_test[col], how='left', left_index=True, right_index=True)
    
    mean_diffs.append(tp[col].mean()-fn[col].mean())
    adj_pval.append(stats.ttest_ind(tp[col], fn[col]).pvalue*len(save_cols))
    ci_low, ci_high = stats.ttest_ind(tp[col], fn[col]).confidence_interval()
    cis.append((ci_low, ci_high))
    
sensitivity_results = pd.DataFrame(index=save_cols, columns=['mean difference (TP-FN)', 'adj_pval'])
sensitivity_results['mean difference (TP-FN)'] = mean_diffs
sensitivity_results['adj_pval'] = adj_pval
sensitivity_results['CI'] = cis

sensitivity_results = sensitivity_results.loc[sensitivity_results['adj_pval']<0.01]

print('Medications:',len(set(med_cols).intersection(sensitivity_results.index))/len(med_cols))
print('Conditions:',len(set(cond_cols).intersection(sensitivity_results.index))/len(cond_cols))
print('Labs:',len(set(lab_cols).intersection(sensitivity_results.index))/len(lab_cols))
print('Procedures:',len(set(procedure_cols).intersection(sensitivity_results.index))/len(procedure_cols))
print('Visits:',len(set(visit_cols).intersection(sensitivity_results.index))/len(visit_cols))

sensitivity_results = sensitivity_results.loc[np.abs(sensitivity_results['mean difference (TP-FN)']) > 1]

# Mean difference (Black Pos-Neg) + pvals + CI # Mean Difference (White Pos-Neg) + pvals + CI

mean_diffs_black = []
adj_pval_black = []
cis_black = []

mean_diffs_white = []
adj_pval_white = []
cis_white = []

mean_diffs_female = []
adj_pval_female = []
cis_female = []

mean_diffs_male = []
adj_pval_male = []
cis_male = []


for col in sensitivity_results.index:
    black_pos = test_labels.loc[(test_labels['sz_flag']==1)&(test_labels['race_concept_id']==8516)]
    black_pos = black_pos.merge(df_test[col], how='left', left_index=True, right_index=True)
    
    white_pos = test_labels.loc[(test_labels['sz_flag']==1)&(test_labels['race_concept_id']==8527)]
    white_pos = white_pos.merge(df_test[col], how='left', left_index=True, right_index=True)

    female_pos = test_labels.loc[(test_labels['sz_flag']==1)&(test_labels['gender_concept_id']==8532)]
    female_pos = female_pos.merge(df_test[col], how='left', left_index=True, right_index=True)
    
    male_pos = test_labels.loc[(test_labels['sz_flag']==1)&(test_labels['gender_concept_id']==8507)]
    male_pos = male_pos.merge(df_test[col], how='left', left_index=True, right_index=True)
    
    all_neg = test_labels.loc[(test_labels['sz_flag']==0)]
    all_neg = all_neg.merge(df_test[col], how='left', left_index=True, right_index=True)
    
    mean_diffs_black.append(black_pos[col].mean()-all_neg[col].mean())
    adj_pval_black.append(stats.ttest_ind(black_pos[col], all_neg[col]).pvalue*len(save_cols))
    ci_low, ci_high = stats.ttest_ind(black_pos[col], all_neg[col]).confidence_interval()
    cis_black.append((ci_low, ci_high))
    
    mean_diffs_white.append(white_pos[col].mean()-all_neg[col].mean())
    adj_pval_white.append(stats.ttest_ind(white_pos[col], all_neg[col]).pvalue*len(save_cols))
    ci_low, ci_high = stats.ttest_ind(white_pos[col], all_neg[col]).confidence_interval()
    cis_white.append((ci_low, ci_high))

    
    mean_diffs_female.append(female_pos[col].mean()-all_neg[col].mean())
    adj_pval_female.append(stats.ttest_ind(female_pos[col], all_neg[col]).pvalue*len(save_cols))
    ci_low, ci_high = stats.ttest_ind(female_pos[col], all_neg[col]).confidence_interval()
    cis_female.append((ci_low, ci_high))

    
    mean_diffs_male.append(male_pos[col].mean()-all_neg[col].mean())
    adj_pval_male.append(stats.ttest_ind(male_pos[col], all_neg[col]).pvalue*len(save_cols))
    ci_low, ci_high = stats.ttest_ind(male_pos[col], all_neg[col]).confidence_interval()
    cis_male.append((ci_low, ci_high))

    
sensitivity_results['mean difference (TP-FN)_black'] = mean_diffs_black
sensitivity_results['adj_pval_black'] = adj_pval_black
sensitivity_results['CI_black'] = cis_black

sensitivity_results['mean difference (TP-FN)_white'] = mean_diffs_white
sensitivity_results['adj_pval_white'] = adj_pval_white
sensitivity_results['CI_white'] = cis_white

sensitivity_results['mean difference (TP-FN)_female'] = mean_diffs_female
sensitivity_results['adj_pval_female'] = adj_pval_female
sensitivity_results['CI_female'] = cis_female

sensitivity_results['mean difference (TP-FN)_male'] = mean_diffs_male
sensitivity_results['adj_pval_male'] = adj_pval_male
sensitivity_results['CI_male'] = cis_male
