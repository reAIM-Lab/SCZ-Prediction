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

df_test = df_test.merge(test_labels[['person_id', 'iteration', 'y_pred']], how='inner', left_on = ['person_id', 'iteration'], right_on=['person_id', 'iteration'])
df_test = df_test.merge(df_pop[['person_id', 'race_concept_id', 'gender_concept_id']], how='inner', left_on = 'person_id', right_on='person_id')

# PPV Analysis
def ppv_analysis(test_labels_subset, df_test_subset, save_cols=save_cols):
    mean_diffs = []
    adj_pval = []
    cis = []
    for col in save_cols:
        tp = test_labels_subset.loc[(test_labels_subset['sz_flag']==1)&(test_labels_subset['y_pred']==1)]
        tp = tp.merge(df_test_subset[col], how='left', left_index=True, right_index=True)

        fp = test_labels_subset.loc[(test_labels_subset['sz_flag']==0)&(test_labels_subset['y_pred']==1)]
        fp = fp.merge(df_test_subset[col], how='left', left_index=True, right_index=True)

        mean_diffs.append(tp[col].mean()-fp[col].mean())
        adj_pval.append(stats.ttest_ind(tp[col], fp[col]).pvalue*len(save_cols))
        ci_low, ci_high = stats.ttest_ind(tp[col], fp[col]).confidence_interval()
        cis.append((ci_low, ci_high))

    ppv_results = pd.DataFrame(index=save_cols, columns=['mean difference (TP-FP)', 'adj_pval'])
    ppv_results['mean difference (TP-FP)'] = mean_diffs
    ppv_results['adj_pval'] = adj_pval
    ppv_results['CI'] = cis
    
    return ppv_results.sort_values('mean difference (TP-FP)')

ppv_results_all = ppv_analysis(test_labels, df_test)

# black
df_test_subset = df_test.loc[df_test['race_concept_id']==8516]
test_label_subset = test_labels.loc[test_labels['person_id'].isin(df_test_subset['person_id'])]
ppv_results_black = ppv_analysis(test_label_subset,df_test_subset)

# white
df_test_subset = df_test.loc[df_test['race_concept_id']==8527]
test_label_subset = test_labels.loc[test_labels['person_id'].isin(df_test_subset['person_id'])]
ppv_results_white = ppv_analysis(test_label_subset,df_test_subset)

# missing
df_test_subset = df_test.loc[df_test['race_concept_id']==0]
test_label_subset = test_labels.loc[test_labels['person_id'].isin(df_test_subset['person_id'])]
ppv_results_missing = ppv_analysis(test_label_subset,df_test_subset)

# male
df_test_subset = df_test.loc[df_test['gender_concept_id']==8507]
test_label_subset = test_labels.loc[test_labels['person_id'].isin(df_test_subset['person_id'])]
ppv_results_male = ppv_analysis(test_label_subset,df_test_subset)

# female
df_test_subset = df_test.loc[df_test['gender_concept_id']==8532]
test_label_subset = test_labels.loc[test_labels['person_id'].isin(df_test_subset['person_id'])]
ppv_results_female = ppv_analysis(test_label_subset,df_test_subset)

# merge results
gender_comparison = ppv_results_male.merge(ppv_results_female, how='outer', left_index=True, right_index=True, suffixes = ['_male', '_female'])
gender_comparison['difference'] = gender_comparison['mean difference (TP-FP)_male']-gender_comparison['mean difference (TP-FP)_female']
gender_comparison.sort_values('difference')
gender_comparison['abs_difference'] = np.abs(gender_comparison['difference'])
gender_comparison = gender_comparison.sort_values('difference', ascending=False).head(20)
gender_comparison.to_csv('results/gender_comparison_ppv.csv')

bw_race_comparison = ppv_results_black.merge(ppv_results_white, how='outer', left_index=True, right_index=True, suffixes = ['_black', '_white'])
bw_race_comparison['difference'] = bw_race_comparison['mean difference (TP-FP)_black']-bw_race_comparison['mean difference (TP-FP)_white']
bw_race_comparison['abs_difference'] = np.abs(bw_race_comparison['difference'])
bw_race_comparison = bw_race_comparison.sort_values('difference', ascending=False).head(20)
bw_race_comparison.to_csv('results/bw_race_comparison_ppv.csv')

wm_race_comparison = ppv_results_missing.merge(ppv_results_white, how='outer', left_index=True, right_index=True, suffixes = ['_missing', '_white'])
wm_race_comparison['difference'] = wm_race_comparison['mean difference (TP-FP)_missing']-wm_race_comparison['mean difference (TP-FP)_white']
wm_race_comparison['abs_difference'] = np.abs(wm_race_comparison['difference'])
wm_race_comparison = wm_race_comparison.sort_values('difference', ascending=False).head(20)
wm_race_comparison.to_csv('results/wm_race_comparison_ppv.csv')

bm_race_comparison = ppv_results_missing.merge(ppv_results_black, how='outer', left_index=True, right_index=True, suffixes = ['_missing', '_black'])
bm_race_comparison['difference'] = bm_race_comparison['mean difference (TP-FP)_missing']-bm_race_comparison['mean difference (TP-FP)_black']
bm_race_comparison['abs_difference'] = np.abs(bm_race_comparison['difference'])
bm_race_comparison = bm_race_comparison.sort_values('difference', ascending=False).head(20)
bm_race_comparison.to_csv('results/bm_race_comparison_ppv.csv')
