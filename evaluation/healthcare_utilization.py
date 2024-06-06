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

connection_string = ('YOUR CREDENTIALS HERE')
conn = pyodbc.connect(connection_string)

data_path = '../prediction_data/'

# read in population dataframe
num_days_prediction = 90
df_pop = pd.read_csv(data_path+"population.csv")
df_pop['psychosis_diagnosis_date'] = pd.to_datetime(df_pop['psychosis_diagnosis_date'], format="%Y-%m-%d")
df_pop['cohort_start_date'] = pd.to_datetime(df_pop['cohort_start_date'])
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction]

all_visits = pd.read_csv(data_path+'temporal_visits.csv')
df_pop = df_pop.merge(all_visits.groupby('person_id').min()['visit_start_date'], how='left', left_on='person_id',right_index=True)
df_pop.rename({'visit_start_date':'first_visit'}, axis=1, inplace=True)
df_pop.head()

# limit to only visits pre-censor
all_visits = all_visits.loc[all_visits['person_id'].isin(df_pop['person_id'])]
all_visits['cohort_start_date'] = pd.to_datetime(all_visits['cohort_start_date'])
all_visits['visit_start_date'] = pd.to_datetime(all_visits['visit_start_date'])
all_visits['visit_end_date'] = pd.to_datetime(all_visits['visit_end_date'])
all_visits = all_visits.loc[(all_visits['cohort_start_date']-all_visits['visit_end_date']).dt.days >= num_days_prediction]
all_visits['days_to_cohort_start'] = (all_visits['cohort_start_date']-all_visits['visit_start_date']).dt.days

# get mental health-related visits
query_scz = ("SELECT sz.*, co.visit_occurrence_id, co.condition_concept_id "+
                  "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                  "LEFT JOIN cdm_mdcd.dbo.condition_occurrence as co on co.person_id = sz.person_id "+
                  "WHERE condition_concept_id IN (SELECT DISTINCT concept_id_2 FROM dbo.concept as c LEFT JOIN dbo.concept_relationship on concept_id_1 = concept_id WHERE c.concept_code LIKE 'F%' AND c.vocabulary_id = 'ICD10CM' AND relationship_id = 'Maps to')")

psych_visits_scz = pd.io.sql.read_sql(query_scz, conn)
psych_visits_scz = psych_visits_scz.loc[psych_visits_scz['person_id'].isin(df_pop['person_id'])]

query_noscz = ("SELECT pc.*, co.visit_occurrence_id, co.condition_concept_id "+
                  "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                  "LEFT JOIN cdm_mdcd.dbo.condition_occurrence as co on co.person_id = pc.person_id "+
                  "WHERE condition_concept_id IN (SELECT DISTINCT concept_id_2 FROM dbo.concept as c LEFT JOIN dbo.concept_relationship on concept_id_1 = concept_id WHERE c.concept_code LIKE 'F%' AND c.vocabulary_id = 'ICD10CM' AND relationship_id = 'Maps to')")

psych_visits_noscz = pd.io.sql.read_sql(query_noscz, conn)
psych_visits_noscz = psych_visits_noscz.loc[psych_visits_noscz['person_id'].isin(df_pop['person_id'])]

merge_visits = all_visits[['person_id', 'visit_occurrence_id', 'visit_concept_id', 'visit_start_date', 'days_to_cohort_start']].drop_duplicates()
psychiatric_visits = psych_visits_scz.merge(merge_visits, how='inner', left_on=['person_id', 'visit_occurrence_id'], right_on=['person_id', 'visit_occurrence_id'])
psychiatric_visits['psychiatric'] = 1

psychiatric_visits_noscz = psych_visits_noscz.merge(merge_visits, how='inner', left_on=['person_id', 'visit_occurrence_id'], right_on=['person_id', 'visit_occurrence_id'])
psychiatric_visits_noscz['psychiatric'] = 1

psychiatric_visits = pd.concat([psychiatric_visits, psychiatric_visits_noscz])
psychiatric_visits.drop(['condition_concept_id', 'psychosis_dx_date', 'cohort_definition_id', 'cohort_start_date'], axis=1, inplace=True)
psychiatric_visits = psychiatric_visits.drop_duplicates()

nonpsychiatric_visits = merge_visits.loc[~merge_visits['visit_occurrence_id'].isin(psychiatric_visits['visit_occurrence_id'])]

psychiatric_visits = psychiatric_visits.merge(df_pop[['person_id', 'psychosis_diagnosis_date', 'cohort_start_date', 'first_visit']], how = 'inner', left_on = 'person_id', right_on = 'person_id')
psychiatric_visits = psychiatric_visits[['person_id', 'cohort_start_date','visit_concept_id', 'visit_start_date', 'psychiatric', 'first_visit']].drop_duplicates()
psychiatric_visits['visit_concept_id'] *= 2 # so that we can treat these as "different" from nonpsychiatric

nonpsychiatric_visits = nonpsychiatric_visits.merge(df_pop[['person_id', 'psychosis_diagnosis_date','cohort_start_date', 'first_visit']], how = 'inner', left_on = 'person_id', right_on = 'person_id')
nonpsychiatric_visits['psychiatric'] = 0
nonpsychiatric_visits = nonpsychiatric_visits[['person_id', 'cohort_start_date','visit_concept_id', 'visit_start_date', 'psychiatric', 'first_visit']].drop_duplicates()

all_visits = pd.concat([psychiatric_visits, nonpsychiatric_visits]).drop_duplicates()

# calculate healthcare utilization
### Score 1: Mental Health Visits only (possible to have up to 1 per day)
mh_visits = all_visits.loc[all_visits['psychiatric']==1]
mh_hcu = pd.DataFrame(mh_visits.groupby(['person_id', 'visit_concept_id']).count()['visit_start_date'])
mh_hcu = pd.pivot_table(mh_hcu, index='person_id', columns = 'visit_concept_id', values = 'visit_start_date').fillna(0)
mh_cols = ['MH_Inpatient', 'MH_Outpatient', 'MH_ED', 'MH_Nonhospitalization']
mh_hcu.columns = mh_cols

mh_visits['first_visit'] = pd.to_datetime(mh_visits['first_visit'])
mh_visits['years_obs'] = (mh_visits['cohort_start_date']-mh_visits['first_visit']).dt.days/365
mh_hcu = mh_hcu.merge(mh_visits[['person_id', 'years_obs']], how='left', left_index=True, right_on='person_id')
mh_hcu[mh_cols] = mh_hcu[mh_cols].div(mh_hcu['years_obs'], axis=0)
mh_hcu.drop_duplicates(inplace=True)

# Score 2: Non-Mental Health Visits only (possible to have up to 1 per day)
nonmh_visits = all_visits.loc[all_visits['psychiatric']==0]
nonmh_hcu = pd.DataFrame(nonmh_visits.groupby(['person_id', 'visit_concept_id']).count()['visit_start_date'])
nonmh_hcu = pd.pivot_table(nonmh_hcu, index='person_id', columns = 'visit_concept_id', values = 'visit_start_date').fillna(0)
nonmh_cols = ['NonMH_Inpatient', 'NonMH_Outpatient', 'NonMH_ED', 'NonMH_Nonhospitalization']
nonmh_hcu.columns = nonmh_cols

nonmh_visits['first_visit'] = pd.to_datetime(nonmh_visits['first_visit'])
nonmh_visits['years_obs'] = (nonmh_visits['cohort_start_date']-nonmh_visits['first_visit']).dt.days/365
nonmh_hcu = nonmh_hcu.merge(nonmh_visits[['person_id', 'years_obs']], how='left', left_index=True, right_on='person_id')
nonmh_hcu[nonmh_cols] = nonmh_hcu[nonmh_cols].div(nonmh_hcu['years_obs'], axis=0)
nonmh_hcu.drop_duplicates(inplace=True)

# Score 3: All Health Visits (possible to have up to 2 per day)
all_visits['real_id'] = all_visits['visit_concept_id']
all_visits.loc[all_visits['psychiatric']==1, 'real_id'] /=2

all_hcu = pd.DataFrame(all_visits.groupby(['person_id', 'real_id']).count()['visit_start_date'])
all_hcu = pd.pivot_table(all_hcu, index='person_id', columns = 'real_id', values = 'visit_start_date').fillna(0)
all_cols = ['All_Inpatient', 'All_Outpatient', 'All_ED', 'All_Nonhospitalization']
all_hcu.columns = all_cols

all_visits['first_visit'] = pd.to_datetime(all_visits['first_visit'])
all_visits['years_obs'] = (all_visits['cohort_start_date']-all_visits['first_visit']).dt.days/365
all_hcu = all_hcu.merge(all_visits[['person_id', 'years_obs']], how='left', left_index=True, right_on='person_id')
all_hcu[all_cols] = all_hcu[all_cols].div(all_hcu['years_obs'], axis=0)
all_hcu.drop_duplicates(inplace=True)

# create scores
# get the feature columns
list_cols = mh_cols + nonmh_cols + all_cols

df_hcu = all_hcu[all_cols+['person_id']].merge(mh_hcu[mh_cols+['person_id']], how='outer', left_on = 'person_id', right_on='person_id')
df_hcu = df_hcu.merge(nonmh_hcu[nonmh_cols+['person_id']], how='outer', left_on='person_id', right_on='person_id')
df_hcu.fillna(0, inplace=True)

# replace columns in df_results with percentiles 
for ind in tqdm(range(len(list_cols))):
    col = list_cols[ind]
    df_hcu[col] = stats.percentileofscore(df_hcu[col], df_hcu[col], kind='weak')

test_labels = pd.read_csv('stored_data/5_21_model_test_output.csv')
save_cols = load('stored_data/df_all_iters_columns_8_visits_5_21')
save_cols.remove('iteration')

test_labels['type_prediction'] = 'TN'
test_labels.loc[(test_labels['sz_flag']==0)&(test_labels['y_pred']==1), 'type_prediction'] = 'FP'
test_labels.loc[(test_labels['sz_flag']==1)&(test_labels['y_pred']==1), 'type_prediction'] = 'TP'
test_labels.loc[(test_labels['sz_flag']==1)&(test_labels['y_pred']==0), 'type_prediction'] = 'FN'



labels_hcu = test_labels[['person_id', 'iteration','sz_flag', 'type_prediction']]
labels_hcu = labels_hcu.merge(df_hcu[['person_id']+list_cols], how='inner', left_on='person_id', right_on='person_id')

prediction_hcu_results = pd.DataFrame(index = list_cols)

tp_hcu = labels_hcu.loc[labels_hcu['type_prediction']=='TP'][list_cols]
low_ci, high_ci = stats.norm.interval(0.95, loc=tp_hcu.mean(), scale=tp_hcu.std()/np.sqrt(len(tp_hcu)))
prediction_hcu_results['TP Mean Percentile'] = np.asarray(tp_hcu.mean())
prediction_hcu_results['TP Error'] = np.asarray(tp_hcu.mean())-low_ci

fn_hcu = labels_hcu.loc[labels_hcu['type_prediction']=='FN'][list_cols]
low_ci, high_ci = stats.norm.interval(0.95, loc=fn_hcu.mean(), scale=fn_hcu.std()/np.sqrt(len(fn_hcu)))
prediction_hcu_results['FN Mean Percentile'] = np.asarray(fn_hcu.mean())
prediction_hcu_results['FN Error'] = np.asarray(fn_hcu.mean())-low_ci

tn_hcu = labels_hcu.loc[labels_hcu['type_prediction']=='TN'][list_cols]
low_ci, high_ci = stats.norm.interval(0.95, loc=tn_hcu.mean(), scale=tn_hcu.std()/np.sqrt(len(tn_hcu)))
prediction_hcu_results['TN Mean Percentile'] = np.asarray(tn_hcu.mean())
prediction_hcu_results['TN Error'] = np.asarray(tn_hcu.mean())-low_ci

fp_hcu = labels_hcu.loc[labels_hcu['type_prediction']=='FP'][list_cols]
low_ci, high_ci = stats.norm.interval(0.95, loc=fp_hcu.mean(), scale=fp_hcu.std()/np.sqrt(len(fp_hcu)))
prediction_hcu_results['FP Mean Percentile'] = np.asarray(fp_hcu.mean())
prediction_hcu_results['FP Error'] = np.asarray(fp_hcu.mean())-low_ci

inpatient_results = prediction_hcu_results.loc[['All_Inpatient', 'MH_Inpatient', 'NonMH_Inpatient']].reset_index()
outpatient_results = prediction_hcu_results.loc[['All_Outpatient', 'MH_Outpatient', 'NonMH_Outpatient']].reset_index()
ed_results = prediction_hcu_results.loc[['All_ED', 'MH_ED', 'NonMH_ED']].reset_index()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(45,12))
font = {'size':28}
matplotlib.rc('font', **font)

#PLOT 1: Inpatient
matplotlib.rc('font', **font)
ax = axes[0]
ax = inpatient_results.plot.bar(x='index', y=['TP Mean Percentile', 'FN Mean Percentile', 'FP Mean Percentile', 'TN Mean Percentile'],
                yerr=inpatient_results[['TP Error', 'FN Error', 'FP Error', 'TN Error']].T.values, color=['red', 'orange','blue', 'purple'], alpha=0.7, rot=45, ax=axes[0])
ax.legend(['TP','FN', 'FP', 'TN'], bbox_to_anchor=[1, 1])
ax.set_title('Inpatient HCU across different prediction types')
ax.set_ylabel('Average HCU Percentile')
ax.set_xlabel('Inpatient HCU Type')
ax.set_ylim([0,100])
ax.set_xticklabels(['All', 'Mental Health', 'Non-Mental Health'], rotation=0, ha='center')

#PLOT 2: Outpatient
matplotlib.rc('font', **font)
ax = axes[1]
ax = outpatient_results.plot.bar(x='index', y=['TP Mean Percentile', 'FN Mean Percentile', 'FP Mean Percentile', 'TN Mean Percentile'],
                yerr=outpatient_results[['TP Error', 'FN Error', 'FP Error', 'TN Error']].T.values, color=['red', 'orange','blue', 'purple'], alpha=0.7, rot=45, ax=axes[1])
ax.legend(['TP','FN', 'FP', 'TN'], bbox_to_anchor=[1, 1])
ax.set_title('Outpatient HCU across different prediction types')
ax.set_ylabel('Average HCU Percentile')
ax.set_xlabel('Outpatient HCU Type')
ax.set_ylim([0,100])
ax.set_xticklabels(['All', 'Mental Health', 'Non-Mental Health'], rotation=0, ha='center')


#PLOT 3: ED
matplotlib.rc('font', **font)
ax = axes[2]
ax = ed_results.plot.bar(x='index', y=['TP Mean Percentile', 'FN Mean Percentile', 'FP Mean Percentile', 'TN Mean Percentile'],
                yerr=ed_results[['TP Error', 'FN Error', 'FP Error', 'TN Error']].T.values, color=['red', 'orange','blue', 'purple'], alpha=0.7, rot=45, ax=axes[2])
ax.legend(['TP','FN', 'FP', 'TN'], bbox_to_anchor=[1, 1])
ax.set_title('ED HCU across different prediction types')
ax.set_ylabel('Average HCU Percentile')
ax.set_xlabel('ED HCU Type')
ax.set_xticklabels(['All', 'Mental Health', 'Non-Mental Health'], rotation=0, ha='center')
ax.set_ylim([0,100])

plt.tight_layout()
plt.savefig('results/prediction_type_hcu_5_21.pdf', dpi=300)