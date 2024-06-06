import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import sys
from tqdm import tqdm
from joblib import dump, load
import pickle
from sklearn.metrics import *
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../utils')
from train_utils import *
from eval_utils import *

data_path = '../prediction_data/'

# read in population dataframe
num_days_prediction = 90
df_pop = pd.read_csv(data_path+"population.csv")
df_pop['psychosis_diagnosis_date'] = pd.to_datetime(df_pop['psychosis_diagnosis_date'], format="%Y-%m-%d")
df_pop['cohort_start_date'] = pd.to_datetime(df_pop['cohort_start_date'])
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction]

# finish table 1
all_race = df_pop.groupby('race_concept_id').count()['cohort_definition_id']
scz_race = df_pop.loc[df_pop['sz_flag']==1].groupby('race_concept_id').count()['cohort_definition_id']
noscz_race = df_pop.loc[df_pop['sz_flag']==0].groupby('race_concept_id').count()['cohort_definition_id']
race_counts = pd.DataFrame(pd.concat([all_race, scz_race, noscz_race], axis=1).values, 
             index=['Missing', 'Black or African American', 'White'], columns = ['All Patients', 'SCZ Patients', 'No SCZ Patients'])

all_gender = df_pop.groupby('gender_concept_id').count()['cohort_definition_id']
scz_gender = df_pop.loc[df_pop['sz_flag']==1].groupby('gender_concept_id').count()['cohort_definition_id']
noscz_gender = df_pop.loc[df_pop['sz_flag']==0].groupby('gender_concept_id').count()['cohort_definition_id']
gender_counts = pd.DataFrame(pd.concat([all_gender, scz_gender, noscz_gender], axis=1).values, 
             index=['Male', 'Female'], columns = ['All Patients', 'SCZ Patients', 'No SCZ Patients'])

age = pd.DataFrame(df_pop.groupby('sz_flag')['age_diagnosis'].agg(['mean','std']).values, index=['SCZ Patients', 'No SCZ Patients'],
            columns = ['Mean Age', 'STD Age']).T
age['All Patients'] = df_pop['age_diagnosis'].mean(), df_pop['age_diagnosis'].std()

t1_counts = pd.concat([race_counts, gender_counts, age])
t1_counts.loc['Total Patients'] = len(df_pop), sum(df_pop['sz_flag']), len(df_pop)-sum(df_pop['sz_flag'])
t1_counts

t1_percents = t1_counts.loc[['Missing', 'Black or African American', 'White', 'Male','Female']]
t1_percents = t1_percents/t1_counts.loc['Total Patients']*100
t1_percents

# table 1: years of observation prior to psychosis
all_visits = pd.read_csv(data_path+'temporal_visits.csv')
df_pop = df_pop.merge(all_visits.groupby('person_id').min()['visit_start_date'], how='left', left_on='person_id',right_index=True)
df_pop.rename({'visit_start_date':'first_visit'}, axis=1, inplace=True)

df_pop['first_visit'] = pd.to_datetime(df_pop['first_visit'])
df_pop['years_obs_pre_psychosis'] = (df_pop['psychosis_diagnosis_date']-df_pop['first_visit']).dt.days/365

# table 1: years of observation between psychosis and index (end of obs) 
df_pop['years_obs_post_psychosis'] = (df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days/365

# table 1: number of visits in the dataset (between psychosis and censor)
all_visits = all_visits.loc[all_visits['person_id'].isin(df_pop['person_id'])]
all_visits['cohort_start_date'] = pd.to_datetime(all_visits['cohort_start_date'])
all_visits['visit_start_date'] = pd.to_datetime(all_visits['visit_start_date'])
all_visits['visit_end_date'] = pd.to_datetime(all_visits['visit_end_date'])
all_visits = all_visits.loc[(all_visits['cohort_start_date']-all_visits['visit_end_date']).dt.days >= num_days_prediction]
all_visits = all_visits.loc[all_visits['visit_start_date'] >= all_visits['psychosis_diagnosis_date']]
all_visits = all_visits[['person_id', 'cohort_start_date', 'visit_start_date']].drop_duplicates()

num_visits = all_visits.groupby('person_id').count()['cohort_start_date']
num_visits.name = 'number_of_visits'
df_pop = df_pop.merge(pd.DataFrame(num_visits), how = 'inner', left_on = 'person_id', right_index=True)
df_pop['number_of_visits'].mean()

# Fisher Exact Tests to compare proportion of genders, races
# table looks like [[scz_demo, scz_non-demo], [non-scz_demo, non-scz_non-demo]]
demos = ['race_concept_id', 'race_concept_id', 'race_concept_id', 'gender_concept_id', 'gender_concept_id']
c_ids = [8516, 8527, 0, 8532, 8507]
for demo, c_id in zip(demos, c_ids):
    scz_demo = len(df_pop.loc[(df_pop['sz_flag']==1)&(df_pop[demo]==c_id)])
    scz_nodemo = len(df_pop.loc[(df_pop['sz_flag']==1)&(df_pop[demo]!=c_id)])
    noscz_demo = len(df_pop.loc[(df_pop['sz_flag']==0)&(df_pop[demo]==c_id)])
    noscz_nodemo = len(df_pop.loc[(df_pop['sz_flag']==0)&(df_pop[demo]!=c_id)])
    arr = np.asarray([[scz_demo, scz_nodemo], [noscz_demo, noscz_nodemo]])
    print(c_id, stats.fisher_exact(arr).pvalue*5)

# t-tests to compare continuous variables
for feature in ['age_diagnosis', 'years_obs_pre_psychosis', 'number_of_visits', 'years_obs_post_psychosis']:
    print(feature)
    scz_subset = df_pop.loc[df_pop['sz_flag']==1, feature]
    noscz_subset = df_pop.loc[df_pop['sz_flag']==0, feature]
    print(stats.ttest_ind(scz_subset, noscz_subset).pvalue*5)