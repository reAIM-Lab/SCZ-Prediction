#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import pandas as pd
import pyodbc
import time
import scipy.stats as stats
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from collections import Counter
from datetime import datetime
import sys
import gc
from scipy.sparse import *
import pyarrow as pa
import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
import pickle 


# In[2]:


connection_string = (
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=OMOP.DBMI.COLUMBIA.EDU;'
    'DATABASE=cdm_mdcd;'
    'TRUSTED_CONNECTION=YES;')

conn = pyodbc.connect(connection_string)


# # Create Cohort
# 1. (In DataGrip) Get all the patients with SCZ/schizoaffective disorder and 7 years of prior observation and save them to the ak4885_schizophrenia_incidence table
# 2. (In DataGrip) Get all patients who have an episode of psychosis (incl. schizophrenia) and 7 years of observation overall -- save this into results as ak4885_psychosis_cohort
# 3. Find all people who are between 10 and 35 years at "cohort start" (SCZ diagnosis or observation period end date)
# 4. Eliminate people with SCZ diagnoses from the nosz_conds df AND make sure that the instance of SCZ is not Schizophreniform disorder (444434, 4184004, 4263364) for the SCZ population
# 5. Eliminate all people who's first episode of psychosis is schizophrenia/schizoaffective disorder is their schizophrenia diagnosis
# 6. Get all conditions in 7 years prior to cohort start for both of the above tables
# 7. Combine dataframes (SCZ and No SCZ) and add SCZ "flag"

# In[3]:


all_psychosis_codes_query = ("SELECT c_rel.concept_id as standard_concept_id, c_icd10.concept_code as icd_code, c_rel.concept_name as standard_concept_name, c_icd10.concept_name as icd_concept_name FROM dbo.concept as c_icd10 LEFT JOIN dbo.concept_relationship as rel on rel.concept_id_1 = c_icd10.concept_id "+
                       "LEFT JOIN dbo.concept as c_rel on rel.concept_id_2 = c_rel.concept_id "+
                         "WHERE (rel.relationship_id = 'Maps to' AND c_rel.standard_concept = 'S') AND (((c_icd10.concept_code IN ('295', '297', '298', '260.0', '260.1', '296.2', '296.5', '296.6', '296.24', '296.34', '291.3', '291.5', '292.1') OR c_icd10.concept_code LIKE '29[578]%') AND c_icd10.vocabulary_id = 'ICD9CM') "+
                         "OR ((c_icd10.concept_code LIKE 'F2[023456789]%' OR c_icd10.concept_code LIKE 'F30.[1234]' OR c_icd10.concept_code LIKE 'F31.[01234567]%' OR c_icd10.concept_code IN ('F32.3', 'F33.3', 'F53.1') OR c_icd10.concept_code LIKE 'F1_.15' OR c_icd10.concept_code LIKE 'F__.25' OR c_icd10.concept_code LIKE 'F__.95') AND c_icd10.vocabulary_id = 'ICD10CM'))")


psychosis_codes = pd.io.sql.read_sql(all_psychosis_codes_query, conn)
psychosis_codes.to_csv('psychosis_prediction/all_psychosis_codes.csv')


# In[4]:


all_scz_codes_query = ("SELECT c_new.concept_id as standard_concept_id, c_icd10.concept_code as icd_code, c_new.concept_name as standard_concept_name, c_icd10.concept_name as icd_name FROM dbo.concept as c_icd10 LEFT JOIN dbo.concept_relationship as rel on rel.concept_id_1 = c_icd10.concept_id "+
                "LEFT JOIN dbo.concept as c_rel on rel.concept_id_2 = c_rel.concept_id "+
                "LEFT JOIN dbo.concept_ancestor as ca ON ca.ancestor_concept_id = rel.concept_id_2 "+
                "LEFT JOIN dbo.concept as c_new on c_new.concept_id = ca.descendant_concept_id " +
                "WHERE (rel.relationship_id = 'Maps to' AND c_new.standard_concept = 'S') "+
                "AND ((c_icd10.concept_code LIKE '295%' AND c_icd10.vocabulary_id = 'ICD9CM') "+
                "OR ((c_icd10.concept_code LIKE 'F2[05]%' AND c_icd10.vocabulary_id = 'ICD10CM')))")

all_scz_codes = pd.io.sql.read_sql(all_scz_codes_query, conn)

for i in all_scz_codes['icd_code']:
    if i not in list(psychosis_codes['icd_code']):
        print(i)
all_scz_codes.to_csv('psychosis_prediction/all_scz_codes.csv')
all_scz_codes.loc[all_scz_codes['standard_concept_id'].isin([444434, 4184004, 4263364])]['standard_concept_name'].unique()


# ### Everyone from original dataset
# - SCZ: at least one schizophrenia code and 7 years prior observation (non-continuous)
# - Psychosis: at least one psychosis code and 7 years observation total (non-continuous); remove people also in SCZ cohort

# In[5]:


df_psychosis_all = pd.io.sql.read_sql("SELECT pc.*, year_of_birth, race_concept_id, gender_concept_id FROM results.ak4885_psychosis_cohort as pc LEFT JOIN dbo.person as p ON p.person_id = pc.person_id", conn)
df_scz_all = pd.io.sql.read_sql("SELECT sc.*, year_of_birth, race_concept_id, gender_concept_id FROM results.ak4885_schizophrenia_cohort as sc LEFT JOIN dbo.person as p ON p.person_id = sc.person_id", conn)
df_scz_all = df_scz_all.merge(df_psychosis_all[['person_id', 'psychosis_dx_date']], how='left', left_on = 'person_id', right_on = 'person_id')
if df_scz_all.isna().sum().sum() > 0:
    print('Undefined psychosis diagnosis date after merge')
df_psychosis_all = df_psychosis_all.loc[~df_psychosis_all['person_id'].isin(list(df_scz_all['person_id']))]
print(len(df_psychosis_all), len(df_scz_all), len(df_scz_all)*100/(len(df_scz_all)+len(df_psychosis_all)))


# ### Make sure that people in the schizophrenia cohort have at least 1 dx (which is not schizophreniform disorder
# Ignore the fact that these variables are called "dx_twice_pids"

# In[6]:


# limit schizophrenia cohort to people with 1 diagnoses
all_sz_query = ("SELECT person_id, condition_concept_id, condition_start_date FROM dbo.condition_occurrence WHERE condition_concept_id IN (SELECT c_new.concept_id FROM dbo.concept as c_icd10 LEFT JOIN dbo.concept_relationship as rel on rel.concept_id_1 = c_icd10.concept_id "+
                "LEFT JOIN dbo.concept as c_rel on rel.concept_id_2 = c_rel.concept_id "+
                "LEFT JOIN dbo.concept_ancestor as ca ON ca.ancestor_concept_id = rel.concept_id_2 "+
                "LEFT JOIN dbo.concept as c_new on c_new.concept_id = ca.descendant_concept_id " +
                "WHERE (rel.relationship_id = 'Maps to' AND c_new.standard_concept = 'S') "+
                "AND ((c_icd10.concept_code LIKE '295%' AND c_icd10.vocabulary_id = 'ICD9CM') "+
                "OR ((c_icd10.concept_code LIKE 'F2[05].%' OR c_icd10.concept_code = 'F20.81' AND c_icd10.vocabulary_id = 'ICD10CM'))))")

all_sz_dx = pd.io.sql.read_sql(all_sz_query, conn)

# remove schizophreniform disorder as "acceptable SCZ diagnosis"
all_sz_dx = all_sz_dx.loc[~all_sz_dx['condition_concept_id'].isin([444434, 4184004, 4263364])]

dx_twice_pids = all_sz_dx[['person_id','condition_start_date']].drop_duplicates().groupby('person_id').count()['condition_start_date'] >= 1
df_scz_all = df_scz_all.loc[df_scz_all['person_id'].isin(list(dx_twice_pids[dx_twice_pids==True].index))]


# In[7]:


print(len(df_psychosis_all), len(df_scz_all), len(df_scz_all)*100/(len(df_scz_all)+len(df_psychosis_all)))


# ## Make sure that people in the non-schizophrenia cohort have no instances of a schizophrenia diagnosis 
# Do this by getting all conditions for people in the psychosis cohort and then removing anyone with any schizophrenia code at any point in time (they can have schizophrenifrom diagnosis). 

# In[8]:


df_psychosis_all = df_psychosis_all.loc[~(df_psychosis_all['person_id'].isin(list(all_sz_dx['person_id'].unique())))]
psychosis_conds = pd.io.sql.read_sql("SELECT DISTINCT pc.person_id, condition_concept_id, condition_start_date FROM results.ak4885_psychosis_cohort as pc LEFT JOIN dbo.condition_occurrence as co ON co.person_id = pc.person_id", conn)

scz_codes = all_scz_codes.loc[~all_scz_codes['standard_concept_id'].isin([444434, 4184004, 4263364])]['standard_concept_id']
scz_in_psychosis = psychosis_conds.loc[psychosis_conds['condition_concept_id'].isin(scz_codes)]
df_psychosis_all = df_psychosis_all.loc[~df_psychosis_all['person_id'].isin(scz_in_psychosis)]

print(len(df_psychosis_all), len(df_scz_all), len(df_scz_all)*100/(len(df_scz_all)+len(df_psychosis_all)))


# ### Make sure that people in the schizophrenia cohort have an accurate cohort_start_date

# In[9]:


scz_conds =pd.io.sql.read_sql ("SELECT DISTINCT sc.person_id, condition_start_date, condition_concept_id FROM results.ak4885_schizophrenia_cohort as sc LEFT JOIN dbo.condition_occurrence as co ON co.person_id = sc.person_id", conn) 
scz_in_scz = scz_conds.loc[scz_conds['condition_concept_id'].isin(scz_codes)]
scz_in_scz = scz_in_scz.merge(df_scz_all, how='outer', left_on='person_id', right_on='person_id')

scz_in_scz['condition_start_date'] = pd.to_datetime(scz_in_scz['condition_start_date'])
scz_in_scz['cohort_start_date'] = pd.to_datetime(scz_in_scz['cohort_start_date'])


min_scz_start = scz_in_scz.groupby('person_id')['condition_start_date'].min()
min_scz_start.name = 'min_scz_start'
scz_in_scz = scz_in_scz.merge(min_scz_start, how='left', left_on='person_id', right_index=True)
scz_in_scz.loc[scz_in_scz['min_scz_start']<scz_in_scz['cohort_start_date'], 'cohort_start_date'] = scz_in_scz.loc[scz_in_scz['min_scz_start']<scz_in_scz['cohort_start_date'], 'min_scz_start']

df_scz_all.drop(['cohort_start_date'], axis=1, inplace=True)
df_scz_all = df_scz_all.merge(scz_in_scz[['person_id', 'cohort_start_date']], how='left', left_on = 'person_id', right_on = 'person_id')


# In[10]:


df_scz_all.drop_duplicates(inplace=True)
print(len(df_psychosis_all), len(df_scz_all), len(df_scz_all)*100/(len(df_scz_all)+len(df_psychosis_all)))
df_scz_all.isna().sum()


# ### Make sure that everyone has an accurate psychosis_dx_date

# In[11]:


psych_in_scz = scz_conds.loc[scz_conds['condition_concept_id'].isin(psychosis_codes['standard_concept_id'])]
psych_in_scz = psych_in_scz.merge(df_scz_all, how='outer', left_on='person_id', right_on='person_id')
psych_in_scz['psychosis_dx_date'] = pd.to_datetime(psych_in_scz['psychosis_dx_date'], format='mixed')
psych_in_scz['condition_start_date'] = pd.to_datetime(psych_in_scz['condition_start_date'])

min_psych_start = psych_in_scz.groupby('person_id')['condition_start_date'].min()
min_psych_start.name = 'min_psych_start'
psych_in_scz = psych_in_scz.merge(min_psych_start, how='left', left_on='person_id', right_index=True)
psych_in_scz.loc[psych_in_scz['min_psych_start']<psych_in_scz['psychosis_dx_date'], 'psychosis_dx_date'] = psych_in_scz.loc[psych_in_scz['min_psych_start']<psych_in_scz['psychosis_dx_date'], 'min_psych_start']
print(len(psych_in_scz.loc[psych_in_scz['condition_start_date']<psych_in_scz['psychosis_dx_date']]))

df_scz_all.drop(['psychosis_dx_date'], axis=1, inplace=True)
df_scz_all = df_scz_all.merge(psych_in_scz[['person_id', 'psychosis_dx_date']].drop_duplicates(), how='left', left_on = 'person_id', right_on = 'person_id')


# In[16]:


psych_in_psych = psychosis_conds.loc[psychosis_conds['condition_concept_id'].isin(psychosis_codes['standard_concept_id'])]
psych_in_psych = psych_in_psych.merge(df_psychosis_all, how='outer', left_on='person_id', right_on='person_id')
psych_in_psych['psychosis_dx_date'] = pd.to_datetime(psych_in_psych['psychosis_dx_date'], format='mixed')
psych_in_psych['condition_start_date'] = pd.to_datetime(psych_in_psych['condition_start_date'])

min_psych_psych_start = psych_in_psych.groupby('person_id')['condition_start_date'].min()
min_psych_psych_start.name = 'min_psych_psych_start'
psych_in_psych = psych_in_psych.merge(min_psych_psych_start, how='left', left_on='person_id', right_index=True)
psych_in_psych.loc[psych_in_psych['min_psych_psych_start']<psych_in_psych['psychosis_dx_date'], 'psychosis_dx_date'] = psych_in_psych.loc[psych_in_psych['min_psych_psych_start']<psych_in_psych['psychosis_dx_date'], 'min_psych_psych_start']
print(len(psych_in_psych.loc[psych_in_psych['condition_start_date']<psych_in_psych['psychosis_dx_date']]))

df_psychosis_all.drop(['psychosis_dx_date'], axis=1, inplace=True)
df_psychosis_all = df_psychosis_all.merge(psych_in_psych[['person_id', 'psychosis_dx_date']].drop_duplicates(), how='left', left_on = 'person_id', right_on = 'person_id')


# ### Ages 10-35 at "cohort start date" 
# (end of observation for psychosis patients, first SCZ diagnosis for SCZ patients)

# In[17]:


df_psychosis_all['end_date'] = pd.to_datetime(df_psychosis_all['end_date'], format = '%Y-%m-%d')
df_psychosis_all['year_of_birth'] = pd.to_datetime(df_psychosis_all['year_of_birth'], format = '%Y')
df_psychosis_all['age_diagnosis'] = (df_psychosis_all['end_date']-df_psychosis_all['year_of_birth']).dt.days/365

df_psychosis_all = df_psychosis_all.loc[df_psychosis_all['age_diagnosis']<=35]
df_psychosis_all = df_psychosis_all.loc[df_psychosis_all['age_diagnosis']>=10]

df_scz_all['cohort_start_date'] = pd.to_datetime(df_scz_all['cohort_start_date'], format = '%Y-%m-%d')
df_scz_all['year_of_birth'] = pd.to_datetime(df_scz_all['year_of_birth'], format = '%Y')
df_scz_all['age_diagnosis'] = (df_scz_all['cohort_start_date']-df_scz_all['year_of_birth']).dt.days/365

df_scz_all = df_scz_all.loc[df_scz_all['age_diagnosis']<=35]
df_scz_all = df_scz_all.loc[df_scz_all['age_diagnosis']>=10]

print(len(df_psychosis_all), len(df_scz_all), len(df_scz_all)*100/(len(df_scz_all)+len(df_psychosis_all)))


# ### Now limit to people whos first diagnosis of SCZ is AFTER their first episode of psychosis
# Restrict to people for whom the cohort start date (schizophrenia diagnosis date) is AFTER the first date of psychosis

# In[18]:


df_scz_all['psychosis_dx_date'] = pd.to_datetime(df_scz_all['psychosis_dx_date'])
df_scz_all = df_scz_all.loc[df_scz_all['cohort_start_date']>df_scz_all['psychosis_dx_date']]
print(len(df_psychosis_all), len(df_scz_all), len(df_scz_all)*100/(len(df_scz_all)+len(df_psychosis_all)))


# ### Loading in temporal conditions data

# In[19]:


sz_conds_query = ("SELECT sz.*, co.condition_start_date, co.condition_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                  "LEFT JOIN cdm_mdcd.dbo.condition_occurrence as co on co.person_id = sz.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = co.condition_concept_id "+
                  "WHERE condition_concept_id > 0")
sz_conds = pd.io.sql.read_sql(sz_conds_query, conn)


# In[20]:


nosz_conds_query = ("SELECT pc.*, co.condition_start_date, co.condition_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                  "LEFT JOIN cdm_mdcd.dbo.condition_occurrence as co on co.person_id = pc.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = co.condition_concept_id "+
                  "WHERE condition_concept_id > 0")

list_chunks = []
for chunk in pd.io.sql.read_sql(nosz_conds_query, conn, chunksize=500000):
    list_chunks.append(chunk.loc[chunk['person_id'].isin(df_psychosis_all['person_id'])])
nosz_conds = pd.concat(list_chunks)


# In[21]:


del list_chunks
gc.collect()


# In[22]:


print(len(nosz_conds))
nosz_conds = nosz_conds.loc[nosz_conds['person_id'].isin(list(df_psychosis_all['person_id']))]
print(len(nosz_conds))

print(len(sz_conds))
sz_conds = sz_conds.loc[sz_conds['person_id'].isin(list(df_scz_all['person_id']))]
print(len(sz_conds))


# In[23]:


nosz_conds['cohort_start_date'] = nosz_conds['end_date']
sz_conds = sz_conds.merge(df_scz_all[['person_id', 'psychosis_dx_date']], how='left', left_on = 'person_id', right_on = 'person_id')


# In[24]:


sz_conds['sz_flag'] = 1
nosz_conds['sz_flag'] = 0
all_conds = pd.concat([sz_conds, nosz_conds])
print(len(all_conds))
print(all_conds.columns)


# In[25]:


del sz_conds
del nosz_conds
gc.collect()


# In[26]:


df_psychosis_all['sz_flag'] = 0
df_scz_all['sz_flag'] = 1

df_psychosis_all['cohort_start_date'] = df_psychosis_all['end_date']

df_pop = pd.concat([df_psychosis_all, df_scz_all])
print(len(df_pop), sum(df_pop['sz_flag'])*100/len(df_pop))


# In[27]:


all_conds.isna().sum().sum(), df_pop.isna().sum().sum()


# # Constrict cohort based on continuous care
# ### First drop instances of conditions where the condition concept id is not defined
# 
# 
# ### To ensure that there is at least 1 service contact per year
# 
# Calculate the differences between consecutive condition occurrences for each patient -- do this by making sure that:
# 1. there are at least 7 unique dates that there is a visit 
# 2. there's at least one visit > 6 years before diagnosis
# 3. the max difference between consecutive dates is 1 year (inclusive)

# In[28]:


# drop undefined conditions
all_conds['condition_start_date'] = pd.to_datetime(all_conds['condition_start_date'], format = '%Y-%m-%d')
all_conds['cohort_start_date'] = pd.to_datetime(all_conds['cohort_start_date'], format = '%Y-%m-%d')

conds_dates = all_conds[['person_id', 'condition_start_date', 'cohort_start_date']].drop_duplicates()
print('done datetime conversions')

# at least 7 unique dates for visits
conds_patients = conds_dates.groupby('person_id').count()
yearly_service_pids = list(conds_patients.loc[conds_patients['condition_start_date'] >= 7].index)
print('done getting at least 7 unique dates for visits')

# at least 1 visit > 6 years before diagnosis
conds_dates = conds_dates.loc[conds_dates['perso=n_id'].isin(yearly_service_pids)]
yearly_service_pids = list(conds_dates.loc[(conds_dates['cohort_start_date']-conds_dates['condition_start_date']).dt.days > 2190]['person_id'].unique())
print('done w/ earliest visit prior to 6 years pre-cohort start')

# maximum of 1 year between conditions
conds_dates = conds_dates.loc[conds_dates['person_id'].isin(yearly_service_pids)]
conds_dates_grouped = conds_dates.groupby(['person_id'])['condition_start_date'].apply(np.hstack)
conds_dates_arr = conds_dates_grouped.reset_index().values
print('done grouping for 1 year between conditions')

yearly_service_pids = []
for ind in tqdm(range(0,len(conds_dates_arr))):
    if len(conds_dates_arr[ind,1])>1:
        conds_dates_arr[ind,1].sort()
        if(max(np.diff(conds_dates_arr[ind,1])).days<=365):
            yearly_service_pids.append(conds_dates_arr[ind,0])
print(len(yearly_service_pids))

pd.DataFrame(yearly_service_pids).to_csv('psychosis_prediction/yearly_service_pids.csv', index=False)


# In[29]:


df_psychosis_all = df_psychosis_all.loc[df_psychosis_all['person_id'].isin(yearly_service_pids)]
df_scz_all = df_scz_all.loc[df_scz_all['person_id'].isin(yearly_service_pids)]

print(len(df_psychosis_all), len(df_scz_all), len(df_scz_all)*100/(len(df_scz_all)+len(df_psychosis_all)))


# In[30]:


df_pop = df_pop.loc[df_pop['person_id'].isin(yearly_service_pids)]
print(len(df_pop))


# ## Maximum of 45 days with no insurance coverage
# - Get insurance information from all people with at least 7 years observation and then limit to only the people in with at least 1 service visit per year
# - Combine all of the overlapping payer periods, with a grace period of 45 days between coverage periods

# In[31]:


insurance_query = ("SELECT ppp.PERSON_ID, ppp.PAYER_PLAN_PERIOD_START_DATE, ppp.PAYER_PLAN_PERIOD_END_DATE, ppp.PAYER_SOURCE_VALUE "+
                   "FROM dbo.PAYER_PLAN_PERIOD as ppp LEFT JOIN dbo.OBSERVATION_PERIOD as op ON op.person_id = ppp.PERSON_ID "+
                   "WHERE DATEDIFF(day, OBSERVATION_PERIOD_START_DATE, OBSERVATION_PERIOD_END_DATE) > 2555")
insurance_df = pd.io.sql.read_sql(insurance_query, conn)


# In[32]:


insurance_df = insurance_df.loc[insurance_df['PERSON_ID'].isin(yearly_service_pids)]
len(insurance_df)


# In[33]:


insurance_df['PAYER_PLAN_PERIOD_START_DATE'] =  pd.to_datetime(insurance_df['PAYER_PLAN_PERIOD_START_DATE'], format='%Y-%m-%d')
insurance_df['PAYER_PLAN_PERIOD_END_DATE'] =  pd.to_datetime(insurance_df['PAYER_PLAN_PERIOD_END_DATE'], format='%Y-%m-%d')

# https://stackoverflow.com/questions/68714898/merge-consecutive-and-overlapping-date-ranges

merged_insurance_df = insurance_df.groupby(["PERSON_ID"], as_index=False).apply(
    lambda d: d.sort_values(["PAYER_PLAN_PERIOD_END_DATE", "PAYER_PLAN_PERIOD_START_DATE"])
    .assign(
        grp=lambda d: (
            ~(d["PAYER_PLAN_PERIOD_START_DATE"] <= (d["PAYER_PLAN_PERIOD_END_DATE"].shift() + pd.Timedelta(days=45)))
        ).cumsum()
    )
    .groupby(["PERSON_ID", "grp"], as_index=False)
    .agg({"PAYER_PLAN_PERIOD_START_DATE": "min", "PAYER_PLAN_PERIOD_END_DATE": "max"})
).reset_index(drop=True)


# Now that we have adjusted for the combined insurance periods with a 45-day grace period, we want to make sure that it extends from 7 years pre-diagnosis to the date of diagnosis

# In[34]:


df_pop['cohort_start_date'] =  pd.to_datetime(df_pop['cohort_start_date'], format='%Y-%m-%d')

insurance_check_df = df_pop.merge(merged_insurance_df, how = 'left', left_on = 'person_id', right_on = 'PERSON_ID')
eligible_pids = insurance_check_df.loc[(insurance_check_df['PAYER_PLAN_PERIOD_END_DATE']>=insurance_check_df['cohort_start_date'])&(insurance_check_df['PAYER_PLAN_PERIOD_START_DATE'] <= insurance_check_df['cohort_start_date']- pd.Timedelta(days=2555))]['person_id'].unique()

print(len(eligible_pids))
eligible_pids = list(eligible_pids)


# In[35]:


df_psychosis_all = df_psychosis_all.loc[df_psychosis_all['person_id'].isin(eligible_pids)]
df_scz_all = df_scz_all.loc[df_scz_all['person_id'].isin(eligible_pids)]

print(len(df_psychosis_all), len(df_scz_all), len(df_scz_all)*100/(len(df_scz_all)+len(df_psychosis_all)))


# In[36]:


df_psychosis_all['sz_flag'] = 0
df_scz_all['sz_flag'] = 1
df_pop = pd.concat([df_psychosis_all, df_scz_all])
print(len(df_pop), sum(df_pop['sz_flag'])*100/len(df_pop))


# In[37]:


df_pop.to_csv('psychosis_prediction/population.csv', index=False)
pd.DataFrame(eligible_pids).to_csv('psychosis_prediction/insurance_pids.csv', index=False)
pd.DataFrame(yearly_service_pids).to_csv('psychosis_prediction/yearly_service_pids.csv', index=False)


# According to the NIMH (https://www.nimh.nih.gov/health/statistics/schizophrenia), 0.66% is within range (0.33% to 0.75%) for prevalence 

# In[38]:


print('Total patients:',len(df_pop))
print('% Patients with Schizophrenia:',100*sum(df_pop['sz_flag'])/len(df_pop))


# ### Save all_conds

# In[39]:


all_conds = all_conds.loc[all_conds['person_id'].isin(eligible_pids)]
all_conds.drop(['cohort_start_date'], axis=1, inplace=True)
all_conds = all_conds.merge(df_pop[['person_id', 'cohort_start_date']], how='left', left_on = 'person_id', right_on = 'person_id')
all_conds.to_csv('psychosis_prediction/temporal_conditions.csv', index=False)


# In[40]:


print(len(all_conds))


# # Load in and Save other Data (Temporal)
# ## Medications

# In[42]:


df_pop = pd.read_csv('psychosis_prediction/population.csv')


# In[43]:


sz_meds_query = ("SELECT sz.*, drug_concept_id, drug_era_start_date, drug_era_end_date, drug_exposure_count, gap_days "+ 
                 "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                   "LEFT JOIN cdm_mdcd.dbo.drug_era on drug_era.person_id = sz.person_id")


sz_meds = pd.io.sql.read_sql(sz_meds_query, conn)
sz_meds.columns = sz_meds.columns.str.lower()


# In[44]:


nosz_meds_query = ("SELECT pc.*, drug_concept_id, drug_era_start_date, drug_era_end_date, drug_exposure_count, gap_days "+ 
                 "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                   "LEFT JOIN cdm_mdcd.dbo.drug_era on drug_era.person_id = pc.person_id")
list_chunks = []
for chunk in pd.io.sql.read_sql(nosz_meds_query, conn, chunksize=1000000):
    list_chunks.append(chunk.loc[chunk['person_id'].isin(df_pop['person_id'])])
nosz_meds = pd.concat(list_chunks)


# In[45]:


del list_chunks
gc.collect()


# In[46]:


sz_meds = sz_meds.merge(df_pop[['person_id', 'psychosis_dx_date']], how='left', left_on = 'person_id', right_on='person_id')
nosz_meds['cohort_start_date'] = nosz_meds['end_date']
print(set(sz_meds.columns) == set(nosz_meds.columns))


# In[47]:


all_meds = pd.concat([sz_meds, nosz_meds])

del sz_meds
del nosz_meds
gc.collect()


# In[48]:


all_meds = all_meds.loc[all_meds['person_id'].isin(list(df_pop['person_id']))]
all_meds.drop(['cohort_start_date'], axis=1, inplace=True)
all_meds = all_meds.merge(df_pop[['person_id', 'cohort_start_date']], how='left', left_on = 'person_id', right_on = 'person_id')
all_meds.dropna(inplace=True)
print(len(all_meds))
print(len(all_meds['person_id'].unique()))
all_meds.to_csv('psychosis_prediction/temporal_medications.csv')


# ## Visits

# In[49]:


sz_visits_query = ("SELECT sz.*, visit_occurrence_id, visit_concept_id, visit_start_date, visit_end_date, visit_type_concept_id " +
                   "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                   "LEFT JOIN cdm_mdcd.dbo.visit_occurrence as v on v.person_id = sz.person_id")

sz_visits = pd.io.sql.read_sql(sz_visits_query, conn)


# In[50]:


nosz_visits_query = ("SELECT pc.*, visit_occurrence_id, visit_concept_id, visit_start_date, visit_end_date, visit_type_concept_id " +
                   "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                   "LEFT JOIN cdm_mdcd.dbo.visit_occurrence as v on v.person_id = pc.person_id")

list_chunks = []
for chunk in pd.io.sql.read_sql(nosz_visits_query, conn, chunksize=1000000):
    list_chunks.append(chunk.loc[chunk['person_id'].isin(df_pop['person_id'])])
nosz_visits = pd.concat(list_chunks)


# In[51]:


del list_chunks
gc.collect()


# In[52]:


sz_visits = sz_visits.merge(df_pop[['person_id', 'psychosis_dx_date']], how='left', left_on = 'person_id', right_on='person_id')
nosz_visits['cohort_start_date'] = nosz_visits['end_date']

all_visits = pd.concat([sz_visits, nosz_visits])

del sz_visits
del nosz_visits
gc.collect()


# In[53]:


df_pop = pd.read_csv('psychosis_prediction/population.csv')
all_visits = all_visits.loc[all_visits['person_id'].isin(list(df_pop['person_id']))]
all_visits.drop(['cohort_start_date'], axis=1, inplace=True)
all_visits = all_visits.merge(df_pop[['person_id', 'cohort_start_date']], how='left', left_on = 'person_id', right_on = 'person_id')

print(len(all_visits))
print(len(all_visits['person_id'].unique()))
all_visits.to_csv('psychosis_prediction/temporal_visits.csv')
print(all_visits.isna().sum().sum())


# ## Procedures

# In[54]:


sz_procedures_query = ("SELECT sz.*, procedure_date, procedure_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                  "LEFT JOIN cdm_mdcd.dbo.procedure_occurrence as po on po.person_id = sz.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = po.procedure_concept_id")

sz_procedures = pd.io.sql.read_sql(sz_procedures_query, conn)


# In[55]:


nosz_procedures_query = ("SELECT pc.*, procedure_date, procedure_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                  "LEFT JOIN cdm_mdcd.dbo.procedure_occurrence as po on po.person_id = pc.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = po.procedure_concept_id")

list_chunks = []
for chunk in pd.io.sql.read_sql(nosz_procedures_query, conn, chunksize=1000000):
    list_chunks.append(chunk.loc[chunk['person_id'].isin(df_pop['person_id'])])
nosz_procedures = pd.concat(list_chunks)


# In[56]:


del list_chunks
gc.collect()


# In[57]:


sz_procedures = sz_procedures.merge(df_pop[['person_id', 'psychosis_dx_date']], how='left', left_on = 'person_id', right_on='person_id')
nosz_procedures['cohort_start_date'] = nosz_procedures['end_date']

all_procedures = pd.concat([sz_procedures, nosz_procedures])
all_procedures = all_procedures.loc[all_procedures['procedure_concept_id']>0]


# In[58]:


del sz_procedures
del nosz_procedures
gc.collect()


# In[59]:


df_pop = pd.read_csv('psychosis_prediction/population.csv')
all_procedures = all_procedures.loc[all_procedures['person_id'].isin(list(df_pop['person_id']))]
all_procedures.dropna(inplace=True)
all_procedures.drop(['cohort_start_date'], axis=1, inplace=True)
all_procedures = all_procedures.merge(df_pop[['person_id', 'cohort_start_date']], how='left', left_on = 'person_id', right_on = 'person_id')
print(len(all_procedures))
print(len(all_procedures['person_id'].unique()))
all_procedures.to_csv('psychosis_prediction/temporal_procedures.csv')


# ## Labs

# In[60]:


sz_measurement_query = ("SELECT sz.*, measurement_date, measurement_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                  "LEFT JOIN cdm_mdcd.dbo.measurement as m on m.person_id = sz.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = m.measurement_concept_id")

sz_labs = pd.io.sql.read_sql(sz_measurement_query, conn)


# In[61]:


nosz_measurements_query = ("SELECT pc.*, measurement_date, measurement_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                  "LEFT JOIN cdm_mdcd.dbo.measurement as m on m.person_id = pc.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = m.measurement_concept_id")
list_chunks = []
for chunk in pd.io.sql.read_sql(nosz_measurements_query, conn, chunksize=1000000):
    list_chunks.append(chunk.loc[chunk['person_id'].isin(df_pop['person_id'])])
nosz_labs = pd.concat(list_chunks)


# In[62]:


del list_chunks
gc.collect()


# In[63]:


sz_labs = sz_labs.merge(df_pop[['person_id', 'psychosis_dx_date']], how='left', left_on = 'person_id', right_on='person_id')
nosz_labs['cohort_start_date'] = nosz_labs['end_date']

all_labs = pd.concat([sz_labs, nosz_labs])
all_labs = all_labs.loc[all_labs['measurement_concept_id']>0]


# In[64]:


del sz_labs
del nosz_labs
gc.collect()


# In[65]:


df_pop = pd.read_csv('psychosis_prediction/population.csv')
all_labs = all_labs.loc[all_labs['person_id'].isin(list(df_pop['person_id']))]
all_labs.dropna(inplace=True)
all_labs.drop(['cohort_start_date'], axis=1, inplace=True)
all_labs = all_labs.merge(df_pop[['person_id', 'cohort_start_date']], how='left', left_on = 'person_id', right_on = 'person_id')
print(len(all_labs))
print(len(all_labs['person_id'].unique()))
all_labs.to_csv('psychosis_prediction/temporal_labs.csv')


# ## Conditions

# In[ ]:


sz_conds_query = ("SELECT sz.*, condition_start_date, condition_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                  "LEFT JOIN cdm_mdcd.dbo.condition_occurrence as co on co.person_id = sz.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = co.condition_concept_id "+
                  "WHERE condition_concept_id > 0")

sz_conds = pd.io.sql.read_sql(sz_conds_query, conn)


# In[ ]:


nosz_conds_query = ("SELECT pc.*, condition_start_date, condition_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                  "LEFT JOIN cdm_mdcd.dbo.condition_occurrence as co on co.person_id = pc.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = co.condition_concept_id "+
                  "WHERE condition_concept_id > 0")
list_chunks = []
for chunk in pd.io.sql.read_sql(nosz_conds_query, conn, chunksize=1000000):
    list_chunks.append(chunk)
nosz_conds = pd.concat(list_chunks)


# In[ ]:


del list_chunks
gc.collect()


# In[ ]:


sz_conds = sz_conds.merge(df_pop[['person_id', 'psychosis_dx_date']], how='left', left_on = 'person_id', right_on='person_id')
nosz_conds['cohort_start_date'] = nosz_conds['end_date']

all_conds = pd.concat([sz_conds, nosz_conds])


# In[ ]:


del sz_conds
del nosz_conds
gc.collect()


# In[45]:


df_pop = pd.read_csv('psychosis_prediction/population.csv')
all_conds = all_conds.loc[all_conds['person_id'].isin(list(df_pop['person_id']))]
all_conds.drop(['cohort_start_date'], axis=1, inplace=True)
all_conds = all_conds.merge(df_pop[['person_id', 'cohort_start_date']], how='left', left_on = 'person_id', right_on = 'person_id')

print(len(all_conds))
print(len(all_conds['person_id'].unique()))
all_conds.to_csv('psychosis_prediction/temporal_conditions.csv')
print(all_conds.isna().sum().sum())


# In[3]:


all_conds = pd.read_csv('psychosis_prediction/temporal_conditions.csv')
df_pop = pd.read_csv('psychosis_prediction/population.csv')
psychosis_codes = pd.read_csv('psychosis_prediction/all_psychosis_codes.csv')


# In[10]:





# In[11]:


psych_conds.loc[psych_conds['condition_start_date']<psych_conds['psychosis_dx_date']]


# # Create the file that sorts everything by time and counts it

# Limit to data between 7 years (2555 days) and 90 days, inclusive prior to diagnosis. This also means that the psychosis_diagnosis date should be at least **90 days** before diagnosis

# In[93]:


num_days_prediction = 90
df_pop = pd.read_csv('psychosis_prediction/population.csv')
df_pop.rename({'psychosis_dx_date':'psychosis_diagnosis_date'}, axis=1, inplace=True)
df_pop['psychosis_diagnosis_date'] = pd.to_datetime(df_pop['psychosis_diagnosis_date'], format="mixed")
df_pop['cohort_start_date'] = pd.to_datetime(df_pop['cohort_start_date'])
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction]


# In[5]:


# only need to read in if you haven't run the top part

all_visits = pd.read_csv('psychosis_prediction/temporal_visits.csv')
all_conds = pd.read_csv('psychosis_prediction/temporal_conditions.csv')
all_procedures = pd.read_csv('psychosis_prediction/temporal_procedures.csv')
all_labs = pd.read_csv('psychosis_prediction/temporal_labs.csv')
all_meds = pd.read_csv('psychosis_prediction/temporal_medications.csv')


# ### Restrict to the data (rows) for the patients/timing that I want

# In[94]:


all_meds = all_meds.loc[all_meds['person_id'].isin(df_pop['person_id'])]
all_meds['cohort_start_date'] = pd.to_datetime(all_meds['cohort_start_date'])
all_meds['drug_era_start_date'] = pd.to_datetime(all_meds['drug_era_start_date'])
all_meds['drug_era_end_date'] = pd.to_datetime(all_meds['drug_era_end_date'])
all_meds = all_meds.loc[(all_meds['cohort_start_date']-all_meds['drug_era_end_date']).dt.days >= num_days_prediction]
all_meds = all_meds.loc[(all_meds['cohort_start_date']-all_meds['drug_era_start_date']).dt.days]
all_meds['days_to_cohort_start'] = (all_meds['cohort_start_date']-all_meds['drug_era_start_date']).dt.days


# In[48]:


all_visits = all_visits.loc[all_visits['person_id'].isin(df_pop['person_id'])]
all_visits['cohort_start_date'] = pd.to_datetime(all_visits['cohort_start_date'])
all_visits['visit_start_date'] = pd.to_datetime(all_visits['visit_start_date'])
all_visits['visit_end_date'] = pd.to_datetime(all_visits['visit_end_date'])
all_visits = all_visits.loc[(all_visits['cohort_start_date']-all_visits['visit_end_date']).dt.days >= num_days_prediction]
all_visits = all_visits.loc[(all_visits['cohort_start_date']-all_visits['visit_start_date']).dt.days <= 2555]
all_visits['days_to_cohort_start'] = (all_visits['cohort_start_date']-all_visits['visit_start_date']).dt.days


# In[49]:


all_conds = all_conds.loc[all_conds['person_id'].isin(df_pop['person_id'])]
all_conds['cohort_start_date'] = pd.to_datetime(all_conds['cohort_start_date'])
all_conds['condition_start_date'] = pd.to_datetime(all_conds['condition_start_date'])
all_conds['days_to_cohort_start'] = (all_conds['cohort_start_date']-all_conds['condition_start_date']).dt.days
all_conds = all_conds.loc[all_conds['days_to_cohort_start'] >= num_days_prediction]
all_conds = all_conds.loc[all_conds['days_to_cohort_start'] <= 2555]


# In[50]:


all_procedures = all_procedures.loc[all_procedures['person_id'].isin(df_pop['person_id'])]
all_procedures['cohort_start_date'] = pd.to_datetime(all_procedures['cohort_start_date'])
all_procedures['procedure_date'] = pd.to_datetime(all_procedures['procedure_date'])
all_procedures['days_to_cohort_start'] = (all_procedures['cohort_start_date']-all_procedures['procedure_date']).dt.days
all_procedures = all_procedures.loc[all_procedures['days_to_cohort_start'] >= num_days_prediction]
all_procedures = all_procedures.loc[all_procedures['days_to_cohort_start'] <= 2555]


# In[51]:


all_labs = all_labs.loc[all_labs['person_id'].isin(df_pop['person_id'])]
all_labs['cohort_start_date'] = pd.to_datetime(all_labs['cohort_start_date'])
all_labs['measurement_date'] = pd.to_datetime(all_labs['measurement_date'])
all_labs['days_to_cohort_start'] = (all_labs['cohort_start_date']-all_labs['measurement_date']).dt.days
all_labs = all_labs.loc[all_labs['days_to_cohort_start'] >= num_days_prediction]
all_labs = all_labs.loc[all_labs['days_to_cohort_start'] <= 2555]


# ### Concatenate all these dfs 

# In[57]:


all_procedures = all_procedures[['person_id', 'days_to_cohort_start', 'procedure_concept_id']].drop_duplicates()
# change the name to make merging the dfs easier later
all_procedures.rename({'procedure_concept_id':'concept_id'}, axis=1, inplace=True)

remove_rare_conds = all_procedures[['person_id', 'concept_id']].drop_duplicates().value_counts('concept_id')
remove_rare_conds = list(remove_rare_conds[remove_rare_conds > 0.01*len(df_pop)].index)
all_procedures = all_procedures.loc[all_procedures['concept_id'].isin(remove_rare_conds)]
len(all_procedures)


# In[58]:


all_labs = all_labs[['person_id', 'days_to_cohort_start', 'measurement_concept_id']].drop_duplicates()
# change the name to make merging the dfs easier later
all_labs.rename({'measurement_concept_id':'concept_id'}, axis=1, inplace=True)

remove_rare_labs = all_labs[['person_id', 'concept_id']].drop_duplicates().value_counts('concept_id')
remove_rare_labs = list(remove_rare_labs[remove_rare_labs > 0.01*len(df_pop)].index)
all_labs = all_labs.loc[all_labs['concept_id'].isin(remove_rare_labs)]
len(all_labs)


# In[59]:


all_visits['los'] = (all_visits['visit_end_date']-all_visits['visit_start_date']).dt.days
los_df = all_visits.loc[all_visits['visit_concept_id'].isin([9201, 9203])]
los_df = los_df.loc[los_df['los']>0]
los_df = los_df[['person_id', 'visit_concept_id', 'los', 'days_to_cohort_start']]

list_temp_arrs = []
for i in tqdm(range(0,len(los_df))):
    n_repeats = los_df.iloc[i]['los']
    temp_arr = los_df.iloc[i].values.reshape(-1, 1).repeat(n_repeats, axis=0).reshape(4,n_repeats).T
    replace_days_to_cohort_start = np.arange(los_df.iloc[i]['days_to_cohort_start'], los_df.iloc[i]['days_to_cohort_start']-n_repeats, -1)
    temp_arr[:,-1] = replace_days_to_cohort_start
    list_temp_arrs.append(temp_arr)

los_df = pd.DataFrame(np.vstack(list_temp_arrs), columns = ['person_id', 'concept_id', 'los', 'days_to_cohort_start'])
los_df['concept_id'].replace({9201:9205, 9203:9207}, inplace=True)
los_df.drop(['los'], axis=1, inplace=True)
len(los_df)


# In[60]:


all_visits = all_visits[['person_id', 'days_to_cohort_start', 'visit_concept_id']].drop_duplicates()
all_visits.rename({'visit_concept_id':'concept_id'}, axis=1, inplace=True)
len(all_visits)


# In[61]:


all_meds = all_meds[['person_id', 'drug_exposure_count','drug_concept_id', 'days_to_cohort_start']].drop_duplicates()
# change the name to make merging the dfs easier later
all_meds.rename({'drug_concept_id':'concept_id'}, axis=1, inplace=True)

remove_rare_meds = all_meds[['person_id', 'concept_id']].drop_duplicates().value_counts('concept_id')
remove_rare_meds = list(remove_rare_meds[remove_rare_meds > 0.01*len(df_pop)].index)
all_meds = all_meds.loc[all_meds['concept_id'].isin(remove_rare_meds)]


# Creating an extended version of all_meds where each day that this person was under a given prescription is treated separately

# In[17]:


single_day_meds = all_meds.loc[all_meds['drug_exposure_count']==1]
print(len(single_day_meds))
multi_day_meds = all_meds.loc[all_meds['drug_exposure_count']>1]
print(len(multi_day_meds))
list_temp_arrs = []
for i in tqdm(range(0,len(multi_day_meds))):
    n_repeats = multi_day_meds.iloc[i]['drug_exposure_count']
    temp_arr = multi_day_meds.iloc[i].values.reshape(-1, 1).repeat(n_repeats, axis=0).reshape(4,n_repeats).T
    replace_days_to_cohort_start = np.arange(multi_day_meds.iloc[i]['days_to_cohort_start'], multi_day_meds.iloc[i]['days_to_cohort_start']-n_repeats, -1)
    temp_arr[:,-1] = replace_days_to_cohort_start
    list_temp_arrs.append(temp_arr)
    
multi_day_meds = pd.DataFrame(np.vstack(list_temp_arrs), columns = single_day_meds.columns)
all_meds = pd.concat([multi_day_meds, single_day_meds])

all_meds.drop(['drug_exposure_count'], axis=1, inplace=True)

len(all_meds)


# In[62]:


all_conds = all_conds[['person_id', 'condition_concept_id', 'days_to_cohort_start']].drop_duplicates()
# change the name to make merging the dfs easier later
all_conds.rename({'condition_concept_id':'concept_id'}, axis=1, inplace=True)

remove_rare_conds = all_conds[['person_id', 'concept_id']].drop_duplicates().value_counts('concept_id')
remove_rare_conds = list(remove_rare_conds[remove_rare_conds > 0.01*len(df_pop)].index)
all_conds = all_conds.loc[all_conds['concept_id'].isin(remove_rare_conds)]
len(all_conds)


# ### Create sparse dataframe

# In[63]:


all_features = pd.concat([all_conds, all_meds, los_df, all_visits, all_labs, all_procedures])
all_features['days_to_cohort_start'] = all_features['days_to_cohort_start']//90
print(len(all_features))


# In[64]:


grpr_row = all_features.groupby(['person_id', 'days_to_cohort_start']).grouper
idx_row = grpr_row.group_info[0]

grpr_col = all_features.groupby('concept_id').grouper
idx_col = grpr_col.group_info[0]

sparse_data = csr_matrix((all_features['concept_id'].values, (idx_row, idx_col)),shape=(grpr_row.ngroups, grpr_col.ngroups))


# In[65]:


#sparse_data[sparse_data>0] = 1
#print(sparse_data.sum()) # this is correct bc that's how many entries there are in the dataframe

df_index = grpr_row.result_index
df_columns = list(grpr_col.result_index)
sparse_df = pd.DataFrame.sparse.from_spmatrix(sparse_data, index = df_index, columns = df_columns)


# In[66]:


for i in sparse_df.columns:
    sparse_df[i] = sparse_df[i]/i


# In[67]:


sparse_df.dtypes


# # Create Table 1: Cohort Overview
# 
# ### Get demographic information

# In[68]:


table1 = pd.DataFrame(index = ['Number of patients','Race: Missing', 'Race: Black or African American', 'Race: White',
                              'Gender: Male', 'Gender: Female', 'Age: Mean', 'Age: STD',
                              'Missing: Conds', 'Missing: Meds', 'Missing: Visits', 'Missing: Procedures', 'Missing: Labs', 'Avg Dates per Patient'], columns = ['No Schizophrenia','Schizophrenia'])

table1.loc['Number of patients'] = df_pop.groupby('sz_flag').count()['person_id'].values
table1.loc[['Race: Missing', 'Race: Black or African American', 'Race: White']] = df_pop.pivot_table(index='race_concept_id', columns = 'sz_flag', aggfunc={'person_id':'count'}).values
table1.loc[['Gender: Male', 'Gender: Female']] = df_pop.pivot_table(index='gender_concept_id', columns = 'sz_flag', aggfunc={'person_id':'count'}).values

table1.loc['Age: Mean'] = df_pop.groupby('sz_flag')['age_diagnosis'].mean().values
table1.loc['Age: STD'] = df_pop.groupby('sz_flag')['age_diagnosis'].std().values


# ### Get the number of people with no recorded features for each type of data

# In[69]:


table1.loc['Missing: Conds']['Schizophrenia'] = len(df_pop.loc[(~df_pop['person_id'].isin(all_conds['person_id']))&(df_pop['sz_flag']==1)])
table1.loc['Missing: Conds']['No Schizophrenia'] = len(df_pop.loc[(~df_pop['person_id'].isin(all_conds['person_id']))&(df_pop['sz_flag']==0)])

table1.loc['Missing: Meds']['Schizophrenia'] = len(df_pop.loc[(~df_pop['person_id'].isin(all_meds['person_id']))&(df_pop['sz_flag']==1)])
table1.loc['Missing: Meds']['No Schizophrenia'] = len(df_pop.loc[(~df_pop['person_id'].isin(all_meds['person_id']))&(df_pop['sz_flag']==0)])

table1.loc['Missing: Labs']['Schizophrenia'] = len(df_pop.loc[(~df_pop['person_id'].isin(all_labs['person_id']))&(df_pop['sz_flag']==1)])
table1.loc['Missing: Labs']['No Schizophrenia'] = len(df_pop.loc[(~df_pop['person_id'].isin(all_labs['person_id']))&(df_pop['sz_flag']==0)])

table1.loc['Missing: Visits']['Schizophrenia'] = len(df_pop.loc[(~df_pop['person_id'].isin(all_visits['person_id']))&(df_pop['sz_flag']==1)])
table1.loc['Missing: Visits']['No Schizophrenia'] = len(df_pop.loc[(~df_pop['person_id'].isin(all_visits['person_id']))&(df_pop['sz_flag']==0)])

table1.loc['Missing: Procedures']['Schizophrenia'] = len(df_pop.loc[(~df_pop['person_id'].isin(all_procedures['person_id']))&(df_pop['sz_flag']==1)])
table1.loc['Missing: Procedures']['No Schizophrenia'] = len(df_pop.loc[(~df_pop['person_id'].isin(all_procedures['person_id']))&(df_pop['sz_flag']==0)])


# ### Get number of dates per person (on average) 

# In[70]:


table1.loc['Avg Dates per Patient'] = [sum(sparse_df.index.get_level_values(0).isin(df_pop.loc[df_pop['sz_flag']==1, 'person_id']))/len(df_pop.loc[df_pop['sz_flag']==0]),
                sum(sparse_df.index.get_level_values(0).isin(df_pop.loc[df_pop['sz_flag']==0, 'person_id']))/len(df_pop.loc[df_pop['sz_flag']==1])]


# ### Get percents

# In[71]:


table1['No Schizophrenia (%)'] = 0
table1['Schizophrenia (%)'] = 0
table1[['No Schizophrenia (%)', 'Schizophrenia (%)']]=100*table1[['No Schizophrenia', 'Schizophrenia']]/[120689, 1842]
table1.loc[['Number of patients', 'Age: Mean', 'Age: STD', 'Avg Dates per Patient'],['No Schizophrenia (%)', 'Schizophrenia (%)']] = np.nan


# In[72]:


table1


# # Make the dataframe into one with a standard shape 
# Each multi-index is pid + 90-day-increments

# In[24]:


index_patients = sparse_df.index.get_level_values(0).unique()
patient_indices = np.arange(0,len(index_patients), 1)

cohort_start_days = np.arange(1, all_features['days_to_cohort_start'].max()+1, 1)
feature_indices = np.arange(0, sparse_df.shape[1], 1)

df_features = pd.DataFrame(data = 0, index = pd.MultiIndex.from_product([index_patients, cohort_start_days], names=["person_id", "months"]), columns = sparse_df.columns, dtype='int8')


# In[58]:


sparse_df = sparse_df[sparse_df.index.get_level_values(1)>0]
print(datetime.now())
df_features.loc[sparse_df.index] = sparse_df.values
print(datetime.now())


# ### Create indices of train-test split and perform standard scaling

# In[59]:


# create my y_values vector in the correct order (according to index_patients)
mat_y = np.asarray(df_pop.set_index('person_id').loc[index_patients, 'sz_flag']).reshape(len(df_pop), 1)

# get first train-test split (to get test test data)
train_pop, test_pop, train_labels, test_labels, train_inds, test_inds = train_test_split(df_pop, mat_y, np.arange(0, len(df_pop), 1), random_state=23, test_size=0.2, stratify=mat_y)


# In[60]:


local_vars = list(locals().items())
for var, obj in local_vars:
    if sys.getsizeof(obj)*1e-6 > 1:
        print(var, sys.getsizeof(obj)*1e-9)


# In[62]:


train_sparse_mat = csr_matrix(df_features.loc[train_pop['person_id']].values)
print(train_sparse_mat.shape)

test_sparse_mat = csr_matrix(df_features.loc[test_pop['person_id']].values)
print(test_sparse_mat.shape)


# In[63]:


test_sparse_mat = test_sparse_mat.todense()
train_sparse_mat = train_sparse_mat.todense()


# In[64]:


scaler = StandardScaler()
train_sparse_mat = scaler.fit_transform(train_sparse_mat)
print('done with fit/first transform')
test_sparse_mat = scaler.transform(test_sparse_mat)


# In[65]:


# create my y_values vector that goes along with this ^^
y_train = mat_y[train_inds]
print(len(y_train))
y_test = mat_y[test_inds]
print(len(y_test))


# In[66]:


del df_features
del sparse_df
gc.collect()


# ### Save the following, noting that the indices are specifically lined up:
# - Sparse matrices for train_sparse_mat and test_sparse_mat
# - y_train and y_test

# In[67]:


#save_npz('psychosis_prediction/sparse_training_mat.npz', train_sparse_mat)
#save_npz('psychosis_prediction/sparse_testing_mat.npz', test_sparse_mat)
np.savez('psychosis_prediction/standard_scaled_mats.npz', train_sparse_mat, test_sparse_mat)
pd.DataFrame(y_train, index=train_inds).to_csv('psychosis_prediction/training_labels.csv')
pd.DataFrame(y_test, index=test_inds).to_csv('psychosis_prediction/testing_labels.csv')


# ### read in sparse training/testing matrices, as well as y_train and y_test

# In[ ]:


#train_sparse_mat = load_npz('psychosis_prediction/sparse_training_mat.npz')
#test_sparse_mat = load_npz('psychosis_prediction/sparse_testing_mat.npz')
loaded_mats = np.load('psychosis_prediction/standard_scaled_mats.npz')

y_train = pd.read_csv('psychosis_prediction/training_labels.csv', index_col = 0)
y_test = pd.read_csv('psychosis_prediction/testing_labels.csv', index_col = 0)


# In[ ]:


train_sparse_mat = loaded_mats['arr_0']
test_sparse_mat = loaded_mats['arr_1']


# In[ ]:


train_sparse_mat.shape


# In[ ]:


pca = IncrementalPCA(n_components=1990, batch_size = 100)
chunk_size = 10000
for i in tqdm(range(0, len(train_sparse_mat)//chunk_size)):
    pca.partial_fit(train_sparse_mat[i*chunk_size : (i+1)*chunk_size,:])

print('done fitting')

pickle.dump(pca, open("psychosis_prediction/pca.pkl","wb"))


# In[ ]:


pca = pickle.load(open("psychosis_prediction/pca.pkl",'rb'))
pca.components_.shape


# In[ ]:


print(datetime.now())
train_sparse_mat = pca.transform(train_sparse_mat)
print(datetime.now())
test_sparse_mat = pca.transform(test_sparse_mat)


# In[ ]:


chunk_size=50000
for i in (range(0, len(df_train_features)//chunk_size)):
    if i==0:
        train_pca_features =  pca.transform(df_train_features.iloc[i*chunk_size : (i+1)*chunk_size])
    else:
        tmp = pca.transform(df_train_features.iloc[i*chunk_size : (i+1)*chunk_size])
        train_pca_features = np.concatenate((train_pca_features, tmp), axis=0)
        save_npz('psychosis_prediction/training_pca.npz', csr_matrix(train_pca_features))
    
for i in tqdm(range(0, len(df_test_features)//chunk_size)):
    if i==0:
        test_pca_features =  pca.transform(df_test_features.iloc[i*chunk_size : (i+1)*chunk_size])
    else:
        tmp = pca.transform(df_test_features.iloc[i*chunk_size : (i+1)*chunk_size])
        test_pca_features = np.concatenate((test_pca_features, tmp), axis=0)
save_npz('psychosis_prediction/testing_pca.npz', csr_matrix(test_pca_features))




# In[ ]:


train_pca_features


# Create a multiindex (y_train.index for pid, 0 to ?? for the timing) for df_train and get a train validation split

# In[68]:


print(len(train_sparse_mat))
y_train = pd.read_csv('psychosis_prediction/training_labels.csv', index_col = 0)
y_test = pd.read_csv('psychosis_prediction/testing_labels.csv', index_col = 0)
train_pop, val_pop, train_labels, test_labels, train_inds, val_inds = train_test_split(y_train.index, y_train, np.arange(0, len(y_train), 1), random_state=24, test_size=0.1, stratify=y_train)
#df_val_features = df_train_features.loc[val_pop]
#df_train_features.drop(val_pop, axis=0, level = 0, inplace=True)
#print(len(df_train_features)+len(df_val_features))


# In[69]:


y_val = y_train.loc[val_pop]
y_train = y_train.loc[train_pop]


# reshape gives me patients * timeseries * features
# (https://stackoverflow.com/questions/54615882/how-to-convert-a-pandas-multiindex-dataframe-into-a-3d-array)
# 
# then I reorder to patients * features * timeseries (0,2,1)

# In[71]:


num_train_pop = len(y_train)
num_test_pop = len(y_test)
num_val_pop = len(y_val)
len_sequence = 28
num_features = train_sparse_mat.shape[1]
#val_mat_features = df_val_features.astype('float32').values.reshape(num_val_pop, len_sequence, num_features).transpose(0, 2, 1)
test_mat_features = test_sparse_mat.reshape(num_test_pop, len_sequence, num_features).transpose(0, 2, 1)
print(test_mat_features.shape)
train_mat_features = train_sparse_mat.reshape(num_train_pop+num_val_pop, len_sequence, num_features).transpose(0, 2, 1)
print(train_mat_features.shape)


# In[72]:


val_mat_features = train_mat_features[val_inds, :, :]
train_mat_features = train_mat_features[train_inds, :, :]


# In[ ]:


# set up the validation
kf = KFold(n_splits=5, random_state=23)
list_indices = list(kf.split(train_mat_features))

list_indices = list(kf.split(train_mat_features))
list_train_idx = []
list_val_idx = []

for fold in list_indices:
    list_train_idx.append(fold[0])
    list_val_idx.append(fold[1])


# In[73]:


local_vars = list(locals().items())
for var, obj in local_vars:
    if sys.getsizeof(obj)*1e-6 > 1:
        print(var, sys.getsizeof(obj)*1e-9)
sys.getsizeof(train_mat_features)*1e-9


# In[79]:


del train_sparse_mat
del test_sparse_mat
del all_features
gc.collect()


# In[ ]:


"""
# NEED TO FIGURE OUT HOW TO DO VALIDATION
x_train_split = train_mat_features[list_train_idx[0], :, :]
y_train_split = y_train[list_train_idx[0], :]

x_val_split = train_mat_features[list_val_idx[0], :, :]
y_val_split = y_train[list_val_idx[0], :]
"""


# In[ ]:


train_mat, val_mat, train_labels, val_labels, train_inds, val_inds = train_test_split(train_mat_features, y_train, np.arange(0, len(y_train), 1), random_state=24, test_size=0.1, stratify=y_train)


# In[ ]:


y_test.values


# In[74]:


train_data = torch.utils.data.TensorDataset(torch.Tensor(train_mat_features), torch.Tensor(y_train.values))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)

del train_data
gc.collect()


# In[75]:


val_data = torch.utils.data.TensorDataset(torch.Tensor(val_mat_features), torch.Tensor(y_val.values))
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)

del val_data
gc.collect()


# In[76]:


test_data = torch.utils.data.TensorDataset(torch.Tensor(test_mat_features), torch.Tensor(y_test.values))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

del test_data
gc.collect()


# In[77]:


#torch.save(train_loader, 'psychosis_prediction/train_loader_pca.pth')
#torch.save(val_loader, 'psychosis_prediction/val_loader_pca.pth')
#torch.save(test_loader, 'psychosis_prediction/test_loader_pca.pth')

torch.save(train_loader, 'psychosis_prediction/train_loader.pth')
torch.save(val_loader, 'psychosis_prediction/val_loader.pth')
torch.save(test_loader, 'psychosis_prediction/test_loader.pth')


# In[78]:


print('done')


# # actual model training

# In[6]:


class AttentionModel(torch.nn.Module):
    def __init__(self, hidden_size, feature_size, data):
        super(AttentionModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        # self.W_s1 = nn.Linear(hidden_size, 350)
        # self.W_s2 = nn.Linear(350, 30)
        self.W_s1 = nn.Linear(hidden_size, 1)
        # self.fc_layer = nn.Linear(30 * hidden_size, 2000)
        self.rnn = nn.GRU(self.feature_size, self.hidden_size, bidirectional=False)
        if data=='mimic':
            """
            self.regressor = nn.Sequential(nn.BatchNorm1d(num_features=hidden_size),
                                       nn.LeakyReLU(),
                                       nn.Linear(hidden_size, 5),
                                       nn.Dropout(0.5),
                                       nn.LeakyReLU(),
                                       nn.Linear(5, hidden_size),
                                       nn.LeakyReLU(),
                                       nn.Linear(hidden_size, 1),
                                       nn.LeakyReLU(),
                                       nn.Sigmoid())
            """
            self.regressor = nn.Sequential(nn.BatchNorm1d(num_features=hidden_size),
                                       nn.LeakyReLU(),
                                       nn.Dropout(0.5),
                                       nn.LeakyReLU(),
                                       nn.Linear(hidden_size, 1),
                                       nn.LeakyReLU(),
                                       nn.Sigmoid())
        elif data=='ghg':
            self.regressor = nn.Sequential(#nn.BatchNorm1d(self.hidden_size),
                                       nn.Linear(hidden_size,200),
                                       nn.LeakyReLU(),
                                       nn.Linear(200,200),
                                       nn.LeakyReLU(),
                                       nn.Linear(200,200),
                                       nn.LeakyReLU(),
                                       #nn.Dropout(0.5),
                                       nn.Linear(200, 1))
        elif 'simulation' in data:
            self.regressor = nn.Sequential(nn.BatchNorm1d(num_features=hidden_size),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(hidden_size, 1),
                                       nn.Sigmoid())

    def attention_net(self, lstm_output):
        attn_weight_vector = F.tanh(self.W_s1(lstm_output))
        attn_weight_vector = torch.nn.Softmax(dim=1)(attn_weight_vector)
        scaled_latent = lstm_output*attn_weight_vector
        return torch.sum(scaled_latent, dim=1), attn_weight_vector

    def forward(self, input):
        input = input.to(self.device)
        batch_size = input.shape[0]
        input = input.permute(2, 0, 1) # Input to GRU should be (seq_len, batch, input_size)
        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size)).to(self.device) #(num_layers * num_directions, batch, hidden_size)

        output, final_hidden_state = self.rnn(input, h_0)   # output.size() =  (seq_len, batch, hidden_size)
                                                            # final_hidden_state.size() = (1, batch, hidden_size)
        output = output.permute(1, 0, 2)  # output.size() = (batch, seq_len, hidden_size)

        concept_vector, attn_weights = self.attention_net(output)         # attn_weight_matrix.size() = (batch_size, num_seq)
        #hidden_matrix = torch.bmm(attn_weight_matrix, output)   # hidden_matrix.size() = (batch_size, r, hidden_size)
        #fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))
        p = self.regressor(concept_vector)
        return p

    def get_attention_weights(self, input):
        input = input.to(self.device)
        batch_size = input.shape[0]
        input = input.permute(2, 0, 1) # Input to GRU should be (seq_len, batch, input_size)
        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size)).to(self.device) #(num_layers * num_directions, batch, hidden_size)

        output, final_hidden_state = self.rnn(input, h_0)   # output.size() =  (seq_len, batch, hidden_size)
                                                            # final_hidden_state.size() = (1, batch, hidden_size)
        output = output.permute(1, 0, 2)  # output.size() = (batch, seq_len, hidden_size)

        _, attn_weights = self.attention_net(output)
        return attn_weights



# In[12]:


def train(train_loader, model, device, optimizer, loss_criterion=torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([67]).to(device))):
    model = model.to(device)
    model.train()
    auc_train = 0
    recall_train, precision_train, auc_train, correct_label, epoch_loss = 0, 0, 0, 0, 0
    list_training_loss = []
    true_ys = []
    pred_ys = []
    for i, (signals, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        signals, labels = torch.Tensor(signals.float()).to(device), torch.Tensor(labels.float()).to(device)
        #labels = labels.view(labels.shape[0], )
        logits = model(signals) #probability of one of the outputs
        pred_labels = (logits > 0.5)*1

        true_ys.append(labels.detach().cpu().numpy())
        pred_ys.append(logits.detach().cpu().numpy())

        """
        risks = torch.nn.Softmax(-1)(logits)[:,1] # should this be logit?

        # need to replace this with labels = (logits > 0.5)*1; pred_proba = logits (?) 
        # WHAT I WANT IS OUTPUT of predicted probability because loss should be loss(output, true labels)
        label_onehot = torch.zeros(logits.shape).to(device)
        pred_onehot = torch.zeros(logits.shape).to(device)
        _, predicted_label = logits.max(1)
        pred_onehot.scatter_(1, predicted_label.view(-1, 1), 1)

        # labels_th = (labels[:,t]>0.5).float()
        label_onehot.zero_()
        label_onehot.scatter_(1, labels.long().view(-1, 1), 1)
        """
        
        # auc, recall, precision, correct = evaluate(labels_th.contiguous().view(-1), predicted_label.contiguous().view(-1), predictions.contiguous().view(-1))
        auc, recall, precision, correct = evaluate(labels, pred_labels, logits)

        # auc, recall, precision, correct = evaluate(label_onehot, predicted_label, risks)
        correct_label += correct
        auc_train = auc_train + auc
        recall_train = + recall
        precision_train = + precision

        loss = loss_criterion(logits, labels)
        epoch_loss = + loss.item()
        list_training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    true_ys_flattened = np.concatenate(true_ys).ravel()
    pred_ys_flattened = np.concatenate(pred_ys).ravel()
    auc_train = roc_auc_score(true_ys_flattened, pred_ys_flattened)
    auprc_train = average_precision_score(true_ys_flattened, pred_ys_flattened)

    return recall_train, precision_train, auc_train,auprc_train, correct_label, np.mean(list_training_loss), i + 1

def test(test_loader, model, device, criteria=torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([67]).to(device)), verbose=True):
    model.to(device)
    correct_label = 0
    recall_test, precision_test, auc_test = 0, 0, 0
    count = 0
    loss = 0
    total = 0
    auc_test = 0
    model.eval()
    list_testing_loss = []
    true_ys = []
    pred_ys = []
    for i, (x, y) in enumerate(test_loader):
        x, y = torch.Tensor(x.float()).to(device), torch.Tensor(y.float()).to(device)
        out = model(x)
        #y = y.view(y.shape[0], )
        
        pred_labels = (out > 0.5)*1
        
        """

        risks = torch.nn.Softmax(-1)(out)[:,1]

        label_onehot = torch.zeros(out.shape).to(device)
        pred_onehot = torch.zeros(out.shape).to(device)
        _, predicted_label = out.max(1)
        pred_onehot.scatter_(1, predicted_label.view(-1, 1), 1)

        label_onehot.zero_()
        label_onehot.scatter_(1, y.long().view(-1, 1), 1)
        """

        auc, recall, precision, correct = evaluate(y, pred_labels, out)
        true_ys.append(y.detach().cpu().numpy())
        pred_ys.append(out.detach().cpu().numpy())

        # prediction = (out > 0.5).view(len(y), ).float()
        # auc, recall, precision, correct = evaluate(y, prediction, out)
        correct_label += correct
        auc_test = auc_test + auc
        recall_test = + recall
        precision_test = + precision
        count = + 1
        loss += criteria(out, y).item()
        list_testing_loss.append(criteria(out, y).item())
        total += len(x)
    true_ys_flattened = np.concatenate(true_ys).ravel()
    pred_ys_flattened = np.concatenate(pred_ys).ravel()
    print(pred_ys_flattened.max())
    auc_test = roc_auc_score(true_ys_flattened, pred_ys_flattened)
    auprc_test = average_precision_score(true_ys_flattened, pred_ys_flattened)
    return recall_test, precision_test, auc_test, auprc_test, correct_label, np.mean(list_testing_loss)


def evaluate(labels, predicted_label, predicted_probability):
    labels_array = labels.detach().cpu().numpy()
    prediction_array = predicted_label.detach().cpu().numpy()
        
    # the if statement is for if we only have one predicted value in our labels_array
    if len(np.unique(labels_array)) >= 2:
        auc = roc_auc_score(labels_array, np.array(predicted_probability.detach().cpu()))
        report = classification_report(labels_array, prediction_array, output_dict=True, zero_division=0)
        recall = report['macro avg']['recall']
        precision = report['macro avg']['precision']
    else:
        auc = 0
        recall = 0
        precision = 0
    correct_label = np.equal(labels_array, prediction_array).sum()
    return auc, recall, precision, correct_label

def train_model(model, train_loader, valid_loader, optimizer, n_epochs, device ,cv=0):
    train_loss_trend = []
    test_loss_trend = []

    for epoch in range(n_epochs + 1):
        recall_train, precision_train, auc_train, auprc_train, correct_label_train, epoch_loss, n_batches = train(train_loader,
                                                                                                     model,
                                                                                                     device, optimizer)
        recall_test, precision_test, auc_test, auprc_test, correct_label_test, test_loss = test(valid_loader, model,
                                                                                    device)
        train_loss_trend.append(epoch_loss)
        test_loss_trend.append(test_loss)
        if epoch % 5 == 0:
            print('\nEpoch %d' % (epoch))
            print('Training ===>loss: ', epoch_loss,
                  ' Accuracy: %.2f percent' % (100 * correct_label_train / (len(train_loader.dataset))),
                  ' AUC: %.2f' % (auc_train),
                 ' AUPRC: %.2f' % (auprc_train))
            print('Test ===>loss: ', test_loss,
                  ' Accuracy: %.2f percent' % (100 * correct_label_test / (len(valid_loader.dataset))),
                  ' AUC: %.2f' % (auc_test),
                 ' AUPRC: %.2f' % (auprc_test))
        """
        if epoch > 10:
            if test_loss_trend[-2] < test_loss_trend[-1]:
                print('Breaking at epoch', epoch)
                break
        """

    # Save model and results
    """
    if not os.path.exists(os.path.join("./ckpt/", data)):
        os.mkdir(os.path.join("./ckpt/", data))
    if not os.path.exists(os.path.join("./plots/", data)):
        os.mkdir(os.path.join("./plots/", data))
    """
    torch.save(model.state_dict(), 'psychosis_prediction/models/new_features_model1.pt')
    plt.plot(train_loss_trend, label='Train loss')
    plt.plot(test_loss_trend, label='Validation loss')
    plt.legend()
    plt.savefig('psychosis_prediction/models/new_features_model1_loss.pdf')
    


# In[15]:


device = torch.device("cuda:0") 
#device = torch.device("cpu")

#train_loader = torch.load('psychosis_prediction/train_loader.pth')
#val_loader = torch.load('psychosis_prediction/val_loader.pth')


n_epochs = 60
batch_size=300
feature_size = 1992#train_mat_features.shape[1]

#for train_loader, val_loader in zip()
model = AttentionModel(hidden_size = 6, feature_size = feature_size, data = 'mimic')
parameters = model.parameters()
optimizer = torch.optim.Adam(parameters, lr=1e-6)

train_model(model, train_loader, val_loader, optimizer, n_epochs, device ,cv=0)


# ## Data: Medications as counted days, count overnight stays
# ### First try: norm --> leaky ReLU --> dropout leaky relu --> linear --> leaky relu --> sigmoid
# - Tried for hidden size = 10, 50
# - Only predicting the negative class
# ### Second try: 

# - significantly reduce the learning rate (1e-5)
# - Try just the GRU/LSTM (RNNs)
# - Consider a transformer model
# - Update loader to have a bigger batch (and maybe startify the batches)
# - Re-initialize the LSTM (it should take H_0)

# In[ ]:




