"""
# Create Cohort
1. (In DataGrip) Get all the patients with SCZ/schizoaffective disorder and 7 years of prior observation and save them to the ak4885_schizophrenia_incidence table
2. (In DataGrip) Get all patients who have an episode of psychosis (incl. schizophrenia) and 7 years of observation overall -- save this into results as ak4885_psychosis_cohort
3. Find all people who are between 10 and 35 years at "cohort start" (SCZ diagnosis or observation period end date)
4. Eliminate people with SCZ diagnoses from the nosz_conds df AND make sure that the instance of SCZ is not Schizophreniform disorder (444434, 4184004, 4263364) for the SCZ population
5. Eliminate all people who's first episode of psychosis is schizophrenia/schizoaffective disorder is their schizophrenia diagnosis
6. Get all conditions in 7 years prior to cohort start for both of the above tables
7. Combine dataframes (SCZ and No SCZ) and add SCZ "flag"
"""

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
import pickle 

connection_string = "CONNECTION TO DATABASE HERE"
conn = pyodbc.connect(connection_string)

"""
GET PSYCHOSIS CODES (see eTable 1)
"""
all_psychosis_codes_query = ("SELECT c_rel.concept_id as standard_concept_id, c_icd10.concept_code as icd_code, c_rel.concept_name as standard_concept_name, c_icd10.concept_name as icd_concept_name FROM dbo.concept as c_icd10 LEFT JOIN dbo.concept_relationship as rel on rel.concept_id_1 = c_icd10.concept_id "+
                       "LEFT JOIN dbo.concept as c_rel on rel.concept_id_2 = c_rel.concept_id "+
                         "WHERE (rel.relationship_id = 'Maps to' AND c_rel.standard_concept = 'S') AND (((c_icd10.concept_code IN ('295', '297', '298', '260.0', '260.1', '296.2', '296.5', '296.6', '296.24', '296.34', '291.3', '291.5', '292.1') OR c_icd10.concept_code LIKE '29[578]%') AND c_icd10.vocabulary_id = 'ICD9CM') "+
                         "OR ((c_icd10.concept_code LIKE 'F2[023456789]%' OR c_icd10.concept_code LIKE 'F30.[1234]' OR c_icd10.concept_code LIKE 'F31.[01234567]%' OR c_icd10.concept_code IN ('F32.3', 'F33.3', 'F53.1') OR c_icd10.concept_code LIKE 'F1_.15' OR c_icd10.concept_code LIKE 'F__.25' OR c_icd10.concept_code LIKE 'F__.95') AND c_icd10.vocabulary_id = 'ICD10CM'))")


psychosis_codes = pd.io.sql.read_sql(all_psychosis_codes_query, conn)
psychosis_codes.to_csv('psychosis_prediction/all_psychosis_codes.csv')

"""
GET SCHIZOPHRENIA CODES (see eTable 2)
"""
all_scz_codes_query = ("SELECT c_new.concept_id as standard_concept_id, c_icd10.concept_code as icd_code, c_new.concept_name as standard_concept_name, c_icd10.concept_name as icd_name FROM dbo.concept as c_icd10 LEFT JOIN dbo.concept_relationship as rel on rel.concept_id_1 = c_icd10.concept_id "+
                "LEFT JOIN dbo.concept as c_rel on rel.concept_id_2 = c_rel.concept_id "+
                "LEFT JOIN dbo.concept_ancestor as ca ON ca.ancestor_concept_id = rel.concept_id_2 "+
                "LEFT JOIN dbo.concept as c_new on c_new.concept_id = ca.descendant_concept_id " +
                "WHERE (rel.relationship_id = 'Maps to' AND c_new.standard_concept = 'S') "+
                "AND ((c_icd10.concept_code LIKE '295%' AND c_icd10.vocabulary_id = 'ICD9CM') "+
                "OR ((c_icd10.concept_code LIKE 'F2[05]%' AND c_icd10.vocabulary_id = 'ICD10CM')))")

all_scz_codes = pd.io.sql.read_sql(all_scz_codes_query, conn)
all_scz_codes.to_csv('psychosis_prediction/all_scz_codes.csv')

"""
Everyone from original dataset
- SCZ: at least one schizophrenia code and 7 years prior observation (non-continuous)
- Psychosis: at least one psychosis code and 7 years observation total (non-continuous); remove people also in SCZ cohort
"""
df_psychosis_all = pd.io.sql.read_sql("SELECT pc.*, year_of_birth, race_concept_id, gender_concept_id FROM results.ak4885_psychosis_cohort as pc LEFT JOIN dbo.person as p ON p.person_id = pc.person_id", conn)
df_scz_all = pd.io.sql.read_sql("SELECT sc.*, year_of_birth, race_concept_id, gender_concept_id FROM results.ak4885_schizophrenia_cohort as sc LEFT JOIN dbo.person as p ON p.person_id = sc.person_id", conn)
df_scz_all = df_scz_all.merge(df_psychosis_all[['person_id', 'psychosis_dx_date']], how='left', left_on = 'person_id', right_on = 'person_id')
if df_scz_all.isna().sum().sum() > 0:
    print('Undefined psychosis diagnosis date after merge')
df_psychosis_all = df_psychosis_all.loc[~df_psychosis_all['person_id'].isin(list(df_scz_all['person_id']))]
print("Get people from SQL Cohorts", len(df_psychosis_all), len(df_scz_all), len(df_scz_all)*100/(len(df_scz_all)+len(df_psychosis_all)))

"""
Make sure that people in the schizophrenia cohort have at least 1 dx (which is not schizophreniform disorder
Ignore the fact that these variables are called "dx_twice_pids"
"""
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
print("Remove schizophreniform disorder", len(df_psychosis_all), len(df_scz_all), len(df_scz_all)*100/(len(df_scz_all)+len(df_psychosis_all)))

"""
Make sure that people in the non-schizophrenia cohort have no instances of a schizophrenia diagnosis 
Do this by getting all conditions for people in the psychosis cohort and then removing anyone with any schizophrenia code at any point in time (they can have schizophrenifrom diagnosis). 
"""
df_psychosis_all = df_psychosis_all.loc[~(df_psychosis_all['person_id'].isin(list(all_sz_dx['person_id'].unique())))]
psychosis_conds = pd.io.sql.read_sql("SELECT DISTINCT pc.person_id, condition_concept_id, condition_start_date FROM results.ak4885_psychosis_cohort as pc LEFT JOIN dbo.condition_occurrence as co ON co.person_id = pc.person_id", conn)

scz_codes = all_scz_codes.loc[~all_scz_codes['standard_concept_id'].isin([444434, 4184004, 4263364])]['standard_concept_id']
scz_in_psychosis = psychosis_conds.loc[psychosis_conds['condition_concept_id'].isin(scz_codes)]
df_psychosis_all = df_psychosis_all.loc[~df_psychosis_all['person_id'].isin(scz_in_psychosis)]

print("Remove people from psychosis cohort w/ SCZ",len(df_psychosis_all), len(df_scz_all), len(df_scz_all)*100/(len(df_scz_all)+len(df_psychosis_all)))

"""
Make sure that people in the schizophrenia cohort have an accurate cohort_start_date
"""
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

"""
Make sure that everyone has an accurate psychosis_dx_date
"""
psych_in_scz = scz_conds.loc[scz_conds['condition_concept_id'].isin(psychosis_codes['standard_concept_id'])]
psych_in_scz = psych_in_scz.merge(df_scz_all, how='outer', left_on='person_id', right_on='person_id')
psych_in_scz['psychosis_dx_date'] = pd.to_datetime(psych_in_scz['psychosis_dx_date'], format='mixed')
psych_in_scz['condition_start_date'] = pd.to_datetime(psych_in_scz['condition_start_date'])

min_psych_start = psych_in_scz.groupby('person_id')['condition_start_date'].min()
min_psych_start.name = 'min_psych_start'
psych_in_scz = psych_in_scz.merge(min_psych_start, how='left', left_on='person_id', right_index=True)
psych_in_scz.loc[psych_in_scz['min_psych_start']<psych_in_scz['psychosis_dx_date'], 'psychosis_dx_date'] = psych_in_scz.loc[psych_in_scz['min_psych_start']<psych_in_scz['psychosis_dx_date'], 'min_psych_start']

df_scz_all.drop(['psychosis_dx_date'], axis=1, inplace=True)
df_scz_all = df_scz_all.merge(psych_in_scz[['person_id', 'psychosis_dx_date']].drop_duplicates(), how='left', left_on = 'person_id', right_on = 'person_id')

psych_in_psych = psychosis_conds.loc[psychosis_conds['condition_concept_id'].isin(psychosis_codes['standard_concept_id'])]
psych_in_psych = psych_in_psych.merge(df_psychosis_all, how='outer', left_on='person_id', right_on='person_id')
psych_in_psych['psychosis_dx_date'] = pd.to_datetime(psych_in_psych['psychosis_dx_date'], format='mixed')
psych_in_psych['condition_start_date'] = pd.to_datetime(psych_in_psych['condition_start_date'])

min_psych_psych_start = psych_in_psych.groupby('person_id')['condition_start_date'].min()
min_psych_psych_start.name = 'min_psych_psych_start'
psych_in_psych = psych_in_psych.merge(min_psych_psych_start, how='left', left_on='person_id', right_index=True)
psych_in_psych.loc[psych_in_psych['min_psych_psych_start']<psych_in_psych['psychosis_dx_date'], 'psychosis_dx_date'] = psych_in_psych.loc[psych_in_psych['min_psych_psych_start']<psych_in_psych['psychosis_dx_date'], 'min_psych_psych_start']

df_psychosis_all.drop(['psychosis_dx_date'], axis=1, inplace=True)
df_psychosis_all = df_psychosis_all.merge(psych_in_psych[['person_id', 'psychosis_dx_date']].drop_duplicates(), how='left', left_on = 'person_id', right_on = 'person_id')

"""
Ages 10-35 at "cohort start date" 
(end of observation for psychosis patients, first SCZ diagnosis for SCZ patients)
"""
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

print("Fix ages in cohort", len(df_psychosis_all), len(df_scz_all), len(df_scz_all)*100/(len(df_scz_all)+len(df_psychosis_all)))

"""
Now limit to people whos first diagnosis of SCZ is AFTER their first episode of psychosis
Restrict to people for whom the cohort start date (schizophrenia diagnosis date) is AFTER the first date of psychosis
"""
df_scz_all['psychosis_dx_date'] = pd.to_datetime(df_scz_all['psychosis_dx_date'])
df_scz_all = df_scz_all.loc[df_scz_all['cohort_start_date']>df_scz_all['psychosis_dx_date']]
print("Psychosis before schizophrenia", len(df_psychosis_all), len(df_scz_all), len(df_scz_all)*100/(len(df_scz_all)+len(df_psychosis_all)))

"""
CONDITIONS + CONTINUOUS CARE
"""
sz_conds_query = ("SELECT sz.*, co.condition_start_date, co.condition_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                  "LEFT JOIN cdm_mdcd.dbo.condition_occurrence as co on co.person_id = sz.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = co.condition_concept_id "+
                  "WHERE condition_concept_id > 0")
sz_conds = pd.io.sql.read_sql(sz_conds_query, conn)
nosz_conds_query = ("SELECT pc.*, co.condition_start_date, co.condition_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                  "LEFT JOIN cdm_mdcd.dbo.condition_occurrence as co on co.person_id = pc.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = co.condition_concept_id "+
                  "WHERE condition_concept_id > 0")

list_chunks = []
for chunk in pd.io.sql.read_sql(nosz_conds_query, conn, chunksize=500000):
    list_chunks.append(chunk.loc[chunk['person_id'].isin(df_psychosis_all['person_id'])])
nosz_conds = pd.concat(list_chunks)
nosz_conds = nosz_conds.loc[nosz_conds['person_id'].isin(list(df_psychosis_all['person_id']))]

sz_conds = sz_conds.loc[sz_conds['person_id'].isin(list(df_scz_all['person_id']))]

nosz_conds['cohort_start_date'] = nosz_conds['end_date']
sz_conds = sz_conds.merge(df_scz_all[['person_id', 'psychosis_dx_date']], how='left', left_on = 'person_id', right_on = 'person_id')
sz_conds['sz_flag'] = 1
nosz_conds['sz_flag'] = 0
all_conds = pd.concat([sz_conds, nosz_conds])

df_psychosis_all['sz_flag'] = 0
df_scz_all['sz_flag'] = 1

df_psychosis_all['cohort_start_date'] = df_psychosis_all['end_date']

df_pop = pd.concat([df_psychosis_all, df_scz_all])
print("Overall population:", len(df_pop), sum(df_pop['sz_flag'])*100/len(df_pop))

"""
Constrict cohort based on continuous care
First drop instances of conditions where the condition concept id is not defined to ensure that there is at least 1 service contact per year
Calculate the differences between consecutive condition occurrences for each patient -- do this by making sure that:
1. there are at least 7 unique dates that there is a visit
2. there's at least one visit > 6 years before diagnosis
3. the max difference between consecutive dates is 1 year (inclusive)
"""
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

pd.DataFrame(yearly_service_pids).to_csv('psychosis_prediction/yearly_service_pids.csv', index=False)

df_psychosis_all = df_psychosis_all.loc[df_psychosis_all['person_id'].isin(yearly_service_pids)]
df_scz_all = df_scz_all.loc[df_scz_all['person_id'].isin(yearly_service_pids)]

df_pop = df_pop.loc[df_pop['person_id'].isin(yearly_service_pids)]
print("One service contact per year", len(df_pop))

"""
Maximum of 45 days with no insurance coverage
Get insurance information from all people with at least 7 years observation and then limit to only the people in with at least 1 service visit per year
Combine all of the overlapping payer periods, with a grace period of 45 days between coverage periods
"""
insurance_query = ("SELECT ppp.PERSON_ID, ppp.PAYER_PLAN_PERIOD_START_DATE, ppp.PAYER_PLAN_PERIOD_END_DATE, ppp.PAYER_SOURCE_VALUE "+
                   "FROM dbo.PAYER_PLAN_PERIOD as ppp LEFT JOIN dbo.OBSERVATION_PERIOD as op ON op.person_id = ppp.PERSON_ID "+
                   "WHERE DATEDIFF(day, OBSERVATION_PERIOD_START_DATE, OBSERVATION_PERIOD_END_DATE) > 2555")
insurance_df = pd.io.sql.read_sql(insurance_query, conn)
insurance_df = insurance_df.loc[insurance_df['PERSON_ID'].isin(yearly_service_pids)]
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

"""
Now that we have adjusted for the combined insurance periods with a 45-day grace period, we want to make sure that it extends from 7 years pre-diagnosis to the date of diagnosis
"""
df_pop['cohort_start_date'] =  pd.to_datetime(df_pop['cohort_start_date'], format='%Y-%m-%d')

insurance_check_df = df_pop.merge(merged_insurance_df, how = 'left', left_on = 'person_id', right_on = 'PERSON_ID')
eligible_pids = insurance_check_df.loc[(insurance_check_df['PAYER_PLAN_PERIOD_END_DATE']>=insurance_check_df['cohort_start_date'])&(insurance_check_df['PAYER_PLAN_PERIOD_START_DATE'] <= insurance_check_df['cohort_start_date']- pd.Timedelta(days=2555))]['person_id'].unique()

eligible_pids = list(eligible_pids)
df_psychosis_all = df_psychosis_all.loc[df_psychosis_all['person_id'].isin(eligible_pids)]
df_scz_all = df_scz_all.loc[df_scz_all['person_id'].isin(eligible_pids)]

df_psychosis_all['sz_flag'] = 0
df_scz_all['sz_flag'] = 1
df_pop = pd.concat([df_psychosis_all, df_scz_all])
print("No large (>45 day) gaps in insurance coverage", len(df_pop), sum(df_pop['sz_flag'])*100/len(df_pop))

df_pop.to_csv('psychosis_prediction/population.csv', index=False)
pd.DataFrame(eligible_pids).to_csv('psychosis_prediction/insurance_pids.csv', index=False)
pd.DataFrame(yearly_service_pids).to_csv('psychosis_prediction/yearly_service_pids.csv', index=False)

print('Total patients:',len(df_pop))
print('% Patients with Schizophrenia:',100*sum(df_pop['sz_flag'])/len(df_pop))

"""
SAVE CONDIIIONS
"""
all_conds = all_conds.loc[all_conds['person_id'].isin(eligible_pids)]
all_conds.drop(['cohort_start_date'], axis=1, inplace=True)
all_conds = all_conds.merge(df_pop[['person_id', 'cohort_start_date']], how='left', left_on = 'person_id', right_on = 'person_id')
all_conds.to_csv('psychosis_prediction/temporal_conditions.csv', index=False)

"""
MEDICATIONS
"""
sz_meds_query = ("SELECT sz.*, drug_concept_id, drug_era_start_date, drug_era_end_date, drug_exposure_count, gap_days "+ 
                 "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                   "LEFT JOIN cdm_mdcd.dbo.drug_era on drug_era.person_id = sz.person_id")


sz_meds = pd.io.sql.read_sql(sz_meds_query, conn)
sz_meds.columns = sz_meds.columns.str.lower()

nosz_meds_query = ("SELECT pc.*, drug_concept_id, drug_era_start_date, drug_era_end_date, drug_exposure_count, gap_days "+ 
                 "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                   "LEFT JOIN cdm_mdcd.dbo.drug_era on drug_era.person_id = pc.person_id")
list_chunks = []
for chunk in pd.io.sql.read_sql(nosz_meds_query, conn, chunksize=1000000):
    list_chunks.append(chunk.loc[chunk['person_id'].isin(df_pop['person_id'])])
nosz_meds = pd.concat(list_chunks)

sz_meds = sz_meds.merge(df_pop[['person_id', 'psychosis_dx_date']], how='left', left_on = 'person_id', right_on='person_id')
nosz_meds['cohort_start_date'] = nosz_meds['end_date']
all_meds = pd.concat([sz_meds, nosz_meds])

all_meds = all_meds.loc[all_meds['person_id'].isin(list(df_pop['person_id']))]
all_meds.drop(['cohort_start_date'], axis=1, inplace=True)
all_meds = all_meds.merge(df_pop[['person_id', 'cohort_start_date']], how='left', left_on = 'person_id', right_on = 'person_id')
all_meds.dropna(inplace=True)
all_meds.to_csv('psychosis_prediction/temporal_medications.csv')

"""
VISITS
"""
sz_visits_query = ("SELECT sz.*, visit_occurrence_id, visit_concept_id, visit_start_date, visit_end_date, visit_type_concept_id " +
                   "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                   "LEFT JOIN cdm_mdcd.dbo.visit_occurrence as v on v.person_id = sz.person_id")

sz_visits = pd.io.sql.read_sql(sz_visits_query, conn)
nosz_visits_query = ("SELECT pc.*, visit_occurrence_id, visit_concept_id, visit_start_date, visit_end_date, visit_type_concept_id " +
                   "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                   "LEFT JOIN cdm_mdcd.dbo.visit_occurrence as v on v.person_id = pc.person_id")

list_chunks = []
for chunk in pd.io.sql.read_sql(nosz_visits_query, conn, chunksize=1000000):
    list_chunks.append(chunk.loc[chunk['person_id'].isin(df_pop['person_id'])])
nosz_visits = pd.concat(list_chunks)
sz_visits = sz_visits.merge(df_pop[['person_id', 'psychosis_dx_date']], how='left', left_on = 'person_id', right_on='person_id')
nosz_visits['cohort_start_date'] = nosz_visits['end_date']

all_visits = pd.concat([sz_visits, nosz_visits])

df_pop = pd.read_csv('psychosis_prediction/population.csv')
all_visits = all_visits.loc[all_visits['person_id'].isin(list(df_pop['person_id']))]
all_visits.drop(['cohort_start_date'], axis=1, inplace=True)
all_visits = all_visits.merge(df_pop[['person_id', 'cohort_start_date']], how='left', left_on = 'person_id', right_on = 'person_id')
all_visits.to_csv('psychosis_prediction/temporal_visits.csv')

"""
PROCEDURES
"""
sz_procedures_query = ("SELECT sz.*, procedure_date, procedure_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                  "LEFT JOIN cdm_mdcd.dbo.procedure_occurrence as po on po.person_id = sz.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = po.procedure_concept_id")

sz_procedures = pd.io.sql.read_sql(sz_procedures_query, conn)
nosz_procedures_query = ("SELECT pc.*, procedure_date, procedure_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                  "LEFT JOIN cdm_mdcd.dbo.procedure_occurrence as po on po.person_id = pc.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = po.procedure_concept_id")

list_chunks = []
for chunk in pd.io.sql.read_sql(nosz_procedures_query, conn, chunksize=1000000):
    list_chunks.append(chunk.loc[chunk['person_id'].isin(df_pop['person_id'])])
nosz_procedures = pd.concat(list_chunks)

sz_procedures = sz_procedures.merge(df_pop[['person_id', 'psychosis_dx_date']], how='left', left_on = 'person_id', right_on='person_id')
nosz_procedures['cohort_start_date'] = nosz_procedures['end_date']

all_procedures = pd.concat([sz_procedures, nosz_procedures])
all_procedures = all_procedures.loc[all_procedures['procedure_concept_id']>0]

df_pop = pd.read_csv('psychosis_prediction/population.csv')
all_procedures = all_procedures.loc[all_procedures['person_id'].isin(list(df_pop['person_id']))]
all_procedures.dropna(inplace=True)
all_procedures.drop(['cohort_start_date'], axis=1, inplace=True)
all_procedures = all_procedures.merge(df_pop[['person_id', 'cohort_start_date']], how='left', left_on = 'person_id', right_on = 'person_id')
all_procedures.to_csv('psychosis_prediction/temporal_procedures.csv')

"""
LABS
"""
sz_measurement_query = ("SELECT sz.*, measurement_date, measurement_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                  "LEFT JOIN cdm_mdcd.dbo.measurement as m on m.person_id = sz.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = m.measurement_concept_id")

sz_labs = pd.io.sql.read_sql(sz_measurement_query, conn)

nosz_measurements_query = ("SELECT pc.*, measurement_date, measurement_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                  "LEFT JOIN cdm_mdcd.dbo.measurement as m on m.person_id = pc.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = m.measurement_concept_id")
list_chunks = []
for chunk in pd.io.sql.read_sql(nosz_measurements_query, conn, chunksize=1000000):
    list_chunks.append(chunk.loc[chunk['person_id'].isin(df_pop['person_id'])])
nosz_labs = pd.concat(list_chunks)

sz_labs = sz_labs.merge(df_pop[['person_id', 'psychosis_dx_date']], how='left', left_on = 'person_id', right_on='person_id')
nosz_labs['cohort_start_date'] = nosz_labs['end_date']

all_labs = pd.concat([sz_labs, nosz_labs])
all_labs = all_labs.loc[all_labs['measurement_concept_id']>0]
df_pop = pd.read_csv('psychosis_prediction/population.csv')
all_labs = all_labs.loc[all_labs['person_id'].isin(list(df_pop['person_id']))]
all_labs.dropna(inplace=True)
all_labs.drop(['cohort_start_date'], axis=1, inplace=True)
all_labs = all_labs.merge(df_pop[['person_id', 'cohort_start_date']], how='left', left_on = 'person_id', right_on = 'person_id')
all_labs.to_csv('psychosis_prediction/temporal_labs.csv')