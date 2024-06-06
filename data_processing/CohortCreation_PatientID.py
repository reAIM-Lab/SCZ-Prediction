import numpy as np
import os
import pandas as pd
import pyodbc
import time
from datetime import datetime
import sys
import gc

sys.path.append('../utils')
from processing import identify_exclusive_mappings


connection_string = ("YOUR CREDENTIALS HERE")
conn = pyodbc.connect(connection_string)

#### ICD codes and their corresponding concept_ids
colnames = ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id', 'concept_class_id', 'standard_concept', 'concept_code', 'valid_start_date', 'valid_end_date', 'invalid_reason']
psychosis_codeset = pd.read_csv('codes_mappings/psychosis_concepts.csv', names = colnames)
# filter out schizophreniform disorder. Note that 295 is not included at all 
psychosis_codeset = psychosis_codeset.loc[psychosis_codeset['concept_code'] != 'F20.81']


    
exclusive_codes, nonexclusive_codes, snomed_to_icd, icd_to_snomed = identify_exclusive_mappings(psychosis_codeset['concept_code'])
exclusive_concept_codes = set(icd_to_snomed.loc[icd_to_snomed['snomed_concept_name'].isin(exclusive_codes), 'concept_code'])

# get schziophrenia codes
all_scz_codes_query = ("SELECT c_new.concept_id as standard_concept_id, c_icd10.concept_code as icd_code, c_new.concept_name as standard_concept_name, c_icd10.concept_name as icd_name FROM dbo.concept as c_icd10 LEFT JOIN dbo.concept_relationship as rel on rel.concept_id_1 = c_icd10.concept_id "+
                "LEFT JOIN dbo.concept as c_rel on rel.concept_id_2 = c_rel.concept_id "+
                "LEFT JOIN dbo.concept_ancestor as ca ON ca.ancestor_concept_id = rel.concept_id_2 "+
                "LEFT JOIN dbo.concept as c_new on c_new.concept_id = ca.descendant_concept_id " +
                "WHERE (rel.relationship_id = 'Maps to' AND c_new.standard_concept = 'S') "+
                "AND ((c_icd10.concept_code LIKE '295%' AND c_icd10.vocabulary_id = 'ICD9CM') "+
                "OR ((c_icd10.concept_code LIKE 'F2[05]%' AND c_icd10.vocabulary_id = 'ICD10CM')))")

all_scz_codes = pd.io.sql.read_sql(all_scz_codes_query, conn)
all_scz_codes = all_scz_codes.loc[~(all_scz_codes['standard_concept_id'].isin([444434, 4184004, 4263364]))]
all_scz_codes.to_csv('codes_mappings/all_scz_codes.csv')

# remove codes that are related to schizophrenia
exclusive_concept_ids = list(icd_to_snomed.loc[icd_to_snomed['snomed_concept_name'].isin(exclusive_codes), 'snomed_mapping_concept'].unique())
psychosis_codes = list(set(exclusive_concept_ids).difference(all_scz_codes['standard_concept_id']))

"""
# Create Cohort
1. (In DataGrip) Get all the patients with SCZ/schizoaffective disorder and 7 years of prior observation and save them to the ak4885_schizophrenia_incidence table
2. (In DataGrip) Get all patients who have an episode of psychosis (incl. schizophrenia) and 7 years of observation overall -- save this into results as ak4885_psychosis_cohort
3. Pull all people with SCZ and make sure that a. they have an accurate cohort start date; b. they have a valid psychosis diagnosis prior; c. they have no history of schizophreniform disorder
4. Pull all people with psychosis and make sure a. they have no history of schziophrenia; b. they have a valid psychosis diagnosis (with an accurate psychosis diagnosis date); c. they have no history of schizophreniform disorder
5. Make sure everyone is between 10 and 35 years at "cohort start" (SCZ diagnosis or observation period end date)
6. Make sure that the first observed schizophrenia is after psychosis
7. Combine dataframes (SCZ and No SCZ) and add SCZ "flag"
8. Get all conditions in 7 years prior to cohort start for both of the above tables
"""

# 1, 2. get everyone from super broad schizophrenia and psychosis cohorts
df_psychosis_all = pd.io.sql.read_sql("SELECT pc.*, year_of_birth, race_concept_id, gender_concept_id FROM results.ak4885_psychosis_cohort as pc LEFT JOIN dbo.person as p ON p.person_id = pc.person_id", conn)
df_scz_all = pd.io.sql.read_sql("SELECT sc.*, year_of_birth, race_concept_id, gender_concept_id FROM results.ak4885_schizophrenia_cohort as sc LEFT JOIN dbo.person as p ON p.person_id = sc.person_id", conn)
df_scz_all = df_scz_all.merge(df_psychosis_all[['person_id', 'psychosis_dx_date']], how='left', left_on = 'person_id', right_on = 'person_id')
if df_scz_all.isna().sum().sum() > 0:
    print('Undefined psychosis diagnosis date after merge')
df_psychosis_all = df_psychosis_all.loc[~df_psychosis_all['person_id'].isin(list(df_scz_all['person_id']))]
print(len(df_psychosis_all), len(df_scz_all), len(df_scz_all)*100/(len(df_scz_all)+len(df_psychosis_all)))

# limit schizophrenia cohort to people with at least m diagnoses of schizophrenia (n=1)
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
print(len(df_scz_all))

# get all conditions for people in the schizophrenia cohort
all_conds_scz = pd.io.sql.read_sql ("SELECT DISTINCT sc.person_id, condition_start_date, condition_concept_id FROM results.ak4885_schizophrenia_cohort as sc LEFT JOIN dbo.condition_occurrence as co ON co.person_id = sc.person_id", conn) 
## 3a. Make sure that there is an appropriate cohort_start_date
scz_conds_scz = all_conds_scz.loc[all_conds_scz['condition_concept_id'].isin(all_scz_codes['standard_concept_id'])]
scz_conds_scz['condition_start_date'] = pd.to_datetime(scz_conds_scz['condition_start_date'], format='%Y-%m-%d')

scz_diagnosis_date = scz_conds_scz.groupby('person_id').min()['condition_start_date']
scz_diagnosis_date.name = "scz_diagnosis_date"
scz_conds_scz = scz_conds_scz.merge(scz_diagnosis_date, left_on='person_id', right_index=True)
scz_conds_scz = scz_conds_scz.loc[scz_conds_scz['scz_diagnosis_date']==scz_conds_scz['condition_start_date']]

df_scz_all = df_scz_all.merge(scz_conds_scz[['person_id', 'scz_diagnosis_date']].drop_duplicates(), how='inner', left_on = 'person_id', right_on='person_id')
df_scz_all['cohort_start_date'] = df_scz_all['scz_diagnosis_date']

## 3b. make sure that there is at least one "acceptable" psychosis code preceeding SCZ dx
##(also store the initial psychosis code, accurate psychosis dx date)
psych_conds_scz = all_conds_scz.loc[all_conds_scz['condition_concept_id'].isin(psychosis_codes)]
psych_diagnosis_date = psych_conds_scz.groupby('person_id').min()['condition_start_date']
psych_diagnosis_date.name = "psychosis_diagnosis_date"
psych_conds_scz = psych_conds_scz.merge(psych_diagnosis_date, left_on='person_id', right_index=True)
psych_conds_scz = psych_conds_scz.loc[psych_conds_scz['psychosis_diagnosis_date']==psych_conds_scz['condition_start_date']]
df_scz_all = df_scz_all.merge(psych_conds_scz[['person_id', 'psychosis_diagnosis_date']], how='inner', left_on='person_id', right_on = 'person_id')
df_scz_all = df_scz_all[['cohort_definition_id', 'person_id', 
                         'cohort_start_date', 'end_date', 
                         'year_of_birth', 'race_concept_id', 
                         'gender_concept_id', 'scz_diagnosis_date', 
                         'psychosis_diagnosis_date']].drop_duplicates()
scz_initial_psychosis = psych_conds_scz[['person_id', 'condition_concept_id', 'psychosis_diagnosis_date']]
print(len(df_scz_all))

## 3c. Remove anyone with a history of schizophreniform disorder
schizophreniform_scz = all_conds_scz.loc[all_conds_scz['condition_concept_id'].isin([444434, 4184004, 4263364])]
df_scz_all = df_scz_all.loc[~(df_scz_all['person_id'].isin(schizophreniform_scz['person_id']))]

# get all conditions for people in the psychosis cohort
psych_conds_query = "SELECT DISTINCT pc.person_id, condition_start_date, condition_concept_id FROM results.ak4885_psychosis_cohort as pc LEFT JOIN dbo.condition_occurrence as co ON co.person_id = pc.person_id WHERE condition_concept_id > 0"
list_chunks = []
for chunk in pd.io.sql.read_sql(psych_conds_query, conn, chunksize=500000):
    list_chunks.append(chunk.loc[chunk['person_id'].isin(df_psychosis_all['person_id'])])
all_conds_psych = pd.concat(list_chunks)

##4a. Make sure that there are no conditions of scz or schizophreniform
scz_conds_psych = all_conds_psych.loc[all_conds_psych['condition_concept_id'].isin(all_scz_codes['standard_concept_id'])]
df_psychosis_all = df_psychosis_all.loc[~(df_psychosis_all['person_id'].isin(scz_conds_psych['person_id']))]

## 4b. make sure that there is at least one "acceptable" psychosis code 
##(also store the initial psychosis code, accurate psychosis dx date)
psych_conds_psych = all_conds_psych.loc[all_conds_psych['condition_concept_id'].isin(psychosis_codes)]
psych_diagnosis_date = psych_conds_psych.groupby('person_id').min()['condition_start_date']
psych_diagnosis_date.name = "psychosis_diagnosis_date"
psych_conds_psych = psych_conds_psych.merge(psych_diagnosis_date, left_on='person_id', right_index=True)
psych_conds_psych = psych_conds_psych.loc[psych_conds_psych['psychosis_diagnosis_date']==psych_conds_psych['condition_start_date']]
df_psychosis_all = df_psychosis_all.merge(psych_conds_psych[['person_id', 'psychosis_diagnosis_date']], how='inner', left_on='person_id', right_on = 'person_id')
df_psychosis_all['scz_diagnosis_date'] = np.nan
df_psychosis_all['cohort_start_date'] = df_psychosis_all['end_date']
df_psychosis_all = df_psychosis_all[['cohort_definition_id', 'person_id', 
                         'cohort_start_date', 'end_date', 
                         'year_of_birth', 'race_concept_id', 
                         'gender_concept_id', 'scz_diagnosis_date', 
                         'psychosis_diagnosis_date']].drop_duplicates()
psych_initial_psychosis = psych_conds_psych[['person_id', 'condition_concept_id', 'psychosis_diagnosis_date']]
psych_initial_psychosis = psych_initial_psychosis.loc[psych_initial_psychosis['person_id'].isin(df_psychosis_all['person_id'])]
all_conds_psych = all_conds_psych.loc[all_conds_psych['person_id'].isin(df_psychosis_all['person_id'])]

## 4c. remove anyone with a history of schizophreniform disorder
schizophreniform_psych = all_conds_psych.loc[all_conds_psych['condition_concept_id'].isin([444434, 4184004, 4263364])]
df_psychosis_all = df_psychosis_all.loc[~(df_psychosis_all['person_id'].isin(schizophreniform_psych['person_id']))]

all_conds_psych = all_conds_psych.loc[all_conds_psych['person_id'].isin(df_psychosis_all['person_id'])]

print(len(df_psychosis_all))

# 5. AGE AT COHORT INDEX
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

all_conds_psych = all_conds_psych.loc[all_conds_psych['person_id'].isin(df_psychosis_all['person_id'])]
all_conds_scz = all_conds_scz.loc[all_conds_scz['person_id'].isin(df_scz_all['person_id'])]

# 6. First SCZ Diagnosis is AFTER first psychosis diagnosis
df_scz_all['psychosis_diagnosis_date'] = pd.to_datetime(df_scz_all['psychosis_diagnosis_date'])
df_scz_all = df_scz_all.loc[df_scz_all['cohort_start_date']>df_scz_all['psychosis_diagnosis_date']]
print(len(df_psychosis_all), len(df_scz_all), len(df_scz_all)*100/(len(df_scz_all)+len(df_psychosis_all)))

# 7. combine data
all_conds_scz = all_conds_scz.loc[all_conds_scz['person_id'].isin(df_scz_all['person_id'])]
all_conds_scz['sz_flag'] = 1
all_conds_psych['sz_flag'] = 0

all_conds = pd.concat([all_conds_scz, all_conds_psych])

df_psychosis_all['sz_flag'] = 0
df_scz_all['sz_flag'] = 1
df_pop = pd.concat([df_psychosis_all, df_scz_all])
print(len(df_pop), sum(df_pop['sz_flag'])*100/len(df_pop))

all_conds = all_conds.merge(df_pop[['person_id','cohort_start_date']], how='inner', left_on = 'person_id', right_on = 'person_id')

# 8. Restrict all conds to 7 years before index to get continuous care
all_conds['condition_start_date'] = pd.to_datetime(all_conds['condition_start_date'], format = '%Y-%m-%d')
all_conds['cohort_start_date'] = pd.to_datetime(all_conds['cohort_start_date'], format = '%Y-%m-%d')

all_conds = all_conds.loc[(all_conds['cohort_start_date']-all_conds['condition_start_date']).dt.days<=2555]

"""
# Constrict cohort based on continuous care
### First drop instances of conditions where the condition concept id is not defined to ensure that there is at least 1 service contact per year

Calculate the differences between consecutive condition occurrences for each patient -- do this by making sure that:
1. there are at least 7 unique dates that there is a visit 
2. there's at least one visit > 6 years before diagnosis
3. the max difference between consecutive dates is 1 year (inclusive)
"""
# drop undefined conditions
conds_dates = all_conds[['person_id', 'condition_start_date', 'cohort_start_date']].drop_duplicates()
print('done datetime conversions')

# at least 7 unique dates for visits
conds_patients = conds_dates.groupby('person_id').count()
yearly_service_pids = list(conds_patients.loc[conds_patients['condition_start_date'] >= 7].index)
print('done getting at least 7 unique dates for visits')

# at least 1 visit > 6 years before diagnosis
conds_dates = conds_dates.loc[conds_dates['person_id'].isin(yearly_service_pids)]
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

df_pop = df_pop.loc[df_pop['person_id'].isin(yearly_service_pids)]
print(len(df_pop), sum(df_pop['sz_flag'])*100/len(df_pop))

"""
## Maximum of 45 days with no insurance coverage
- Get insurance information from all people with at least 7 years observation and then limit to only the people in with at least 1 service visit per year
- Combine all of the overlapping payer periods, with a grace period of 45 days between coverage periods
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

df_pop['cohort_start_date'] =  pd.to_datetime(df_pop['cohort_start_date'], format='%Y-%m-%d')

insurance_check_df = df_pop.merge(merged_insurance_df, how = 'left', left_on = 'person_id', right_on = 'PERSON_ID')
eligible_pids = insurance_check_df.loc[(insurance_check_df['PAYER_PLAN_PERIOD_END_DATE']>=insurance_check_df['cohort_start_date'])&(insurance_check_df['PAYER_PLAN_PERIOD_START_DATE'] <= insurance_check_df['cohort_start_date']- pd.Timedelta(days=2555))]['person_id'].unique()

print(len(eligible_pids))
eligible_pids = list(eligible_pids)
df_pop = df_pop.loc[df_pop['person_id'].isin(eligible_pids)]
print(len(df_pop), sum(df_pop['sz_flag'])*100/len(df_pop))

# table 1
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
table1 = t1_counts.merge(t1_percents, how='inner', left_index=True, right_index=True, suffixes = [' (n)', ' (%)'])
table1.to_csv('prediction_data/demographic_breakdown.csv')
df_pop.to_csv('prediction_data/population.csv')