import numpy as np
import os
import pandas as pd
import pyodbc
import time
from datetime import datetime
import sys
import gc
import pickle
from itertools import product

sys.path.append('../utils')
from processing import drop_rare_occurrences, generate_code_list, make_static_df

connection_string = ("YOUR CREDENTIALS HERE")
conn = pyodbc.connect(connection_string)

data_path = '../prediction_data/'
create_data_path = 'stored_data/'

# DATASET CREATION: Import the population dataframe and constrict to the correct set of patients
num_days_prediction = 90
df_pop = pd.read_csv(data_path+'population.csv')
df_pop.rename({'psychosis_dx_date':'psychosis_diagnosis_date'}, axis=1, inplace=True)
df_pop['psychosis_diagnosis_date'] = pd.to_datetime(df_pop['psychosis_diagnosis_date'], format="%Y-%m-%d")
df_pop['cohort_start_date'] = pd.to_datetime(df_pop['cohort_start_date'])
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction]

all_visits = pd.read_csv(data_path+'temporal_visits.csv')
df_pop = df_pop.merge(all_visits.groupby('person_id').min()['visit_start_date'], how='left', left_on='person_id',right_index=True)
df_pop.rename({'visit_start_date':'first_visit'}, axis=1, inplace=True)
df_pop.head()

# Import rest of temporal data
all_conds = pd.read_csv(data_path+'temporal_conditions.csv')
all_meds = pd.read_csv(data_path+'temporal_medications.csv')
all_procedures = pd.read_csv(data_path+'temporal_procedures.csv')
all_labs = pd.read_csv(data_path+'temporal_labs.csv')

# Restrict to appropriate time periods
all_meds = all_meds.loc[all_meds['person_id'].isin(df_pop['person_id'])]
all_meds['cohort_start_date'] = pd.to_datetime(all_meds['cohort_start_date'])
all_meds['drug_era_start_date'] = pd.to_datetime(all_meds['drug_era_start_date'])
all_meds['drug_era_end_date'] = pd.to_datetime(all_meds['drug_era_end_date'])
all_meds = all_meds.loc[(all_meds['cohort_start_date']-all_meds['drug_era_end_date']).dt.days >= num_days_prediction]
all_meds['days_to_cohort_start'] = (all_meds['cohort_start_date']-all_meds['drug_era_start_date']).dt.days

all_visits = all_visits.loc[all_visits['person_id'].isin(df_pop['person_id'])]
all_visits['cohort_start_date'] = pd.to_datetime(all_visits['cohort_start_date'])
all_visits['visit_start_date'] = pd.to_datetime(all_visits['visit_start_date'])
all_visits['visit_end_date'] = pd.to_datetime(all_visits['visit_end_date'])
all_visits = all_visits.loc[(all_visits['cohort_start_date']-all_visits['visit_end_date']).dt.days >= num_days_prediction]
all_visits['days_to_cohort_start'] = (all_visits['cohort_start_date']-all_visits['visit_start_date']).dt.days

all_conds = all_conds.loc[all_conds['person_id'].isin(df_pop['person_id'])]
all_conds['cohort_start_date'] = pd.to_datetime(all_conds['cohort_start_date'])
all_conds['condition_start_date'] = pd.to_datetime(all_conds['condition_start_date'])
all_conds['days_to_cohort_start'] = (all_conds['cohort_start_date']-all_conds['condition_start_date']).dt.days
all_conds = all_conds.loc[all_conds['days_to_cohort_start'] >= num_days_prediction]

all_procedures = all_procedures.loc[all_procedures['person_id'].isin(df_pop['person_id'])]
all_procedures['cohort_start_date'] = pd.to_datetime(all_procedures['cohort_start_date'])
all_procedures['procedure_date'] = pd.to_datetime(all_procedures['procedure_date'])
all_procedures['days_to_cohort_start'] = (all_procedures['cohort_start_date']-all_procedures['procedure_date']).dt.days
all_procedures = all_procedures.loc[all_procedures['days_to_cohort_start'] >= num_days_prediction]

all_labs = all_labs.loc[all_labs['person_id'].isin(df_pop['person_id'])]
all_labs['cohort_start_date'] = pd.to_datetime(all_labs['cohort_start_date'])
all_labs['measurement_date'] = pd.to_datetime(all_labs['measurement_date'])
all_labs['days_to_cohort_start'] = (all_labs['cohort_start_date']-all_labs['measurement_date']).dt.days
all_labs = all_labs.loc[all_labs['days_to_cohort_start'] >= num_days_prediction]

all_labs['concept_name'].replace({'Methadone':'Methadone_Lab'}, inplace=True)
all_procedures['concept_name'].replace({'Methadone':'Methadone_Procedure'}, inplace=True)

# delete rare occurrences (e.g. any concept id that does not appear at least once for 1% of patients)
all_conds = drop_rare_occurrences(all_conds, 'condition_concept_id')
all_meds = drop_rare_occurrences(all_meds, 'drug_concept_id')
all_procedures = drop_rare_occurrences(all_procedures, 'procedure_concept_id')
all_labs = drop_rare_occurrences(all_labs, 'measurement_concept_id')
all_visits = drop_rare_occurrences(all_visits, 'visit_concept_id')

# Check that the minimum time between cohort start date and start/end dates for healthcare services is over 90 days
check = (all_labs['cohort_start_date']-all_labs['measurement_date']).dt.days
print('Labs:', check.min(), check.max())

check = (all_procedures['cohort_start_date']-all_procedures['procedure_date']).dt.days
print('Procedures:', check.min(), check.max())

check = (all_conds['cohort_start_date']-all_conds['condition_start_date']).dt.days
print('Conditions:', check.min(), check.max())

check = (all_meds['cohort_start_date']-all_meds['drug_era_start_date']).dt.days
print('Meds (Start of prescription):', check.min(), check.max())
check = (all_meds['cohort_start_date']-all_meds['drug_era_end_date']).dt.days
print('Meds (End of prescription):', check.min(), check.max())

check = (all_visits['cohort_start_date']-all_visits['visit_start_date']).dt.days
print('Visits (Start of visit):', check.min(), check.max())
check = (all_visits['cohort_start_date']-all_visits['visit_end_date']).dt.days
print('Visits (End of visit):', check.min(), check.max())

print('Check presence of SCZ:',len(all_conds.loc[all_conds['concept_name'].isin(['Schizophrenia', 'Paranoid schizophrenia'])]))

check_cohort_start = df_pop[['person_id','cohort_start_date']]
check_cohort_start = check_cohort_start.merge(all_conds[['person_id','cohort_start_date']].drop_duplicates(),how='left', left_on='person_id', right_on='person_id', suffixes=['_pop','_cond'])
check_cohort_start = check_cohort_start.merge(all_visits[['person_id','cohort_start_date']].drop_duplicates(),how='left', left_on='person_id', right_on='person_id', suffixes = ['_old1','_visits'])
check_cohort_start = check_cohort_start.merge(all_procedures[['person_id','cohort_start_date']].drop_duplicates(),how='left', left_on='person_id', right_on='person_id', suffixes=['_old2','_pro'])
check_cohort_start = check_cohort_start.merge(all_labs[['person_id','cohort_start_date']].drop_duplicates(),how='left', left_on='person_id', right_on='person_id', suffixes=['_old3','_labs'])
check_cohort_start = check_cohort_start.merge(all_meds[['person_id','cohort_start_date']].drop_duplicates(),how='left', left_on='person_id', right_on='person_id', suffixes=['_old4','_meds'])
check_cohort_start.set_index('person_id',inplace=True)
check_cohort_start = check_cohort_start.T
num_unique = check_cohort_start.T.apply(lambda x: x.nunique(), axis=1)
print('Number of places where cohort start date doesnt align:',(num_unique>1).sum())

# check for schizophrenia codes, schizophreniform codes
scz_codes = list(pd.read_csv('../codes_mappings/all_scz_codes.csv')['standard_concept_id'].unique())
print('Number of SCZ instances:', len(all_conds.loc[all_conds['condition_concept_id'].isin(scz_codes)]))
print('Number of schizophreniform instances:', len(all_conds.loc[all_conds['condition_concept_id'].isin([444434, 4184004, 4263364])]))

# SQL Queries for grouping conditions and medications
# conditions mapping
conditions_mapping_query = ("SELECT c_icd10.concept_id as icd10_ancestor_concept_id,c_icd10.concept_name as icd10_ancestor_concept_name, c_icd10.concept_code as icd10_code, rel.concept_id_2 as standard_ancestor_concept_id, c_rel.concept_name as standard_ancestor_concept_name, ca.descendant_concept_id as standard_descendant_concept_id, c_new.concept_name as standard_descendant_concept_name, c_new.standard_concept as check_standard "+
    "FROM cdm_mdcd.dbo.concept as c_icd10 "+
    "LEFT JOIN cdm_mdcd.dbo.concept_relationship as rel on rel.concept_id_1 = c_icd10.concept_id "+
    "LEFT JOIN cdm_mdcd.dbo.concept as c_rel on rel.concept_id_2 = c_rel.concept_id "+
    "LEFT JOIN cdm_mdcd.dbo.concept_ancestor as ca ON ca.ancestor_concept_id = rel.concept_id_2 "+
    "LEFT JOIN cdm_mdcd.dbo.concept as c_new on c_new.concept_id = ca.descendant_concept_id "+
    "WHERE c_icd10.concept_class_id = '3-char nonbill code' and c_icd10.vocabulary_id = 'ICD10CM' "+
    "AND rel.relationship_id = 'Maps to' AND c_rel.standard_concept = 'S'")
conditions_mapping = pd.io.sql.read_sql(conditions_mapping_query, conn)

conditions_mapping = conditions_mapping.loc[conditions_mapping['standard_descendant_concept_id'].isin(list(all_conds['condition_concept_id'].unique()))]

# medications mapping 
medications_mapping_query = ("SELECT c_atc.concept_id as atc_concept_id, c_atc.concept_name as atc_concept_name, c_standard.concept_id as standard_concept_id, c_standard.concept_name as standard_concept_name "+
                             "FROM cdm_mdcd.dbo.concept as c_atc "+
                             "LEFT JOIN cdm_mdcd.dbo.concept_ancestor as ca on ancestor_concept_id=c_atc.concept_id "+
                             "LEFT JOIN cdm_mdcd.dbo.concept as c_standard on c_standard.concept_id = descendant_concept_id "+
                             "WHERE c_atc.concept_class_id = 'ATC 3rd' AND c_standard.standard_concept = 'S'")

medications_mapping = pd.io.sql.read_sql(medications_mapping_query, conn)
medications_mapping = medications_mapping.loc[medications_mapping['standard_concept_id'].isin(list(all_meds['drug_concept_id'].unique()))]

# medications mapping: move Lithium to the antiepileptics category
lithium_list = generate_code_list('Lithium', 'ATC 4th')
medications_mapping.loc[(medications_mapping['standard_concept_id'].isin(lithium_list))&(medications_mapping['atc_concept_name']=='ANTIPSYCHOTICS'), 'atc_concept_name'] = 'ANTIEPILEPTICS'
medications_mapping['atc_concept_name'].replace({'ANTIEPILEPTICS': 'MOOD STABILIZERS'}, inplace=True)

# INPATIENT PSYCH VISITS
query = ("SELECT vo.person_id, vo.visit_occurrence_id, vo.visit_concept_id, co.condition_start_date, vo.visit_start_date, vo.visit_end_date, co.condition_concept_id, c.concept_name as condition_name, p.race_concept_id, p.gender_concept_id "+
         "FROM cdm_mdcd.dbo.visit_occurrence as vo LEFT JOIN dbo.condition_occurrence as co on co.visit_occurrence_id = vo.visit_occurrence_id "+
         "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = co.condition_concept_id "+
         "LEFT JOIN cdm_mdcd.dbo.person as p on p.person_id = vo.person_id "+
         "WHERE vo.visit_concept_id = 9201 AND condition_concept_id IN "+
         "(SELECT DISTINCT concept_id_2 FROM cdm_mdcd.dbo.concept as c LEFT JOIN cdm_mdcd.dbo.concept_relationship on concept_id_1 = concept_id WHERE c.concept_code LIKE 'F%' AND c.vocabulary_id = 'ICD10CM' AND relationship_id = 'Maps to')")

psych_hosp = pd.io.sql.read_sql(query, conn)
list_psych_visits = list(psych_hosp['visit_occurrence_id'].unique())

"""
# Create dataframe that identifies the iteration, first visit (start date), last visit (start date), and years of observation

We do this by ordering all the visits within each patient and then choosing every 4th visit
Then manually want to add back the psychosis date as the first date
"""
sorted_visits = all_visits.groupby('person_id').apply(pd.DataFrame.sort_values, ['visit_start_date'])
sorted_visits.reset_index(drop=True, inplace=True)
sorted_visits = sorted_visits.merge(df_pop[['person_id', 'first_visit']], how = 'left', left_on = 'person_id', right_on = 'person_id')
sorted_visits = sorted_visits[['person_id', 'psychosis_diagnosis_date', 'first_visit', 'visit_start_date', 'cohort_start_date']].drop_duplicates()
sorted_visits = sorted_visits.loc[(sorted_visits['visit_start_date']>sorted_visits['psychosis_diagnosis_date'])]
nth_visit = np.asarray(sorted_visits.groupby('person_id').cumcount())
sorted_visits['nth_visit'] = nth_visit
sorted_visits = sorted_visits.loc[sorted_visits['nth_visit']%8==0]
sorted_visits['nth_visit'] = sorted_visits['nth_visit'] // 8
sorted_visits['nth_visit'] += 1
add_psychosis_date = df_pop[['person_id', 'psychosis_diagnosis_date', 'first_visit', 'cohort_start_date']]
add_psychosis_date['visit_start_date'] = add_psychosis_date['psychosis_diagnosis_date']
add_psychosis_date['nth_visit'] = 0
df_iter_pop = pd.concat([sorted_visits, add_psychosis_date])

df_iter_pop.rename({'nth_visit':'iteration', 'visit_start_date':'cutoff_date'}, axis=1, inplace=True)

df_iter_pop['cohort_start_date'] = pd.to_datetime(df_iter_pop['cohort_start_date'])
df_iter_pop['cutoff_date'] = pd.to_datetime(df_iter_pop['cutoff_date'])
df_iter_pop['first_visit'] = pd.to_datetime(df_iter_pop['first_visit'])

df_iter_pop['years_obs'] = (df_iter_pop['cutoff_date']-df_iter_pop['first_visit']).dt.days/365

df_iter_pop['censor_date'] = df_iter_pop['cohort_start_date']-pd.Timedelta(90, 'days') 
df_iter_pop = df_iter_pop.loc[df_iter_pop['cutoff_date']<=df_iter_pop['censor_date']]
df_iter_pop.to_csv('stored_data/iterated_population_8_visits_5_21.csv')

# Loop through iteration to get features for each step
list_feature_dfs = []

for iteration in tqdm(range(0,df_iter_pop['iteration'].max())): 
    # only need iter0 if you change stuff above
    temp_df_iter_pop = df_iter_pop.loc[(df_iter_pop['iteration'] == iteration)&df_iter_pop['years_obs']>0.5]

    # for conditions, labs, procedures, just compare the start_date to the cutoff date
    temp_conds = all_conds.loc[all_conds['person_id'].isin(temp_df_iter_pop['person_id'])]
    temp_conds = temp_conds.merge(temp_df_iter_pop[['person_id','cutoff_date']], how = 'left', left_on = 'person_id', right_on = 'person_id')
    temp_conds = temp_conds.loc[temp_conds['condition_start_date']< temp_conds['cutoff_date']]

    temp_labs = all_labs.loc[all_labs['person_id'].isin(temp_df_iter_pop['person_id'])]
    temp_labs = temp_labs.merge(temp_df_iter_pop[['person_id','cutoff_date']], how = 'left', left_on = 'person_id', right_on = 'person_id')
    temp_labs = temp_labs.loc[temp_labs['measurement_date']< temp_labs['cutoff_date']]

    temp_procedures = all_procedures.loc[all_procedures['person_id'].isin(temp_df_iter_pop['person_id'])]
    temp_procedures = temp_procedures.merge(temp_df_iter_pop[['person_id','cutoff_date']], how = 'left', left_on = 'person_id', right_on = 'person_id')
    temp_procedures = temp_procedures.loc[temp_procedures['procedure_date']< temp_procedures['cutoff_date']]

    """
    for medications and visits, we want to look at 
    1. limit to visit/medication start dates prior to cutoff date
    2. if cutoff date is prior to visit/medication end date, make the cutoff date the new end date
    """
    temp_meds = all_meds.loc[all_meds['person_id'].isin(temp_df_iter_pop['person_id'])]
    temp_meds = temp_meds.merge(temp_df_iter_pop[['person_id','cutoff_date']], how = 'left', left_on = 'person_id', right_on = 'person_id')
    temp_meds = temp_meds.loc[temp_meds['drug_era_start_date']< temp_meds['cutoff_date']]
    temp_meds.loc[temp_meds['drug_era_end_date']>temp_meds['cutoff_date'], 'drug_era_end_date']=temp_meds.loc[temp_meds['drug_era_end_date']>temp_meds['cutoff_date'], 'cutoff_date']

    temp_visits = all_visits.loc[all_visits['person_id'].isin(temp_df_iter_pop['person_id'])]
    temp_visits = temp_visits.merge(temp_df_iter_pop[['person_id','cutoff_date']], how = 'left', left_on = 'person_id', right_on = 'person_id')
    temp_visits = temp_visits.loc[temp_visits['visit_start_date']< temp_visits['cutoff_date']]
    temp_visits.loc[temp_visits['visit_end_date']>temp_visits['cutoff_date'], 'visit_end_date']=temp_visits.loc[temp_visits['visit_end_date']>temp_visits['cutoff_date'], 'cutoff_date']
    
    if min((temp_conds['cutoff_date']-temp_conds['condition_start_date']).dt.days) < 0:
        print('Leakage in conds')        
    if min((temp_labs['cutoff_date']-temp_labs['measurement_date']).dt.days) < 0:
        print('Leakage in labs')
    if min((temp_procedures['cutoff_date']-temp_procedures['procedure_date']).dt.days) < 0:
        print('Leakage in procedures')
    if min((temp_visits['cutoff_date']-temp_visits['visit_start_date']).dt.days) < 0:
        print('Leakage in visit starts')
    if min((temp_visits['cutoff_date']-temp_visits['visit_end_date']).dt.days) < 0:
        print('Leakage in visit ends')
    if min((temp_meds['cutoff_date']-temp_meds['drug_era_start_date']).dt.days) < 0:
        print('Leakage in med starts')
    if min((temp_meds['cutoff_date']-temp_meds['drug_era_end_date']).dt.days) < 0:
        print('Leakage in med ends')

    all_features = make_static_df(temp_df_iter_pop, temp_conds, temp_meds, temp_visits, temp_procedures, temp_labs)
    all_features['iteration'] = iteration
    
    list_feature_dfs.append(all_features)

df_all_iters = pd.concat(list_feature_dfs)
df_all_iters.drop(["('los max', 9203)","('los mean', 9203)", "('los min', 9203)", "('los sum', 9203)"], axis=1, inplace=True)
df_all_iters.to_csv('stored_data/xgboost_all_iters_8_visits_5_21.csv')
print(len(df_all_iters), len(df_all_iters.columns))

save_cols = list(df_all_iters.columns)
with open("stored_data/df_all_iters_columns_8_visits_5_21", "wb") as fp:   #Pickling
    pickle.dump(save_cols, fp)