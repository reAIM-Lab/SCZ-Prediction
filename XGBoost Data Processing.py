import numpy as np
import os
import pandas as pd
import pyodbc
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from datetime import datetime
import sys
import gc
import pickle
import joblib
from itertools import product

connection_string = "YOUR CONNECTION STRING HERE"
conn = pyodbc.connect(connection_string)

path = '../'

# read in the population 
num_days_prediction = 90
df_pop = pd.read_csv(path+'population.csv')
df_pop.rename({'psychosis_dx_date':'psychosis_diagnosis_date'}, axis=1, inplace=True)
df_pop['psychosis_diagnosis_date'] = pd.to_datetime(df_pop['psychosis_diagnosis_date'], format="%Y-%m-%d")
df_pop['cohort_start_date'] = pd.to_datetime(df_pop['cohort_start_date'])
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction]

# get the first visit
all_visits = pd.read_csv(path+'temporal_visits.csv')
df_pop = df_pop.merge(all_visits.groupby('person_id').min()['visit_start_date'], how='left', left_on='person_id',right_index=True)
df_pop.rename({'visit_start_date':'first_visit'}, axis=1, inplace=True)

# GET TABLE 1
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

# import rest of temporal data
all_conds = pd.read_csv(path+'temporal_conditions.csv')
all_meds = pd.read_csv(path+'temporal_medications.csv')
all_procedures = pd.read_csv(path+'temporal_procedures.csv')
all_labs = pd.read_csv(path+'temporal_labs.csv')

# restrict data to appropriate time persiods
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

# these have the same featue name so change that
all_labs['concept_name'].replace({'Methadone':'Methadone_Lab'}, inplace=True)
all_procedures['concept_name'].replace({'Methadone':'Methadone_Procedure'}, inplace=True)

# delete rare occurrences
def drop_rare_occurrences(df, col_concept, col_id = 'person_id', size_pop = len(df_pop)):
    unique_occurrences = df[['person_id', col_concept]].drop_duplicates()
    unique_occurrences = unique_occurrences.value_counts(col_concept)
    common_occurrences = unique_occurrences[unique_occurrences/size_pop > 0.01].index
    return df.loc[df[col_concept].isin(common_occurrences)]
all_conds = drop_rare_occurrences(all_conds, 'condition_concept_id')
all_meds = drop_rare_occurrences(all_meds, 'drug_concept_id')
all_procedures = drop_rare_occurrences(all_procedures, 'procedure_concept_id')
all_labs = drop_rare_occurrences(all_labs, 'measurement_concept_id')
all_visits = drop_rare_occurrences(all_visits, 'visit_concept_id')

# CHECK FOR LEAKAGE IN DATA
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

# SQL Queries for grouping medications and conditions
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

def generate_code_list(drugtype, concept_class):
    sql_query = ("SELECT ancestor_concept_id, descendant_concept_id, concept_name " + 
               "FROM cdm_mdcd.dbo.concept_ancestor JOIN cdm_mdcd.dbo.concept ON descendant_concept_id = concept_id "+
               "WHERE ancestor_concept_id = (SELECT concept_id from cdm_mdcd.dbo.concept WHERE concept_class_id = '"+concept_class+"' AND concept_name = '"+drugtype+"');")
    codes_list = pd.read_sql(sql_query, conn)
    return list(codes_list['descendant_concept_id'])
# move lithium to mood stabilizers -- per psychiatrist advice
lithium_list = generate_code_list('Lithium', 'ATC 4th')
medications_mapping.loc[(medications_mapping['standard_concept_id'].isin(lithium_list))&(medications_mapping['atc_concept_name']=='ANTIPSYCHOTICS'), 'atc_concept_name'] = 'ANTIEPILEPTICS'
medications_mapping['atc_concept_name'].replace({'ANTIEPILEPTICS': 'MOOD STABILIZERS'}, inplace=True)

# get inpatient visits associated with a psychiatric condition (F-diagnosis under ICD10)
query = ("SELECT vo.person_id, vo.visit_occurrence_id, vo.visit_concept_id, co.condition_start_date, vo.visit_start_date, vo.visit_end_date, co.condition_concept_id, c.concept_name as condition_name, p.race_concept_id, p.gender_concept_id "+
         "FROM cdm_mdcd.dbo.visit_occurrence as vo LEFT JOIN dbo.condition_occurrence as co on co.visit_occurrence_id = vo.visit_occurrence_id "+
         "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = co.condition_concept_id "+
         "LEFT JOIN cdm_mdcd.dbo.person as p on p.person_id = vo.person_id "+
         "WHERE vo.visit_concept_id = 9201 AND condition_concept_id IN "+
         "(SELECT DISTINCT concept_id_2 FROM cdm_mdcd.dbo.concept as c LEFT JOIN cdm_mdcd.dbo.concept_relationship on concept_id_1 = concept_id WHERE c.concept_code LIKE 'F%' AND c.vocabulary_id = 'ICD10CM' AND relationship_id = 'Maps to')")

psych_hosp = pd.io.sql.read_sql(query, conn)
list_psych_visits = list(psych_hosp['visit_occurrence_id'].unique())

# main processing for each temporal sample for each patient
def make_static_df(temp_pop, temp_conds, temp_meds, temp_visits, temp_procedures, temp_labs):
    ### CONDITIONS
    count_conds = temp_conds.groupby(by=["person_id", "condition_concept_id"]).size().reset_index()
    cond_features = pd.DataFrame(data=0, index=temp_pop['person_id'], columns=conditions_mapping['icd10_code'].unique())
    icd_dict = conditions_mapping.groupby('icd10_code')['standard_descendant_concept_id'].apply(list).to_dict()

    list_icd_codes = list(conditions_mapping['icd10_code'].unique())
    for i in (range(0,len(icd_dict))):
        temp = count_conds.loc[count_conds['condition_concept_id'].isin(icd_dict[list_icd_codes[i]])].groupby('person_id').count()['condition_concept_id']
        cond_features.loc[temp.index, list_icd_codes[i]] = temp.values
        
    cond_features_binary = (cond_features > 0)*1
    # adjust so that cond_features is per year of observation
    cond_features = cond_features.merge(temp_pop[['person_id','years_obs']].set_index('person_id'), how='left', left_index=True, right_index=True)
    cond_features = cond_features.div(cond_features.years_obs, axis=0) 
    cond_features.drop(['years_obs'], axis=1, inplace=True)
    
    ### MEDICATIONS
    med_features = pd.DataFrame(data=0, index=temp_pop['person_id'], columns=medications_mapping['atc_concept_name'].unique())
    meds_dict = medications_mapping.groupby('atc_concept_name')['standard_concept_id'].apply(list).to_dict()
    temp_meds['drug_exposure_days'] = (temp_meds['drug_era_end_date']-temp_meds['drug_era_start_date']).dt.days
    count_meds = temp_meds[['person_id', 'drug_concept_id', 'drug_exposure_days']].groupby(['person_id', 'drug_concept_id']).sum().reset_index()

    list_atc_codes = list(medications_mapping['atc_concept_name'].unique())
    for i in (range(0,len(meds_dict))):
        temp = count_meds.loc[count_meds['drug_concept_id'].isin(meds_dict[list_atc_codes[i]])].groupby('person_id')['drug_exposure_days'].sum()
        med_features.loc[temp.index, list_atc_codes[i]] = temp.values
        
    # adjust so that med_features is per year of observation
    med_features = med_features.merge(temp_pop[['person_id','years_obs']].set_index('person_id'), how='left', left_index=True, right_index=True)
    med_features = med_features.div(med_features.years_obs, axis=0) 
    med_features.drop(['years_obs'], axis=1, inplace=True)

    ###VISITS
    # Time from most recent visit to end of observation period
    temp_visits.merge(temp_pop[['person_id', 'cutoff_date']], how='left', left_on = 'person_id', right_on='person_id')
    temp_visits['days_to_end_obs'] = (temp_visits['cutoff_date']-temp_visits['visit_end_date']).dt.days
    if (temp_visits['visit_start_date'] > temp_visits['cutoff_date']).sum()>0:
        print('Start date issue')
    if (temp_visits['visit_end_date'] > temp_visits['cutoff_date']).sum()>0:
        print('End date issue')
    if (temp_visits['days_to_end_obs']).max() < 0:
        print('Issue with end obs')

    visits_timing = temp_visits.groupby(['person_id', 'visit_concept_id']).min()['days_to_end_obs']
    visits_timing = visits_timing.reset_index()
    visits_timing = visits_timing.pivot_table(index='person_id', columns = 'visit_concept_id', values='days_to_end_obs', fill_value = 2190)
    visits_timing.rename({9201:'most_recent_inpatient', 9202: 'most_recent_outpatient', 9203:'most_recent_ED', 42898160:'most_recent_nonhospital'}, axis=1, inplace=True)
    
    # Number of visits
    num_visits = temp_visits.groupby(['person_id', 'visit_concept_id']).count()['cohort_start_date'].reset_index()
    num_visits = num_visits.pivot_table(index='person_id', columns = 'visit_concept_id', values = 'cohort_start_date', fill_value=0)
    num_visits.rename({9201:'num_visits_inpatient', 9202: 'num_visits_outpatient', 9203:'num_visits_ED', 42898160:'num_visits_nonhospital'}, axis=1, inplace=True)
    # adjust so that num_visits is per year of observation
    num_visits = num_visits.merge(temp_pop[['person_id','years_obs']].set_index('person_id'), how='left', left_index=True, right_index=True)
    num_visits = num_visits.div(num_visits.years_obs, axis=0) 
    num_visits.drop(['years_obs'], axis=1, inplace=True)
    
    # Length of Stay
    non_outpatient = temp_visits.loc[temp_visits['visit_concept_id']!=9202]

    non_outpatient['los'] = (non_outpatient['visit_end_date']-non_outpatient['visit_start_date']).dt.days
    los = non_outpatient.groupby(['person_id', 'visit_concept_id']).agg({'los':['sum', 'max', 'min', 'mean']})
    los = los.reset_index()
    los.columns = [' '.join(col).strip() for col in los.columns.values]

    los = los.pivot_table(index='person_id', columns = 'visit_concept_id', values=['los sum', 'los max', 'los min', 'los mean'], fill_value = 0)
    los.columns = [''.join(str(col)).strip() for col in los.columns.values]

    #rename columns
    if len(los.columns) == 12:
        los.columns = ['los_max_inpatient', 'los_max_ed', 'los_max_nonhospitalization',
        'los_mean_inpatient', 'los_mean_ed', 'los_mean_nonhospitalization',
        'los_min_inpatient', 'los_min_ed', 'los_min_nonhospitalization',
        'los_sum_inpatient', 'los_sum_ed', 'los_sum_nonhospitalization']
    elif len(los.columns) == 8:
        los.columns = ['los_max_inpatient', 'los_max_ed',
        'los_mean_inpatient', 'los_mean_ed',
        'los_min_inpatient', 'los_min_ed', 
        'los_sum_inpatient', 'los_sum_ed']

    
    visits_features = visits_timing.merge(num_visits, how='outer', left_index=True, right_index=True)
    visits_features = visits_features.merge(los, how='outer', left_index=True, right_index=True)
    
    #### VISITS: INPATIENT HOSPITALIZATIONS
    # limit psych hospitalizations to ones eligible (according to preprocessed visits df)
    psych_hospitalizations = temp_visits.loc[temp_visits['visit_occurrence_id'].isin(list_psych_visits)]
    
    # Number of visits
    num_visits = psych_hospitalizations.groupby('person_id').count()['cohort_start_date'].reset_index()
    num_visits.rename({'cohort_start_date':'num_psych_hospitalizations'}, inplace=True, axis=1)
    num_visits.set_index('person_id', inplace=True)
    # adjust so that num_visits is per year of observation
    num_visits = num_visits.merge(temp_pop[['person_id','years_obs']].set_index('person_id'), how='left', left_index=True, right_index=True)
    num_visits = num_visits.div(num_visits.years_obs, axis=0) 
    num_visits.drop(['years_obs'], axis=1, inplace=True)

    visits_features = visits_features.merge(num_visits, how = 'left', right_index=True, left_index=True).fillna(0)

    # Length of Stay
    temp_visits['los'] = (temp_visits['visit_end_date']-temp_visits['visit_start_date']).dt.days
    los = temp_visits.groupby(['person_id']).agg({'los':['sum', 'max', 'min', 'mean']})
    los.columns = [' '.join(col).strip() for col in los.columns.values]
    los.columns = ['los psych sum', 'los psych max', 'los psych min', 'los psych mean']

    visits_features = visits_features.merge(los, how = 'left', right_index=True, left_index=True).fillna(0)

    # Time from most recent visit to end of observation period
    visits_timing = psych_hospitalizations.groupby('person_id').min()['days_to_end_obs']
    visits_timing.name = 'most_recent_psych_inpatient'

    # merge into visits_features
    visits_features = visits_features.merge(visits_timing, how = 'left', right_index=True, left_index=True).fillna(2190)

    ### PROCEDURES
    procedures_features = temp_procedures.pivot_table(index='person_id', columns='concept_name', aggfunc='size', fill_value=0)
    
    # get procedures per year
    procedures_features = procedures_features.merge(temp_pop[['person_id','years_obs']].set_index('person_id'), how='left', left_index=True, right_index=True)
    procedures_features = procedures_features.div(procedures_features.years_obs, axis=0) 
    procedures_features.drop(['years_obs'], axis=1, inplace=True)
    
    ### LABS
    lab_features = temp_labs.pivot_table(index='person_id', columns='concept_name', aggfunc='size', fill_value=0)

    # get labs per year
    lab_features = lab_features.merge(temp_pop[['person_id','years_obs']].set_index('person_id'), how='left', left_index=True, right_index=True)
    lab_features = lab_features.div(lab_features.years_obs, axis=0) 
    lab_features.drop(['years_obs'], axis=1, inplace=True)
    
    atemporal_features = pd.concat([cond_features, med_features, procedures_features, lab_features, visits_features], axis=1)
    return atemporal_features


# Create the sampling dataframe

# 1. sort all visits and then number them sequentially
sorted_visits = all_visits.groupby('person_id').apply(pd.DataFrame.sort_values, ['visit_start_date'])
sorted_visits.reset_index(drop=True, inplace=True)
sorted_visits = sorted_visits.merge(df_pop[['person_id', 'psychosis_diagnosis_date', 'first_visit']], how = 'left', left_on = 'person_id', right_on = 'person_id')
sorted_visits = sorted_visits[['person_id', 'psychosis_diagnosis_date', 'first_visit', 'visit_start_date', 'cohort_start_date']].drop_duplicates()
sorted_visits = sorted_visits.loc[(sorted_visits['visit_start_date']>sorted_visits['psychosis_diagnosis_date'])]

nth_visit = np.asarray(sorted_visits.groupby('person_id').cumcount())
sorted_visits['nth_visit'] = nth_visit

# limit to every 6th visit
sorted_visits = sorted_visits.loc[sorted_visits['nth_visit']%6==0]
sorted_visits['nth_visit'] = sorted_visits['nth_visit'] // 6
sorted_visits['nth_visit'] += 1

# add psychosis date as "0th" visit
add_psychosis_date = df_pop[['person_id', 'psychosis_diagnosis_date', 'first_visit', 'cohort_start_date']]
add_psychosis_date['visit_start_date'] = add_psychosis_date['psychosis_diagnosis_date']
add_psychosis_date['nth_visit'] = 0
df_iter_pop = pd.concat([sorted_visits, add_psychosis_date])

df_iter_pop.rename({'nth_visit':'iteration', 'visit_start_date':'cutoff_date'}, axis=1, inplace=True)

df_iter_pop['cohort_start_date'] = pd.to_datetime(df_iter_pop['cohort_start_date'])
df_iter_pop['cutoff_date'] = pd.to_datetime(df_iter_pop['cutoff_date'])
df_iter_pop['first_visit'] = pd.to_datetime(df_iter_pop['first_visit'])

# get the years of observation (from first vist) for each time sample
df_iter_pop['years_obs'] = (df_iter_pop['cutoff_date']-df_iter_pop['first_visit']).dt.days/365

df_iter_pop['censor_date'] = df_iter_pop['cohort_start_date']-pd.Timedelta(90, 'days') 
df_iter_pop = df_iter_pop.loc[df_iter_pop['cutoff_date']<=df_iter_pop['censor_date']]
df_iter_pop.to_csv('stored_data/iterated_population_6_visits.csv')

# loop through df_iter_pop to get feature vector for each time sample. 
# Note that this saves to a folder and each time step is a different csv -- for memory reasons
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
    temp_meds.loc[temp_meds['drug_era_end_date']>temp_meds['cutoff_date'], 'drug_era_end_date']=temp_meds['cutoff_date']

    temp_visits = all_visits.loc[all_visits['person_id'].isin(temp_df_iter_pop['person_id'])]
    temp_visits = temp_visits.merge(temp_df_iter_pop[['person_id','cutoff_date']], how = 'left', left_on = 'person_id', right_on = 'person_id')
    temp_visits = temp_visits.loc[temp_visits['visit_start_date']< temp_visits['cutoff_date']]
    temp_visits.loc[temp_visits['visit_end_date']>temp_visits['cutoff_date'], 'visit_end_date']=temp_visits['cutoff_date']
    
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
    
    all_features.to_csv('stored_data/visit_iters_6/iter_'+str(iteration)+'.csv')

