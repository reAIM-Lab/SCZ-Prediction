import numpy as np
import os
import pandas as pd
import time
from datetime import datetime
import sys
import gc
import pickle
from itertools import product


def identify_exclusive_mappings(psychosis_codes):
    """
    This function takes in a list of ICD codes and returns all of the 
    snomed mappings, divided into "exclusive" (this SNOMED term only 
    maps to ICD codes in the psychosis_codes list) and "nonexclusive" 
    (this snomed term maps to ICD codes both on and not on the psychosis_codes list)
    """
    
    
    # get all ICD codes and their SNOMED Mappings
    icd_to_snomed_query = """SELECT icd_c.concept_name as icd_concept_name, icd_c.concept_code, icd_c.vocabulary_id, concept_id_1 as icd_mapping_concept, concept_id_2 as snomed_mapping_concept, snomed_c.concept_name as snomed_concept_name, snomed_c.standard_concept
    FROM dbo.concept as icd_c
    LEFT JOIN dbo.concept_relationship AS cr ON cr.concept_id_1 = icd_c.concept_id
    LEFT JOIN dbo.concept AS snomed_c ON cr.concept_id_2 = snomed_c.concept_id
    WHERE icd_c.vocabulary_id IN ('ICD9CM', 'ICD10CM') AND relationship_id = 'Maps to'
    """

    icd_to_snomed_df = pd.io.sql.read_sql(icd_to_snomed_query, conn)
    
    # get all snomed conditions and their respective ICD Codes
    snomed_to_icd_query = """SELECT snomed_c.concept_id as snomed_concept_id, snomed_c.concept_name as snomed_concept_name, snomed_c.standard_concept, icd_c.concept_id as icd_concept_id, icd_c.concept_name as icd_concept_name, icd_c.vocabulary_id, icd_c.concept_code
    FROM dbo.concept as snomed_c
    LEFT JOIN dbo.concept_relationship as cr ON snomed_c.concept_id = cr.concept_id_1
    LEFT JOIN dbo.concept as icd_c ON cr.concept_id_2 = icd_c.concept_id
    WHERE icd_c.vocabulary_id IN ('ICD9CM', 'ICD10CM') AND relationship_id = 'Mapped from'
    """

    snomed_to_icd_df = pd.io.sql.read_sql(snomed_to_icd_query, conn)
    
    broad_psychosis_codes = icd_to_snomed_df.loc[icd_to_snomed_df['concept_code'].isin(psychosis_codes)]
    snomed_to_icd_psychosis = snomed_to_icd_df.loc[snomed_to_icd_df['snomed_concept_id'].isin(broad_psychosis_codes['snomed_mapping_concept'])]

    # Get all snomed codes in broad psychosis_codes that only map to psychosis ICD codes (list_keep)
    list_toss = []
    for i in snomed_to_icd_psychosis['snomed_concept_id'].unique():
        icds_per_snomeds = (snomed_to_icd_psychosis.loc[snomed_to_icd_psychosis['snomed_concept_id']==i, 'concept_code'].unique())
        if len(set(icds_per_snomeds).difference(psychosis_codeset['concept_code'])) > 0:
            list_toss.append(snomed_to_icd_psychosis.loc[snomed_to_icd_psychosis['snomed_concept_id']==i, 'snomed_concept_name'].unique()[0])
            
    list_keep = []
    for i in snomed_to_icd_psychosis['snomed_concept_id'].unique():
        icds_per_snomeds = (snomed_to_icd_psychosis.loc[snomed_to_icd_psychosis['snomed_concept_id']==i, 'concept_code'].unique())
        if len(set(icds_per_snomeds).difference(psychosis_codeset['concept_code'])) == 0:
            list_keep.append(snomed_to_icd_psychosis.loc[snomed_to_icd_psychosis['snomed_concept_id']==i, 'snomed_concept_name'].unique()[0])
    
    return list_keep, list_toss, snomed_to_icd_psychosis, broad_psychosis_codes

def drop_rare_occurrences(df, col_concept, col_id = 'person_id', size_pop = len(df_pop)):
    unique_occurrences = df[['person_id', col_concept]].drop_duplicates()
    unique_occurrences = unique_occurrences.value_counts(col_concept)
    common_occurrences = unique_occurrences[unique_occurrences/size_pop > 0.01].index
    return df.loc[df[col_concept].isin(common_occurrences)]

def generate_code_list(drugtype, concept_class):
    sql_query = ("SELECT ancestor_concept_id, descendant_concept_id, concept_name " + 
               "FROM cdm_mdcd.dbo.concept_ancestor JOIN cdm_mdcd.dbo.concept ON descendant_concept_id = concept_id "+
               "WHERE ancestor_concept_id = (SELECT concept_id from cdm_mdcd.dbo.concept WHERE concept_class_id = '"+concept_class+"' AND concept_name = '"+drugtype+"');")
    codes_list = pd.read_sql(sql_query, conn)
    return list(codes_list['descendant_concept_id'])


# Define a function that gives us a dataframe of person_id by features (features units are count per year)
# temp pop needs to have a "years obs" column
def make_static_df(temp_pop, temp_conds, temp_meds, temp_visits, temp_procedures, temp_labs):
    ### CONDITIONS
    count_conds = temp_conds.groupby(by=["person_id", "condition_concept_id"]).size().reset_index()
    cond_features = pd.DataFrame(data=0, index=temp_pop['person_id'], columns=conditions_mapping['icd10_code'].unique())
    icd_dict = conditions_mapping.groupby('icd10_code')['standard_descendant_concept_id'].apply(list).to_dict()

    list_icd_codes = list(conditions_mapping['icd10_code'].unique())
    for i in (range(0,len(icd_dict))):
        temp = count_conds.loc[count_conds['condition_concept_id'].isin(icd_dict[list_icd_codes[i]])].groupby('person_id').sum()[0]
        cond_features.loc[temp.index, list_icd_codes[i]] = temp.values
        
    cond_features_binary = (cond_features > 0)*1
    # eliminate icd10 codes with <= 1% prevalence
    #cond_features_binary = cond_features_binary[cond_features_binary.columns[100*cond_features_binary.sum(axis=0)/len(cond_features_binary)>1]]
    #cond_features = cond_features[cond_features_binary.columns[100*cond_features_binary.sum(axis=0)/len(cond_features_binary)>1]]
    # adjust so that cond_features is per year of observation
    cond_features = cond_features.merge(temp_pop[['person_id','years_obs']].set_index('person_id'), how='left', left_index=True, right_index=True)
    cond_features = cond_features.div(cond_features.years_obs, axis=0) 
    cond_features.drop(['years_obs'], axis=1, inplace=True)
    
    ### MEDICATIONS
    med_features = pd.DataFrame(data=0, index=temp_pop['person_id'], columns=medications_mapping['atc_concept_name'].unique())
    meds_dict = medications_mapping.groupby('atc_concept_name')['standard_concept_id'].apply(list).to_dict()
    temp_meds['drug_exposure_days'] = (temp_meds['drug_era_end_date']-temp_meds['drug_era_start_date']).dt.days + 1 # +1 so a 1-day prescription will not be 0 days
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
    """
    # drop procedures do not occur in at least 1% of patients
    #procedure_feature_select = temp_procedures[['person_id', 'procedure_concept_id']].drop_duplicates()
    #procedure_feature_select = procedure_feature_select.groupby('procedure_concept_id').count()
    #procedure_feature_select['prevalence'] = 100*procedure_feature_select['person_id']/len(temp_pop)
    #common_procedures = list(procedure_feature_select.loc[procedure_feature_select['prevalence']>1].index)
    
    #all_common_procedures = temp_procedures.loc[temp_procedures['procedure_concept_id'].isin(common_procedures)]
    # use a pivot table to get the counts and binary occurrence of each procedure code
    """
    procedures_features = temp_procedures.pivot_table(index='person_id', columns='concept_name', aggfunc='size', fill_value=0)
    
    # get procedures per year
    procedures_features = procedures_features.merge(temp_pop[['person_id','years_obs']].set_index('person_id'), how='left', left_index=True, right_index=True)
    procedures_features = procedures_features.div(procedures_features.years_obs, axis=0) 
    procedures_features.drop(['years_obs'], axis=1, inplace=True)
    
    ### LABS
    # drop labs that do not occur in at least 1% of patients
    """
    lab_feature_select = temp_labs[['person_id', 'measurement_concept_id']].drop_duplicates()
    lab_feature_select = lab_feature_select.groupby('measurement_concept_id').count()
    lab_feature_select['prevalence'] = 100*lab_feature_select['person_id']/len(temp_pop)
    common_labs = list(lab_feature_select.loc[lab_feature_select['prevalence']>1].index)

    all_common_labs = temp_labs.loc[temp_labs['measurement_concept_id'].isin(common_labs)]
    # use a pivot table to get the counts and binary occurrence of each lab code
    """
    lab_features = temp_labs.pivot_table(index='person_id', columns='concept_name', aggfunc='size', fill_value=0)

    # get procedures per year
    lab_features = lab_features.merge(temp_pop[['person_id','years_obs']].set_index('person_id'), how='left', left_index=True, right_index=True)
    lab_features = lab_features.div(lab_features.years_obs, axis=0) 
    lab_features.drop(['years_obs'], axis=1, inplace=True)
    
    atemporal_features = pd.concat([cond_features, med_features, procedures_features, lab_features, visits_features], axis=1)
    return atemporal_features
