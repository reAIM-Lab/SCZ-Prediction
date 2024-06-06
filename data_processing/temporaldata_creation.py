import numpy as np
import os
import pandas as pd
import pyodbc
import time
import pickle
import gc

connection_string = ('YOUR CREDENTIALS HERE')
conn = pyodbc.connect(connection_string)

path = 'prediction_data/'
df_pop = pd.read_csv(path+'population.csv')

# Medications
sz_meds_query = ("SELECT sz.person_id, sz.end_date, drug_concept_id, drug_era_start_date, drug_era_end_date, drug_exposure_count, gap_days "+ 
                 "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                   "LEFT JOIN cdm_mdcd.dbo.drug_era on drug_era.person_id = sz.person_id")


sz_meds = pd.io.sql.read_sql(sz_meds_query, conn)
sz_meds.columns = sz_meds.columns.str.lower()

nosz_meds_query = ("SELECT pc.person_id, pc.end_date, drug_concept_id, drug_era_start_date, drug_era_end_date, drug_exposure_count, gap_days "+ 
                 "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                   "LEFT JOIN cdm_mdcd.dbo.drug_era on drug_era.person_id = pc.person_id")
list_chunks = []
for chunk in pd.io.sql.read_sql(nosz_meds_query, conn, chunksize=1000000):
    list_chunks.append(chunk.loc[chunk['person_id'].isin(df_pop['person_id'])])
nosz_meds = pd.concat(list_chunks)
all_meds = pd.concat([sz_meds, nosz_meds])
all_meds = all_meds.loc[all_meds['person_id'].isin(list(df_pop['person_id']))]
all_meds = all_meds.merge(df_pop[['person_id', 'cohort_start_date', 'psychosis_diagnosis_date']], how='left', left_on = 'person_id', right_on='person_id')
all_meds = all_meds.loc[all_meds['drug_concept_id']>0]
all_meds.dropna(inplace=True)
print(len(all_meds))
print(len(all_meds['person_id'].unique()))
print(all_meds.isna().sum().sum())
all_meds.to_csv(path+'temporal_medications.csv')


# Visits
sz_visits_query = ("SELECT sz.person_id, sz.end_date, visit_occurrence_id, visit_concept_id, visit_start_date, visit_end_date, visit_type_concept_id " +
                   "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                   "LEFT JOIN cdm_mdcd.dbo.visit_occurrence as v on v.person_id = sz.person_id")

sz_visits = pd.io.sql.read_sql(sz_visits_query, conn)

nosz_visits_query = ("SELECT pc.person_id, pc.end_date, visit_occurrence_id, visit_concept_id, visit_start_date, visit_end_date, visit_type_concept_id " +
                   "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                   "LEFT JOIN cdm_mdcd.dbo.visit_occurrence as v on v.person_id = pc.person_id")

list_chunks = []
for chunk in pd.io.sql.read_sql(nosz_visits_query, conn, chunksize=1000000):
    list_chunks.append(chunk.loc[chunk['person_id'].isin(df_pop['person_id'])])
nosz_visits = pd.concat(list_chunks)
all_visits = pd.concat([sz_visits, nosz_visits])
all_visits = all_visits.loc[all_visits['person_id'].isin(list(df_pop['person_id']))]
all_visits = all_visits.merge(df_pop[['person_id', 'cohort_start_date', 'psychosis_diagnosis_date']], how='left', left_on = 'person_id', right_on='person_id')
all_visits = all_visits.loc[all_visits['visit_concept_id']>0]
all_visits.dropna(inplace=True)
print(len(all_visits))
print(len(all_visits['person_id'].unique()))
all_visits.to_csv(path+'temporal_visits.csv')
print(all_visits.isna().sum().sum())

# Procedures
sz_procedures_query = ("SELECT sz.person_id, sz.end_date, procedure_date, procedure_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                  "LEFT JOIN cdm_mdcd.dbo.procedure_occurrence as po on po.person_id = sz.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = po.procedure_concept_id")

sz_procedures = pd.io.sql.read_sql(sz_procedures_query, conn)

nosz_procedures_query = ("SELECT pc.person_id, pc.end_date, procedure_date, procedure_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                  "LEFT JOIN cdm_mdcd.dbo.procedure_occurrence as po on po.person_id = pc.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = po.procedure_concept_id")

list_chunks = []
for chunk in pd.io.sql.read_sql(nosz_procedures_query, conn, chunksize=1000000):
    list_chunks.append(chunk.loc[chunk['person_id'].isin(df_pop['person_id'])])
nosz_procedures = pd.concat(list_chunks)
all_procedures = pd.concat([sz_procedures, nosz_procedures])
all_procedures = all_procedures.loc[all_procedures['person_id'].isin(list(df_pop['person_id']))]
all_procedures = all_procedures.loc[all_procedures['procedure_concept_id']>0]
all_procedures = all_procedures.merge(df_pop[['person_id', 'cohort_start_date', 'psychosis_diagnosis_date']], how='left', left_on = 'person_id', right_on='person_id')
all_procedures.dropna(inplace=True)
print(len(all_procedures))
print(len(all_procedures['person_id'].unique()))
all_procedures.to_csv(path+'temporal_procedures.csv')
print(all_procedures.isna().sum().sum())

# Labs
sz_measurement_query = ("SELECT sz.person_id, sz.end_date, measurement_date, measurement_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                  "LEFT JOIN cdm_mdcd.dbo.measurement as m on m.person_id = sz.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = m.measurement_concept_id")

sz_labs = pd.io.sql.read_sql(sz_measurement_query, conn)

nosz_measurements_query = ("SELECT pc.person_id, pc.end_date, measurement_date, measurement_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                  "LEFT JOIN cdm_mdcd.dbo.measurement as m on m.person_id = pc.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = m.measurement_concept_id")
list_chunks = []
for chunk in pd.io.sql.read_sql(nosz_measurements_query, conn, chunksize=1000000):
    list_chunks.append(chunk.loc[chunk['person_id'].isin(df_pop['person_id'])])
nosz_labs = pd.concat(list_chunks)
all_labs = pd.concat([sz_labs, nosz_labs])
all_labs = all_labs.loc[all_labs['person_id'].isin(list(df_pop['person_id']))]
all_labs = all_labs.loc[all_labs['measurement_concept_id']>0]
all_labs = all_labs.merge(df_pop[['person_id', 'cohort_start_date', 'psychosis_diagnosis_date']], how='left', left_on = 'person_id', right_on='person_id')
all_labs.dropna(inplace=True)
print(len(all_labs))
print(len(all_labs['person_id'].unique()))
all_labs.to_csv(path+'temporal_labs.csv')
print(all_labs.isna().sum().sum())

# Conditions
sz_conds_query = ("SELECT sz.person_id, sz.end_date, condition_start_date, condition_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_schizophrenia_cohort as sz "+
                  "LEFT JOIN cdm_mdcd.dbo.condition_occurrence as co on co.person_id = sz.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = co.condition_concept_id "+
                  "WHERE condition_concept_id > 0")

sz_conds = pd.io.sql.read_sql(sz_conds_query, conn)

nosz_conds_query = ("SELECT pc.person_id, pc.end_date, condition_start_date, condition_concept_id, c.concept_name "+
                  "FROM cdm_mdcd.results.ak4885_psychosis_cohort as pc "+
                  "LEFT JOIN cdm_mdcd.dbo.condition_occurrence as co on co.person_id = pc.person_id "+
                  "LEFT JOIN cdm_mdcd.dbo.concept as c on c.concept_id = co.condition_concept_id "+
                  "WHERE condition_concept_id > 0")
list_chunks = []
for chunk in pd.io.sql.read_sql(nosz_conds_query, conn, chunksize=1000000):
    list_chunks.append(chunk)
nosz_conds = pd.concat(list_chunks)
all_conds = pd.concat([sz_conds, nosz_conds])
all_conds = all_conds.loc[all_conds['person_id'].isin(list(df_pop['person_id']))]
all_conds = all_conds.merge(df_pop[['person_id', 'cohort_start_date', 'psychosis_diagnosis_date']], how='left', left_on = 'person_id', right_on='person_id')
all_conds.dropna(inplace=True)
print(len(all_conds))
print(len(all_conds['person_id'].unique()))
all_conds.to_csv(path+'temporal_conditions.csv')
print(all_conds.isna().sum().sum())