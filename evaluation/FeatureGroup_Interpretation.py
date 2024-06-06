import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from tqdm import tqdm
from joblib import dump, load
import pickle
from sklearn.metrics import *
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb

sys.path.append('../utils')
from eval_utils import *

data_path = '../prediction_data/'
# read in population dataframe
num_days_prediction = 90
df_pop = pd.read_csv(data_path+"population.csv")
df_pop['psychosis_diagnosis_date'] = pd.to_datetime(df_pop['psychosis_diagnosis_date'], format="%Y-%m-%d")
df_pop['cohort_start_date'] = pd.to_datetime(df_pop['cohort_start_date'])
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction]

shap_values_conditions, top_shap_values_conditions = get_shap_values('test_df_5_21_condsonly.csv', 'xgb_8_visits_5_21_condsonly.pkl')
shap_values_meds, top_shap_values_meds = get_shap_values('test_df_5_21_medsonly.csv', 'xgb_8_visits_5_21_medsonly.pkl')
shap_values_visits, top_shap_values_visits = get_shap_values('test_df_5_21_visitsonly.csv', 'xgb_8_visits_5_21_visitsonly.pkl')
shap_values_procedures, top_shap_values_procedures = get_shap_values('test_df_5_21_proceduresonly.csv', 'xgb_8_visits_5_21_proceduresonly.pkl')
shap_values_labs, top_shap_values_labs = get_shap_values('test_df_5_21_labsonly.csv', 'xgb_8_visits_5_21_labsonly.pkl')

print(top_shap_values_conditions.columns) #RENAME
condition_names = ['J06 (Acute upper respiratory infection)', 'L30 (Unspecified dermatitis)',
                  'H52 (Disorders of refraction and accommodation)', 'F78 (Intellectual disabilities)',
                  'B95 (Streptococcus, staphylococcus, & \nenterococcus as cause of disease)', 'H54 (Blindness and low vision)',
                  'R05 (Cough)', 'R86 (Abnormal findings, male genital organs)', 
                  'H60 (Otitis externa)', 'N39 (Other disorders of urinary system)']

print(top_shap_values_meds.columns) #RENAME
med_names = ['Antipsychotics', 'Antiinflammatory/antirheumatic products (non-steroid)', 
             'Psychostimulants (used for ADHD & nootropics)', 'Other antibacterials',
             'Mood stabilizers', 'Antidepressants', 
             'Other opthamologicals', 'Antacids',
             'Vitamin B12 and folic acid', 'Corticosteroids for systemic use']

print(top_shap_values_visits.columns) #RENAME
visit_names = ['Total inpatient LOS', 'Total ED LOS', 
              'Number of outpatient visits', 'Shortest inpatient LOS', 
              'Number of psychiatric hospitalizations', 'Number of ED visits', 
              'Longest psychiatric inpatient LOS', 'Mean inpatient LOS',
               'Number of inpatient visits', 'Days since last outpatient visit']

print(top_shap_values_procedures.columns) #RENAME
procedure_names = ['Psychiatric diagnostic evaluation (with medical services)', 'Physical examination, limited',
                  'Mental health assessment (non-physician)', 'Psychiatric diagnostic interview examination',
                  'Office or outpatient visit, new patient', 'Periodic oral examination', 
                  'Office or outpatient visit, established patient', 'Collection of venous blood by venipuncture',
                  'Pharmacologic management with minimal psychotherapy', 'Determination of refractive state']

print(top_shap_values_labs.columns) #RENAME
lab_names = ['Urinalysis with microscopy', 'Comprehensive metabolic panel', 
             'Basic metabolic panel', 'Lipid panel', 
             'Blood count; hemoglobin', 'Thyroxine; free',
            'HIV antibody test', 'Urinalysis without microscopy',
            'Influenza immunoassay', 'General health panel']

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(20,40))
font = {'weight' : 'bold',
        'size'   : 24}
matplotlib.rc('font', **font)

plt.subplot(5,1,1)
matplotlib.rc('font', **font)
plt.violinplot(top_shap_values_conditions, vert=False, showmeans=True, showextrema=True)
plt.yticks(np.arange(1, len(condition_names)+1), condition_names, fontsize=22)
plt.xlabel('SHAP Value')
plt.xlim([-7.5, 12])
plt.title('Most important features in condition-only model')
ax = axes[0]
ax.invert_yaxis()

plt.subplot(5,1,2)
matplotlib.rc('font', **font)
plt.violinplot(top_shap_values_meds, vert=False, showmeans=True, showextrema=True)
plt.yticks(np.arange(1, len(med_names)+1), med_names, fontsize=22)
plt.xlabel('SHAP Value')
plt.xlim([-7.5, 12])
plt.title('Most important features in medication-only model')
ax = axes[1]
ax.invert_yaxis()

plt.subplot(5,1,3)
matplotlib.rc('font', **font)
plt.violinplot(top_shap_values_visits, vert=False, showmeans=True, showextrema=True)
plt.yticks(np.arange(1, len(visit_names)+1), visit_names, fontsize=22)
plt.xlabel('SHAP Value')
plt.xlim([-7.5, 12])
plt.title('Most important features in visit-only model')
ax = axes[2]
ax.invert_yaxis()

plt.subplot(5,1,4)
matplotlib.rc('font', **font)
plt.violinplot(top_shap_values_procedures, vert=False, showmeans=True, showextrema=True)
plt.yticks(np.arange(1, len(procedure_names)+1), procedure_names, fontsize=22)
plt.xlabel('SHAP Value')
plt.xlim([-7.5, 12])
plt.title('Most important features in procedure-only model')
ax = axes[3]
ax.invert_yaxis()

plt.subplot(5,1,5)
matplotlib.rc('font', **font)
plt.violinplot(top_shap_values_labs, vert=False, showmeans=True, showextrema=True)
plt.yticks(np.arange(1, len(lab_names)+1), lab_names, fontsize=22)
plt.xlabel('SHAP Value')
plt.xlim([-7.5, 12])
plt.title('Most important features in lab test-only model')
ax = axes[4]
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('results/shap_ablation_5_21.pdf', dpi=300, bbox_inches='tight')
plt.show()