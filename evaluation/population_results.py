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

# load test data for evaluation
test_labels = pd.read_csv('stored_data/5_21_model_test_output.csv')
df_iter_pop = pd.read_csv('stored_data/iterated_population_8_visits_5_21.csv')

test_labels = test_labels.merge(df_pop[['person_id', 'race_concept_id', 'gender_concept_id']], how='left',
                 left_on='person_id', right_on='person_id')

test_labels = test_labels.merge(df_iter_pop[['person_id', 'iteration', 'psychosis_diagnosis_date', 'cutoff_date', 'censor_date']], how='inner', left_on = ['person_id', 'iteration'], right_on=['person_id', 'iteration'])

print('Accuracy', (test_labels['sz_flag'] == test_labels['y_pred']).sum()/len(test_labels))
tn, fp, fn, tp = confusion_matrix(test_labels['sz_flag'], test_labels['y_pred']).ravel()
print('Sensitivity', (tp/(tp+fn)))
print('Specificity', (tn/(tn+fp)))
print('AUPRC', average_precision_score(test_labels['sz_flag'], test_labels['prob_1']))
print('PPV', precision_score(test_labels['sz_flag'], test_labels['y_pred']))
print('AUROC', roc_auc_score(test_labels['sz_flag'], test_labels['prob_1']))

table2 = pd.DataFrame(columns = ['AUROC', 'AUROC CI', 'Accuracy', 'Accuracy CI',
                                   'Sensitivity', 'Sensitivity CI', 'Specificity', 'Specificity CI',
                                'AUPRC', 'AUPRC_CI', 'PPV', 'PPV_CI'])

table2.loc['All'] = create_table2_row(test_labels)

black_patients = test_labels.loc[test_labels['race_concept_id']==8516]
table2.loc['Black'] = create_table2_row(black_patients)

white_patients = test_labels.loc[test_labels['race_concept_id']==8527]
table2.loc['White'] = create_table2_row(white_patients)

missing_race_patients = test_labels.loc[test_labels['race_concept_id']==0]
table2.loc['Missing'] = create_table2_row(missing_race_patients)

female_patients = test_labels.loc[test_labels['gender_concept_id']==8532]
table2.loc['Women'] = create_table2_row(female_patients)

male_patients = test_labels.loc[test_labels['gender_concept_id']==8507]
table2.loc['Men'] = create_table2_row(male_patients)

table2.to_csv('results/table2_5_21.csv')

# supplementary figure 1: performance over time
test_labels['psychosis_diagnosis_date'] = pd.to_datetime(test_labels['psychosis_diagnosis_date'])
test_labels['cutoff_date'] = pd.to_datetime(test_labels['cutoff_date'])
test_labels['censor_date'] = pd.to_datetime(test_labels['censor_date'])

# get time from psychosis dx to cutoff
test_labels['time_to_cutoff'] = (test_labels['cutoff_date']-test_labels['psychosis_diagnosis_date']).dt.days/365
print(test_labels['time_to_cutoff'].max())

# get time from psychosis dx to censor
test_labels['time_to_censor'] = (test_labels['censor_date']-test_labels['psychosis_diagnosis_date']).dt.days/365

# because we have a max of 9.75 years, we will look up to 10 years
time_checks = np.arange(0,10.5,0.5)
test_labels_with_index = test_labels.set_index(['person_id', 'iteration'])

list_timed_subgroups = []
for ind in time_checks:

    # get the max iteration where the time between psychosis and cutoff is still under our time
    most_recent_visit = (test_labels.loc[test_labels['time_to_cutoff']<= ind].groupby('person_id').max()['iteration']).reset_index().values
    timed_subgroup = test_labels_with_index.loc[list(map(tuple, most_recent_visit))]

    # remove anyone for whom the time between psychosis and censor date is less than the 
    # amount of time out we are looking (i.e. they have reached their index date)
    timed_subgroup = timed_subgroup.loc[timed_subgroup['time_to_censor']>=ind]
    
    timed_subgroup['time_forward_iteration'] = ind
    list_timed_subgroups.append(timed_subgroup)
forward_iter_test_labels = pd.concat(list_timed_subgroups)
forward_iter_test_labels.reset_index(inplace=True)

## Forwards iterations, by time
iterations = forward_iter_test_labels['time_forward_iteration'].unique()
iterations.sort()
auroc, auroc_ci, auprc, auprc_ci, num_patients, frac_pos_samples, num_visits, num_visits_ci = results_per_iter(forward_iter_test_labels, 'time_forward_iteration')

black_patients = forward_iter_test_labels.loc[forward_iter_test_labels['race_concept_id']==8516]
white_patients = forward_iter_test_labels.loc[forward_iter_test_labels['race_concept_id']==8527]
missing_patients = forward_iter_test_labels.loc[forward_iter_test_labels['race_concept_id']==0]

female_patients = forward_iter_test_labels.loc[forward_iter_test_labels['gender_concept_id']==8532]
male_patients = forward_iter_test_labels.loc[forward_iter_test_labels['gender_concept_id']==8507]

auroc_b, auroc_ci_b, auprc_b, auprc_ci_b, num_patients_b, frac_pos_samples_b, num_visits_b, num_visits_ci_b = results_per_iter(black_patients, 'time_forward_iteration')
auroc_w, auroc_ci_w, auprc_w, auprc_ci_w, num_patients_w, frac_pos_samples_w, num_visits_w, num_visits_ci_w = results_per_iter(white_patients, 'time_forward_iteration')
auroc_f, auroc_ci_f, auprc_f, auprc_ci_f, num_patients_f, frac_pos_samples_f, num_visits_f, num_visits_ci_f = results_per_iter(female_patients, 'time_forward_iteration')
auroc_m, auroc_ci_m, auprc_m, auprc_ci_m, num_patients_m, frac_pos_samples_m, num_visits_m, num_visits_ci_m = results_per_iter(male_patients, 'time_forward_iteration')
auroc_missing, auroc_ci_missing, auprc_missing, auprc_ci_missing, num_patients_missing, frac_pos_samples_missing, num_visits_missing, num_visits_ci_missing = results_per_iter(missing_patients, 'time_forward_iteration')

font = {'weight' : 'bold',
        'size'   : 22}


plt.figure(figsize=(25, 7))
matplotlib.rc('font', **font)

cutoff_ind = 16
## Forwards iterations, by time
n_years = iterations[0:cutoff_ind]

auroc = np.asarray(auroc[0:cutoff_ind])
auroc_ci = np.asarray(auroc_ci[0:cutoff_ind])

plt.subplot(1,3,1)
# all patients
plt.plot(n_years, auroc, color = 'black', marker = 'o')
plt.fill_between(n_years, np.abs(auroc_ci.T)[0,:], np.abs(auroc_ci.T)[1,:], color='black', alpha=0.3)

plt.xlabel('Number of years after psychosis diagnosis')
plt.ylabel('Area under the reciever operating curve')
plt.title('Overall model performance')
plt.ylim([0.90, 1])


plt.subplot(1,3,2)
## Black Patients
auroc_b = np.asarray(auroc_b[0:cutoff_ind])
auroc_ci_b = np.asarray(auroc_ci_b[0:cutoff_ind])
plt.plot(n_years, auroc_b, label = 'Black Patients', color = 'red', marker = '^')
plt.fill_between(n_years, np.abs(auroc_ci_b.T)[0,:], np.abs(auroc_ci_b.T)[1,:], color='red', alpha=0.2)

## White Patients
auroc_w = np.asarray(auroc_w[0:cutoff_ind])
auroc_ci_w = np.asarray(auroc_ci_w[0:cutoff_ind])
plt.plot(n_years, auroc_w, label = 'White Patients', color = 'blue', marker = 's')
plt.fill_between(n_years, np.abs(auroc_ci_w.T)[0,:], np.abs(auroc_ci_w.T)[1,:], color='blue', alpha=0.2)

## Missing Patients
auroc_missing = np.asarray(auroc_missing[0:cutoff_ind])
auroc_ci_missing = np.asarray(auroc_ci_missing[0:cutoff_ind])
plt.plot(n_years, auroc_missing, label = 'Missing Race Patients', color = 'orange', marker = 's')
plt.fill_between(n_years, np.abs(auroc_ci_missing.T)[0,:], np.abs(auroc_ci_missing.T)[1,:], color='orange', alpha=0.2)

plt.xlabel('Number of years after psychosis diagnosis')
plt.ylabel('Area under the reciever operating curve')
plt.title('Model performance by race')
plt.ylim([0.90, 1])
plt.legend()

plt.subplot(1,3,3)
## Female Patients
auroc_f = np.asarray(auroc_f[0:cutoff_ind])
auroc_ci_f = np.asarray(auroc_ci_f[0:cutoff_ind])
plt.plot(n_years, auroc_f, label = 'Women', color = 'red', marker = 'd')
plt.fill_between(n_years, np.abs(auroc_ci_f.T)[0,:], np.abs(auroc_ci_f.T)[1,:], color='red', alpha=0.2)

## Male Patients
auroc_m = np.asarray(auroc_m[0:cutoff_ind])
auroc_ci_m = np.asarray(auroc_ci_m[0:cutoff_ind])
plt.plot(n_years, auroc_m, label = 'Men', color = 'blue', marker = 'v')
plt.fill_between(n_years, np.abs(auroc_ci_m.T)[0,:], np.abs(auroc_ci_m.T)[1,:], color='blue', alpha=0.2)

plt.xlabel('Number of years after psychosis diagnosis')
plt.title('Model performance by gender')
plt.ylabel('Area under the reciever operating curve')
plt.ylim([0.90, 1])
plt.legend()
plt.tight_layout()
plt.savefig('results/xgboost_time_performance_auroc_5_21.pdf', dpi=300)
plt.show()

