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

path = '../../'

# read in population dataframe
num_days_prediction = 90
df_pop = pd.read_csv(path+"population.csv")
df_pop.rename({'psychosis_dx_date':'psychosis_diagnosis_date'}, axis=1, inplace=True)
df_pop['psychosis_diagnosis_date'] = pd.to_datetime(df_pop['psychosis_diagnosis_date'], format="%Y-%m-%d")
df_pop['cohort_start_date'] = pd.to_datetime(df_pop['cohort_start_date'])
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction]

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
t1_percents

# table 1: years of observation prior to psychosis
all_visits = pd.read_csv(path+'temporal_visits.csv')
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
all_visits = all_visits.merge(df_pop[['person_id', 'psychosis_diagnosis_date']], left_on = 'person_id', right_on = 'person_id', how='inner')
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

# load testing data
test_pids = np.load('../stored_data/test_pids_4_2_24.npy')

list_files = []
list_filenames = os.listdir('../stored_data/visit_iters_6')
for filename_ind in tqdm(range(len(list_filenames))):
    filename = list_filenames[filename_ind]
    list_files.append(pd.read_csv('../stored_data/visit_iters_6/'+filename))
df_all_iters = pd.concat(list_files)
df_all_iters.fillna(0, inplace=True)

num_days_prediction = 90
df_pop = pd.read_csv(path+'population.csv')
df_pop.rename({'psychosis_dx_date':'psychosis_diagnosis_date'}, axis=1, inplace=True)
df_pop['psychosis_diagnosis_date'] = pd.to_datetime(df_pop['psychosis_diagnosis_date'], format="%Y-%m-%d")
df_pop['cohort_start_date'] = pd.to_datetime(df_pop['cohort_start_date'])
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction]

df_pop = df_pop.loc[df_pop['person_id'].isin(df_all_iters['person_id'])]

labels = df_all_iters[['person_id', 'iteration']].merge(df_pop[['person_id','sz_flag']], how='left', left_on = 'person_id', right_on='person_id')
labels.set_index('person_id', inplace=True)

df_all_iters.set_index('person_id', inplace=True)
df_all_iters.drop(['iteration'], inplace=True, axis=1)
save_cols = df_all_iters.columns

X_test = df_all_iters.loc[test_pids]
y_test = labels.loc[test_pids, 'sz_flag']

scaler = load('../stored_data/scaler_4_2_24.bin')
X_test = scaler.transform(X_test)

df_iter_pop = pd.read_csv('../stored_data/iterated_population_6_visits.csv')
iterated_testing_df = df_iter_pop.loc[df_iter_pop['person_id'].isin(test_pids)]

# Run model on test set
with open('../models/xgb_every_6_visits.pkl', 'rb') as f:
    testing_clf = pickle.load(f)

y_pred_proba = testing_clf.predict_proba(X_test)
test_labels = labels.loc[test_pids]

test_labels['prob_1'] = y_pred_proba[:,1]
test_labels['y_pred'] = np.round(test_labels['prob_1'])
test_labels.reset_index(inplace=True)
test_labels = test_labels.merge(iterated_testing_df[['person_id', 'iteration', 'years_obs', 'psychosis_diagnosis_date', 'cutoff_date', 'cohort_start_date','censor_date', 'first_visit']], how = 'left', left_on = ['person_id', 'iteration'], right_on = ['person_id', 'iteration'])

test_labels = test_labels.merge(df_pop[['person_id', 'race_concept_id', 'gender_concept_id']], how='left',
                 left_on='person_id', right_on='person_id')


# table 2: model performance
def get_ci(y_test, y_pred, pred_prob, threshold=0.95):
    """
    gives us 95% CI for auroc, auprc
    """
    rng = np.random.RandomState(seed=44)
    idx = np.arange(y_test.shape[0])

    test_auroc = []
    test_auprc = []
    for i in range(300):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        if len(set(y_test.iloc[pred_idx])) > 1:
            test_auroc.append(roc_auc_score(y_test.iloc[pred_idx], pred_prob.iloc[pred_idx]))
            test_auprc.append(average_precision_score(y_test.iloc[pred_idx], pred_prob.iloc[pred_idx]))
            
    auroc_interval = (np.percentile(test_auroc, 2.5), np.percentile(test_auroc, 97.5))
    auprc_interval = (np.percentile(test_auprc, 2.5), np.percentile(test_auprc, 97.5))
    
    return auroc_interval, auprc_interval

def results_per_iter(test_value_subset, iterations_name):
    iterations = test_value_subset[iterations_name].unique()
    iterations.sort()
    auroc = []
    auroc_ci = []
    auprc = []
    auprc_ci = []

    num_patients = []
    frac_pos_samples = []
    num_visits = []
    num_visits_ci = []
    for i in iterations:
        df_subset = test_value_subset.loc[test_value_subset[iterations_name]==i]
        if len(df_subset['sz_flag'].unique()) > 1:
            auroc_val = roc_auc_score(df_subset['sz_flag'], df_subset['prob_1'])
            auroc.append(auroc_val)
            
            auprc_val = average_precision_score(df_subset['sz_flag'], df_subset['prob_1'])
            auprc.append(auprc_val)
            
            confidence_intervals = get_ci(df_subset['sz_flag'], df_subset['y_pred'], df_subset['prob_1'])
            auroc_ci.append(confidence_intervals[0])
            auprc_ci.append(confidence_intervals[1])
        else:
            print('iteration',i, 'has only one class')
            auroc.append(np.nan)
            auroc_ci.append((np.nan, np.nan))
            auprc.append(np.nan)
            auprc_ci.append((np.nan, np.nan))
            

        num_patients.append(len(df_subset))
        frac_pos_samples.append(sum(df_subset['sz_flag'])/len(df_subset))
        num_visits.append(np.mean(df_subset['iteration']*10))
        num_visits_ci.append(stats.sem(df_subset['iteration']*10) * stats.t.ppf((1 + 0.95) / 2., len(df_subset)-1))
        
    return auroc, auroc_ci, auprc, auprc_ci, num_patients, frac_pos_samples, num_visits, num_visits_ci

def get_ci_all_table(y_test, y_pred, pred_prob, threshold=0.95):
    """
    gives us 95% CI for auroc, auprc
    """
    rng = np.random.RandomState(seed=44)
    idx = np.arange(y_test.shape[0])

    test_auroc = []
    test_acc = []
    test_sensitivity = []
    test_specificity = []
    test_auprc = []
    test_ppv = []
    for i in tqdm(range(300)):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        if len(set(y_test.iloc[pred_idx])) > 1:
            test_auroc.append(roc_auc_score(y_test.iloc[pred_idx], pred_prob.iloc[pred_idx]))
            test_acc.append(accuracy_score(y_test.iloc[pred_idx], y_pred.iloc[pred_idx]))
            tn, fp, fn, tp = confusion_matrix(y_test.iloc[pred_idx], y_pred.iloc[pred_idx]).ravel()
            test_sensitivity.append(tp/(tp+fn))
            test_specificity.append(tn/(tn+fp))
            test_auprc.append(average_precision_score(y_test.iloc[pred_idx], pred_prob.iloc[pred_idx]))
            test_ppv.append(precision_score(y_test.iloc[pred_idx], y_pred.iloc[pred_idx]))
            

            
    auroc_interval = (np.percentile(test_auroc, 2.5), np.percentile(test_auroc, 97.5))
    acc_interval = (np.percentile(test_acc, 2.5), np.percentile(test_acc, 97.5))
    sensitivity_interval = (np.percentile(test_sensitivity, 2.5), np.percentile(test_sensitivity, 97.5))
    specificity_interval = (np.percentile(test_specificity, 2.5), np.percentile(test_specificity, 97.5))
    auprc_interval = (np.percentile(test_auprc, 2.5), np.percentile(test_auprc, 97.5))
    ppv_interval = (np.percentile(test_ppv, 2.5), np.percentile(test_ppv, 97.5))

    return auroc_interval, acc_interval, sensitivity_interval, specificity_interval, auprc_interval, ppv_interval

def create_table2_row(sample_test, prob_col = 'prob_1', round_col = 'y_pred'):
    auroc_interval, acc_interval, sensitivity_interval, specificity_interval, auprc_interval, ppv_interval = get_ci_all_table(sample_test['sz_flag'], sample_test[round_col], sample_test[prob_col])
    auroc = roc_auc_score(sample_test['sz_flag'], sample_test[prob_col])
    acc = accuracy_score(sample_test['sz_flag'], sample_test[round_col])
    tn, fp, fn, tp = confusion_matrix(sample_test['sz_flag'], sample_test[round_col]).ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    auprc = average_precision_score(sample_test['sz_flag'], sample_test[prob_col])
    ppv = precision_score(sample_test['sz_flag'], sample_test[round_col])
    
    return [auroc, auroc_interval, acc, acc_interval, sensitivity, sensitivity_interval, 
            specificity, specificity_interval, auprc, auprc_interval, ppv, ppv_interval]

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


# supplementary figure 1: performance over time
test_labels['psychosis_diagnosis_date'] = pd.to_datetime(test_labels['psychosis_diagnosis_date'])
test_labels['cutoff_date'] = pd.to_datetime(test_labels['cutoff_date'])
test_labels['censor_date'] = pd.to_datetime(test_labels['censor_date'])

# get time from psychosis dx to cutoff
test_labels['time_to_cutoff'] = (test_labels['cutoff_date']-test_labels['psychosis_diagnosis_date']).dt.days/365

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

matplotlib.rc('font', **font)

plt.figure(figsize=(25, 7))

cutoff_ind = 16
## Forwards iterations, by time
n_years = iterations[0:cutoff_ind]

auprc = np.asarray(auprc[0:cutoff_ind])
auprc_ci = np.asarray(auprc_ci[0:cutoff_ind])

plt.subplot(1,3,1)
# all patients
plt.plot(n_years, auprc, color = 'black', label = 'AUPRC', marker = 'o')
plt.fill_between(n_years, np.abs(auprc_ci.T)[0,:], np.abs(auprc_ci.T)[1,:], color='black', alpha=0.3)

plt.xlabel('Number of years after psychosis diagnosis')
plt.ylabel('Area under precision-recall curve')
plt.title('Overall model performance')
plt.ylim([0.7, 1])


plt.subplot(1,3,2)
## Black Patients
auprc_b = np.asarray(auprc_b[0:cutoff_ind])
auprc_ci_b = np.asarray(auprc_ci_b[0:cutoff_ind])
plt.plot(n_years, auprc_b, label = 'Black Patients', color = 'red', marker = '^')
plt.fill_between(n_years, np.abs(auprc_ci_b.T)[0,:], np.abs(auprc_ci_b.T)[1,:], color='red', alpha=0.2)

## White Patients
auprc_w = np.asarray(auprc_w[0:cutoff_ind])
auprc_ci_w = np.asarray(auprc_ci_w[0:cutoff_ind])
plt.plot(n_years, auprc_w, label = 'White Patients', color = 'blue', marker = 's')
plt.fill_between(n_years, np.abs(auprc_ci_w.T)[0,:], np.abs(auprc_ci_w.T)[1,:], color='blue', alpha=0.2)

## Missing Patients
auprc_missing = np.asarray(auprc_missing[0:cutoff_ind])
auprc_ci_missing = np.asarray(auprc_ci_missing[0:cutoff_ind])
plt.plot(n_years, auprc_missing, label = 'Missing Race Patients', color = 'orange', marker = 's')
plt.fill_between(n_years, np.abs(auprc_ci_missing.T)[0,:], np.abs(auprc_ci_missing.T)[1,:], color='orange', alpha=0.2)

plt.xlabel('Number of years after psychosis diagnosis')
plt.ylabel('Area under precision-recall curve')
plt.title('Model performance by race')
plt.ylim([0.7, 1])
plt.legend()

plt.subplot(1,3,3)
## Female Patients
auprc_f = np.asarray(auprc_f[0:cutoff_ind])
auprc_ci_f = np.asarray(auprc_ci_f[0:cutoff_ind])
plt.plot(n_years, auprc_f, label = 'Women', color = 'red', marker = 'd')
plt.fill_between(n_years, np.abs(auprc_ci_f.T)[0,:], np.abs(auprc_ci_f.T)[1,:], color='red', alpha=0.2)

## Male Patients
auprc_m = np.asarray(auprc_m[0:cutoff_ind])
auprc_ci_m = np.asarray(auprc_ci_m[0:cutoff_ind])
plt.plot(n_years, auprc_m, label = 'Men', color = 'blue', marker = 'v')
plt.fill_between(n_years, np.abs(auprc_ci_m.T)[0,:], np.abs(auprc_ci_m.T)[1,:], color='blue', alpha=0.2)

plt.xlabel('Number of years after psychosis diagnosis')
plt.title('Model performance by gender')
plt.ylabel('Area under precision-recall curve')
plt.ylim([0.7, 1])
plt.legend()
plt.tight_layout()
plt.savefig('xgboost_time_performance.pdf', dpi=300)