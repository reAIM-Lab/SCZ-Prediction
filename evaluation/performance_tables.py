import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib
import scipy.stats as stats
from sklearn.metrics import *
import sys
import seaborn as sns

from eval_utils import * 

model_path = f'../skip_model_training/models/{model_name}'
features_name = 'fullfeatures'
output_folder = f'figures_skiptrain_6mo_{features_name}/'

metric_functions = {
    'AUROC': roc_auc_score,
    'F1 Score': f1_score,
    'Accuracy':accuracy_score,
    'Sensitivity': sensitivity,
    'Specificity': specificity,
    'PPV': precision_score, 
    'Brier': brier_score_loss}
binary_metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'F1 Score']

test_output = pd.read_csv(f'{output_folder}/test_outputs.csv')
test_output = test_output.loc[test_output['days_since_start'] >= 365]
print(test_output['cutoff_prob_overall'].unique())
cutoff_prob = test_output['cutoff_prob_overall'].unique()[0]
print(cutoff_prob)

psychosis_test_output = test_output.merge(
                        test_output.groupby("person_id")["tte"].max().reset_index(), 
                        on=["person_id", "tte"], how="inner")

final_test_output = test_output.merge(
                        test_output.groupby("person_id")["tte"].min().reset_index(), 
                        on=["person_id", "tte"], how="inner")

test_dict = {'All Data': test_output, 'First Prediction': psychosis_test_output, 'Final Prediction': final_test_output}


# 2. test performance by race/gender [a. at psychosis, b. at final time point, c. overall]
list_demo_test_perf = []
for perf_type in test_dict.keys():
    temp_test = test_dict[perf_type]
    demographic_test_perf = pd.DataFrame()
    demographic_test_perf = create_table2_row(demographic_test_perf, temp_test, 'All', 'y_pred', 'y_true', cutoff_prob, metric_functions, binary_metrics)
    demographic_test_perf = create_table2_row(demographic_test_perf, temp_test.loc[temp_test['race_concept_id']==8516], 'Black', 'y_pred', 'y_true', cutoff_prob, metric_functions, binary_metrics)
    demographic_test_perf = create_table2_row(demographic_test_perf, temp_test.loc[temp_test['race_concept_id']==8527], 'White', 'y_pred', 'y_true', cutoff_prob, metric_functions, binary_metrics)
    demographic_test_perf = create_table2_row(demographic_test_perf, temp_test.loc[temp_test['race_concept_id']==0], 'Missing Race', 'y_pred', 'y_true', cutoff_prob, metric_functions, binary_metrics)
    demographic_test_perf = create_table2_row(demographic_test_perf, temp_test.loc[temp_test['gender_concept_id']==8532], 'Female', 'y_pred', 'y_true', cutoff_prob, metric_functions, binary_metrics)
    demographic_test_perf = create_table2_row(demographic_test_perf, temp_test.loc[temp_test['gender_concept_id']==8507], 'Male', 'y_pred', 'y_true', cutoff_prob, metric_functions, binary_metrics)
    demographic_test_perf['Performance Timing'] = perf_type
    list_demo_test_perf.append(demographic_test_perf)

demographic_test_perf = pd.concat(list_demo_test_perf)
demographic_test_perf.to_csv(f'{output_folder}/demographic_performance.csv')

# 3. test performance by initial psychosis dx [a. at psychosis, b. at final time point, c. overall]
substance_induced_psychosis = [372607, 442582, 434900, 440987, 374317,
                               4024296, 4272313, 4097389, 443559, 
                               4176286, 4137955, 443930, 4155336]
bipolar_psychosis = [4220617, 439256, 436386, 439246, 35622934,
                    35610112, 439262]
psychosis_nos = [436073]
reactive_psychosis = [435520, 441540, 438737, 440983, 435237, 
                      4182683, 436952]
delusions = [432590, 440684]
depressive_psychosis = [438406, 434911]
all_psychosis_codes = substance_induced_psychosis+bipolar_psychosis+psychosis_nos+reactive_psychosis+delusions+depressive_psychosis
df_first_psychosis = pd.read_csv(f'{data_path}/first_psychosis_dx_per_pt.csv')

list_psychosis_specific_perfs = []
for perf_type in test_dict.keys():
    table_df = pd.DataFrame()
    table_df = psychosis_specific_eval(substance_induced_psychosis, df_first_psychosis, test_dict[perf_type], table_df, 'Substance-Induced Psychosis')
    table_df = psychosis_specific_eval(bipolar_psychosis, df_first_psychosis, test_dict[perf_type], table_df, 'Bipolar Disorder')
    table_df = psychosis_specific_eval(psychosis_nos, df_first_psychosis, test_dict[perf_type], table_df, 'Psychosis NOS')
    table_df = psychosis_specific_eval(reactive_psychosis, df_first_psychosis, test_dict[perf_type], table_df,'Reactive Psychosis')
    table_df = psychosis_specific_eval(delusions, df_first_psychosis, test_dict[perf_type], table_df,'Delusional Disorder')
    table_df = psychosis_specific_eval(depressive_psychosis, df_first_psychosis, test_dict[perf_type], table_df, 'Depression with psychosis')
    table_df = psychosis_specific_eval(all_psychosis_codes, df_first_psychosis, test_dict[perf_type], table_df, 'All Psychosis')
    table_df['Performance Timing'] = perf_type
    list_psychosis_specific_perfs.append(table_df)
psychosis_performance = pd.concat(list_psychosis_specific_perfs)
psychosis_performance.to_csv(f'{output_folder}/performance_by_psychosis.csv')

 