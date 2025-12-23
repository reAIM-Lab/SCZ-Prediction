import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib
import scipy.stats as stats
from sklearn.metrics import *
import sys
import seaborn as sns

sys.path.append('../')
from eval_utils import * 
"""
Get validation and test performance on all models. 
Ideally only want to show validation?
***WE USE THE NONSPECIFIC CUTOFF PROB
"""

figures_output = ''

metric_functions = {
    'AUROC': roc_auc_score,
    'F1 Score': f1_score,
    'Accuracy':accuracy_score,
    'Sensitivity': sensitivity,
    'Specificity': specificity,
    'PPV': precision_score, 
    'Brier': brier_score_loss}
binary_metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'F1 Score']


# Now get the validation and test performance (overall, first, last) for each model. 
def add_rows_per_model_type(model_name):
    model_path = f'../../skip_model_training/models/{model_name}'
    
    val_output = get_prediction_dates_demos(f'{model_path}/val_outputs.csv', f'{data_path}/population.csv', f'{int_data_path}/hcu_visit_counts.csv')
    test_output = get_prediction_dates_demos(f'{model_path}/test_outputs.csv', f'{data_path}/population.csv', f'{int_data_path}/hcu_visit_counts.csv')

    val_output = val_output.loc[val_output['days_since_start'] >= 365]
    test_output = test_output.loc[test_output['days_since_start'] >= 365]

    cutoff_prob = get_cutoff_prob(val_output['y_true'], val_output['y_pred'])

    # interested in overall performance
    val_dict = {'All Data': val_output}

    # 1. validation performance, test performance [a. at psychosis, b. at final time point, c. overall]
    val_perf = pd.DataFrame()
    test_perf = pd.DataFrame()

    
    val_perf = create_table2_row(val_perf, val_output, 'Validation', 'y_pred', 'y_true', cutoff_prob, metric_functions, binary_metrics)
    val_perf['Model Type'] = model_name

    test_perf = create_table2_row(test_perf, test_output, 'Test', 'y_pred', 'y_true', cutoff_prob, metric_functions, binary_metrics)
    test_perf['Model Type'] = model_name

    return val_perf, test_perf

list_folders = ['transformer_fullhistory_6mopred', 'lstm_fullhistory_6mopred', 'grud_fullhistory_6mopred',
                'transformer_fullhistory_12mopred', 'lstm_fullhistory_12mopred', 'grud_fullhistory_12mopred',
                'transformer_fullhistory_18mopred', 'lstm_fullhistory_18mopred', 'grud_fullhistory_18mopred',
                'transformer_fullhistory_24mopred', 'lstm_fullhistory_24mopred', 'grud_fullhistory_24mopred']
list_val_perfs = []
list_test_perfs = []
for model_name in list_folders:
    val_t2, test_t2 = add_rows_per_model_type(model_name)
    list_val_perfs.append(val_t2)
    list_test_perfs.append(test_t2)
val_perf = pd.concat(list_val_perfs)
val_perf.to_csv(f'{figures_output}/supplementary_{dataset}_model_validation_performance_allmodels.csv')

test_perf = pd.concat(list_test_perfs)
test_perf.to_csv(f'{figures_output}/supplementary_{dataset}_model_test_performance_allmodels.csv')
