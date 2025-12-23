import pandas as pd 
import numpy as np
import sys 

sys.path.append('../')
from eval_utils import * 


metric_functions = {
    'AUROC': roc_auc_score,
    'AUPRC': average_precision_score,
    'Accuracy':accuracy_score,
    'Sensitivity': sensitivity,
    'Specificity': specificity,
    'PPV': precision_score}
binary_metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV']

list_folders = ['transformer_fullhistory_6mopred', 'lstm_fullhistory_6mopred', 'grud_fullhistory_6mopred',
                'transformer_fullhistory_12mopred', 'lstm_fullhistory_12mopred', 'grud_fullhistory_12mopred',
                'transformer_fullhistory_18mopred', 'lstm_fullhistory_18mopred', 'grud_fullhistory_18mopred',
                'transformer_fullhistory_24mopred', 'lstm_fullhistory_24mopred', 'grud_fullhistory_24mopred']

for model_name in list_folders:
    df_val = pd.read_csv(f'../../skip_model_training/models/{model_name}/val_outputs.csv')
    if roc_auc_score(df_val["y_true"], df_val["y_pred"]) > 0.807:
        print(f'{model_name}: {roc_auc_score(df_val["y_true"], df_val["y_pred"])}')
    # print(f'{model_name}:', roc_auc_score(df_val["y_true"], df_val["y_pred"]) > 0.807)


list_folders = ['transformer_fullhistory_long', 'lstm_fullhistory_long', 'grud_fullhistory_long']

for model_name in list_folders:
    df_val = pd.read_csv(f'../../long_model_training/models/{model_name}/val_outputs.csv')
    print(f'{model_name}: {roc_auc_score(df_val["y_true"], df_val["y_pred"])}')
    print(f'{model_name}:', roc_auc_score(df_val["y_true"], df_val["y_pred"]) > 0.8)
