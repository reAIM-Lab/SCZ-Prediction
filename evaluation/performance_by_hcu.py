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
feature_type = 'fullfeatures'
output_folder = f'figures_skiptrain_6mo_{feature_type}'

test_output = pd.read_csv(f'{output_folder}/test_outputs.csv')
test_hcu = pd.read_csv(f'../figures/hcu_grud_fullhistory_long/test_outputs_with_utilization.csv')
test_hcu.drop(['y_pred'], axis=1, inplace=True)

test_hcu = test_hcu.loc[test_hcu['days_since_start'] >= 365]
test_hcu = test_hcu.merge(test_output[['person_id', 'date_prediction', 'y_binary_pred']], how = 'inner', on = ['person_id', 'date_prediction'])

trainval_hcu = pd.read_csv(f'../figures/hcu_grud_fullhistory_long/trainval_outputs_with_utilization.csv')
trainval_hcu = trainval_hcu.loc[trainval_hcu['days_since_start']>=365]

list_visitnames = ['pharmacy', 'outpatient',  'inpatientedcombo', 'allvisits']
# get HCU deciles
decile_dict = {}
for name in list_visitnames:
    for prefix in ['all', 'mh', 'nonmh']:
        for suffix in ['hcu', '1yr_visits']:
            col = f'{prefix}_{name}_{suffix}'
            decile_dict[col] = np.percentile(trainval_hcu[col], np.linspace(0, 100, 10))

test_hcu['Prediction Type'] = 'TN'
test_hcu.loc[(test_hcu['y_true']==1) & (test_hcu['y_binary_pred'] == 1), 'Prediction Type'] = 'TP'
test_hcu.loc[(test_hcu['y_true']==1) & (test_hcu['y_binary_pred'] == 0), 'Prediction Type'] = 'FN'
test_hcu.loc[(test_hcu['y_true']==0) & (test_hcu['y_binary_pred'] == 1), 'Prediction Type'] = 'FP'

def plot_hcu_by_prediction_type(df):
    # Define your setting groups
    setting_groups = {
        "All Visits": ["all_allvisits_hcu", "mh_allvisits_hcu", "nonmh_allvisits_hcu"],
        "Pharmacy": ["all_pharmacy_hcu", "mh_pharmacy_hcu", "nonmh_pharmacy_hcu"],
        "Inpatient and ED": ["all_inpatientedcombo_hcu", "mh_inpatientedcombo_hcu", "nonmh_inpatientedcombo_hcu"],
        "Outpatient": ["all_outpatient_hcu", "mh_outpatient_hcu", "nonmh_outpatient_hcu"]
    }

    pred_types = ["TP", "FP", "TN", "FN"]
    colors = {
        "TP": "#4daf4a",  # green
        "FP": "#e41a1c",  # red
        "TN": "orange",  # 
        "FN": "#377eb8"   # blue
    }

    fig, axes = plt.subplots(1, 4, figsize=(30, 8))
    for ax, (setting_name, cols) in zip(axes, setting_groups.items()):
        means = []
        sems = []
        for col in cols:
            group_stats = df.groupby("Prediction Type")[col].agg(["mean", "sem"]).reindex(pred_types)
            means.append(group_stats["mean"].values)
            sems.append(group_stats["sem"].values)

        means = np.array(means)  # shape (3, 4)
        sems = np.array(sems)

        x = np.arange(len(cols))
        width = 0.18

        for i, ptype in enumerate(pred_types):
            ax.bar(
                x + i * width - 1.5 * width,
                means[:, i],
                yerr=sems[:, i],
                width=width,
                color=colors[ptype],
                label=ptype if setting_name == "All Visits" else None,  # one legend
                capsize=4,
            )  
        ax.set_xticks(x)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(22)
        ax.set_xticklabels(["All", "Mental\nHealth", "Medical\nHealth"], fontsize=22, fontweight="bold")
        ax.set_title(setting_name, fontsize=22, fontweight="bold")
        ax.set_xlabel("")

    axes[0].set_ylabel("Percentile of utilization", fontsize=22, fontweight="bold")
    fig.suptitle("Healthcare utilization differences across prediction types", fontsize=24, fontweight="bold")
    axes[-1].legend(pred_types, loc="lower right", fontsize = 22)
    plt.tight_layout()
    return fig

def replace_with_percentiles(test_outputs, trainval_outputs, list_columns):
    test_outputs = test_outputs.copy()
    
    for col in list_columns:
        # Get the training distribution (drop NaNs)
        train_values = trainval_outputs[col].dropna().values
        
        # Sort training values once
        sorted_train = np.sort(train_values)
        
        # Compute percentile rank for each test value
        ranks = np.searchsorted(sorted_train, test_outputs[col].values, side='right')
        percentiles = 100 * ranks / len(sorted_train)
        
        # Replace column with percentile values
        test_outputs[col] = percentiles
    
    return test_outputs

list_cols_percentiles = ["all_allvisits_hcu", "mh_allvisits_hcu", "nonmh_allvisits_hcu",
                        "all_pharmacy_hcu", "mh_pharmacy_hcu", "nonmh_pharmacy_hcu",
                        "all_inpatientedcombo_hcu", "mh_inpatientedcombo_hcu", "nonmh_inpatientedcombo_hcu",
                        "all_outpatient_hcu", "mh_outpatient_hcu", "nonmh_outpatient_hcu"]
test_hcu = replace_with_percentiles(test_hcu, trainval_hcu, list_cols_percentiles)
fig = plot_hcu_by_prediction_type(test_hcu)
fig.savefig(f"{output_folder}/performance_by_hcu_absolute.pdf", dpi=300)
