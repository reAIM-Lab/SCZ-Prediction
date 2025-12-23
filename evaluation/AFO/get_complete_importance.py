import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import torch 
import sys

sys.path.append('../')
from eval_utils import *

def summarize_importance_over_time(afo_load_path, data_columns, time_points=np.arange(10, 73, 5)):
    """
    Summarizes feature importance scores across time points with 95% CI SEM (1.96 * stats.sem).

    Args:
        afo_load_path (str): Base path to AFO outputs (expects files named AFO_outputs_time{t}.pt)
        data_columns (list): Feature names corresponding to importance score dimensions
        time_points (iterable): Time values (default: np.arange(10, 73, 5))

    Returns:
        pd.DataFrame: summary of per-feature means and SEMs across times + overall mean/sem
    """
    all_scores = {}

    for t in time_points:
        results_path = f"{afo_load_path}/AFO_outputs_time{t}.pt"
        results_dict = torch.load(results_path, weights_only=False)
        imp_list = results_dict["importance_scores"]

        # Stack all samples for that time
        imp_array = np.vstack([
            imps.cpu().numpy() if torch.is_tensor(imps) else np.asarray(imps)
            for imps in imp_list
        ])
        all_scores[t] = imp_array

        print(f"Processed time {t}: {imp_array.shape[0]} samples, {imp_array.shape[1]} features")

    # Compute per-time mean and SEM (95% CI)
    summary_data = {"feature": data_columns}

    for t in time_points:
        arr = all_scores[t]
        summary_data[f"time_iter_{t}_mean"] = arr.mean(axis=0)
        summary_data[f"time_iter_{t}_sem"] = 1.96 * stats.sem(arr, axis=0, nan_policy='omit')

    # Combine all times for overall stats
    all_concat = np.vstack(list(all_scores.values()))
    summary_data["overall_mean"] = all_concat.mean(axis=0)
    summary_data["overall_sem"] = 1.96 * stats.sem(all_concat, axis=0, nan_policy='omit')

    df_summary = pd.DataFrame(summary_data)
    print(f"\nOverall summary: {all_concat.shape[0]} total samples across {len(time_points)} time points.")
    return df_summary


if __name__ == "__main__":

    with open(f'{int_data_path}/9_26_mdcd_2dx_fullhistory_du_snomed_colnames', "rb") as fp:   # Unpickling
        data_columns = pickle.load(fp)

    df_summary = summarize_importance_over_time(afo_load_path, data_columns, time_points=np.arange(10, 73, 5))

    print('saving full ranking and importance')
    df_summary.to_csv(f'{afo_load_path}/complete_importance_dfs.csv')
    print(df_summary.shape)
