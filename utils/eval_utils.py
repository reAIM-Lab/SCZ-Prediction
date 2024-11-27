import numpy as np
import os
import pandas as pd
import pyodbc
import time
import scipy.stats as stats
from datetime import datetime
from tqdm import tqdm
from collections import Counter
from datetime import datetime
import sys
import gc
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle 
import random
import math
from joblib import dump, load
# from balancers import *
# from tools import *



    
def get_cutoff_prob(y_true, y_pred, stored_data_path = None, save_filename = None):
    """
    This function takes in the true outputs and the predicted probabilities
    (from the validation dataset) and returns the optimal cutoff probability

    if save_filename is not none, then stored_data_path should also be defined 
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    idx = np.argmax(tpr - fpr)
    cutoff_prob = thresholds[idx]
    print('Cutoff Probability:', cutoff_prob)

    if save_filename != None: 
        dump(cutoff_prob, stored_data_path+save_filename)

    return cutoff_prob
    
# function for showing overall model performance
def print_performance(X, y, model, cutoff_prob):
    probs = model.predict_proba(X)
    probs = probs[:,1]

    y_pred = (probs >= cutoff_prob)*1
    print('Accuracy' , accuracy_score(y, y_pred))
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    print('Sensitivity', (tp/(tp+fn)))
    print('Specificity', (tn/(tn+fp)))
    print('PPV', precision_score(y, y_pred))

    print('AUPRC', average_precision_score(y, probs))
    print('AUROC', roc_auc_score(y, probs))

def evaluate_metrics(y_true, y_pred, cutoff_prob, dict_metrics, binary_funcs):
    """
    Evaluate a list of metrics for the given predictions and true values.

    Parameters:
    y_pred (array-like): Predicted values
    y_true (array-like): True values
    cutoff (float): Cutoff value for binary classification
    dict_metrics (dict): Dictionary of metric names and their corresponding functions
    binary_funcs (list): list of function names that take in y_pred_binary instead of y_pred (probability)

    Returns:
    dict: Dictionary of metric results
    """
    if cutoff_prob == None:
        # if we are dealing with postprocessing and don't have the cutoff
        y_pred_binary = y_pred.copy()
    else:
        # Apply cutoff to y_pred to get binary predictions
        y_pred_binary = (y_pred >= cutoff_prob).astype(int)
    
    results = {}
    for metric_name, metric_func in dict_metrics.items():
        if metric_name in binary_funcs:
            results[metric_name] = metric_func(y_true, y_pred_binary)
        else: 
            results[metric_name] = metric_func(y_true, y_pred)
    
    return results

def sensitivity(y_true, y_pred_binary):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    return tp/(tp+fn)
def specificity(y_true, y_pred_binary):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    return tn/(tn+fp)




def bootstrap_evaluation(y_true, y_pred, cutoff_prob, dict_metrics, binary_funcs, n_bootstraps = 300, alpha = 0.05):
    #alpha = 0.05 gives us 95% CI
    # make y_true and y_pred arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # get sampling
    rng = np.random.RandomState(seed=44)
    idx = np.arange(y_true.shape[0])

    # get blank dictionary for results
    metrics_results = {metric: [] for metric in dict_metrics.keys()}

    for i in tqdm(range(0, n_bootstraps)):
        subsample_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        y_true_subsample = y_true[subsample_idx]
        y_pred_subsample = y_pred[subsample_idx]
        
        if len(set(y_true_subsample)) > 1:
            subsample_results = evaluate_metrics(y_true_subsample, y_pred_subsample, cutoff_prob, dict_metrics, binary_funcs)

            for metric_name, metric_value in subsample_results.items():
                metrics_results[metric_name].append(metric_value)
            
    # get summary stats
    summary_stats = {}
    for metric_name, values in metrics_results.items():
        mean = np.mean(values)
        ci_low = np.percentile(values, 100 * alpha / 2)
        ci_high = np.percentile(values, 100 * (1 - alpha / 2))
        
        summary_stats[metric_name] = {
            'mean': mean,
            'CI_low': ci_low,
            'CI_high': ci_high
        }
    
    summary_df = pd.DataFrame(summary_stats).T
    return summary_df

def create_table2_row(table2, subset_df, subset_name, prob_col, cutoff_prob, dict_metrics, binary_funcs):
    # this is to get a cosmetically appropriate table 2
    eval_df = bootstrap_evaluation(subset_df['sz_flag'], subset_df[prob_col], cutoff_prob, dict_metrics, binary_funcs, n_bootstraps = 300, alpha = 0.05)
    eval_df = pd.DataFrame(index=eval_df.index, columns = eval_df.columns, data = np.round(eval_df.values, 3))
    for i in eval_df.index:
        table2.loc[subset_name, i] = str(eval_df.loc[i, 'mean'])+' '+str(tuple(eval_df.loc[i, ['CI_low', 'CI_high']]))
    return table2

# postprocessing: equalized odds
def get_subgroup_tpr_fpr(demo_df, demo_column, confusion_col1, confusion_col2):
    
    list_tprs = []
    list_fprs = []
    for a in demo_df[demo_column].unique():
        s0 = demo_df.loc[demo_df[demo_column]==a]
        s0_tn = len(s0.loc[(s0[confusion_col1] == 0) & (s0[confusion_col2] == 0)])
        s0_tp = len(s0.loc[(s0[confusion_col1] == 1) & (s0[confusion_col2] == 1)])
        s0_fn = len(s0.loc[(s0[confusion_col1] == 1) & (s0[confusion_col2] == 0)])
        s0_fp = len(s0.loc[(s0[confusion_col1] == 0) & (s0[confusion_col2] == 1)])
        
        list_tprs.append(s0_tp / (s0_tp + s0_fn))
        list_fprs.append(s0_fp / (s0_fp + s0_tn))
    
    return list_tprs, list_fprs

    
def postprocess_data(val_df, test_df, true_col, prob_col, sensitive_col, type_adjustment,):
    
    postprocess_optimizer = BinaryBalancer(y=val_df[true_col].values, 
                                           y_=val_df[prob_col].values, 
                                           a=val_df[sensitive_col].values, summary=True)
    postprocess_optimizer.adjust(goal=type_adjustment, summary=True)
    
    y_val_adj = postprocess_optimizer.predict(y_ = val_df[prob_col].values, a = val_df[sensitive_col].values)
    y_test_adj = postprocess_optimizer.predict(y_ = test_df[prob_col].values, a = test_df[sensitive_col].values)
        
    # pre-adjustment test:
    list_pre_tprs, list_pre_fprs = get_subgroup_tpr_fpr(test_df, sensitive_col, true_col, 'y_pred_normal')
    
    test_df['adj_y'] = np.asarray(y_test_adj)
    list_post_tprs, list_post_fprs = get_subgroup_tpr_fpr(test_df, sensitive_col, true_col, 'adj_y')

    # TPR
    print('Test, pre-processed TPRs:', np.round(list_pre_tprs, 4))
    print('Test, post-processed TPRs:', np.round(list_post_tprs, 4))
    
    # FPR
    print('Test, pre-processed FPRs:', np.round(list_pre_fprs, 4))
    print('Test, post-processed FPRs:', np.round(list_post_fprs, 4))
    
    return y_val_adj, y_test_adj



