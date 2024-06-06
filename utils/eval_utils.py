import numpy as np
import os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from datetime import datetime
import sys
import gc
import pyodbc
from scipy.sparse import *
import torch
from sklearn.metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle 
import random
import math
from joblib import dump, load

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

# get cutoff with boostrapped samples from the test data
def get_cutoff(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    idx = np.argmax(tpr - fpr)
    cutoff_prob = thresholds[idx]
    return cutoff_prob

def bootstrapp_cutoff(df, label_col, prob_col, true_cutoff):
    rng = np.random.RandomState(seed=44)
    idx = np.arange(df.shape[0])

    cutoff_prob_list = []
    for i in range(300):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        labels = df.iloc[pred_idx][label_col]
        probs = df.iloc[pred_idx][prob_col]
        cutoff_prob_list.append(get_cutoff(labels, probs))
    
    cutoff_val = get_cutoff(df[label_col], df[prob_col])
    cutoff_interval = (np.percentile(cutoff_prob_list, 2.5), np.percentile(cutoff_prob_list, 97.5))
    
    
    ttest_results = stats.ttest_1samp(cutoff_prob_list, true_cutoff)
    ttest_ci = ttest_results.confidence_interval(confidence_level=0.95)
    return cutoff_val, cutoff_interval, ttest_results.pvalue

# SHAPLEY VALUES
def get_shap_values(df_test_filename, model_filename):
    df_test = pd.read_csv('stored_data/' + df_test_filename)
    df_test.drop(['Unnamed: 0', 'iteration', 'person_id', 'sz_flag'], axis=1, inplace=True)
    with open('models/' + model_filename, 'rb') as f:
        testing_clf = pickle.load(f)

    save_cols = list(df_test.columns)

    X_test = df_test.values
    # approximate shapley values using XGBoost
    X_dmatrix = xgb.DMatrix(X_test)
    xgb_model = testing_clf.get_booster()
    model_pred_detail = xgb_model.predict(X_dmatrix, pred_contribs=True)
    shap_values = model_pred_detail[:,0:-1] # ignore bias term
    shap_values = pd.DataFrame(shap_values, columns = save_cols)

    # get shapley values associated with top features
    mean_shap_values = shap_values.mean(axis=0)
    mean_shap_values = pd.DataFrame(mean_shap_values, columns = ['Mean SHAP'], index=save_cols)
    mean_shap_values['Abs SHAP'] = np.abs(mean_shap_values['Mean SHAP'])
    top_mean_shap = mean_shap_values.sort_values('Abs SHAP', ascending=False)[0:10]
    top_shap_values = shap_values[top_mean_shap.index]
    
    return shap_values, top_shap_values

# function for showing overall model performance
def print_performance(X, y, cutoff_prob):
    probs = grid.best_estimator_.predict_proba(X)
    probs = probs[:,1]

    y_pred = (probs >= cutoff_prob)*1
    print('Accuracy' , accuracy_score(y, y_pred))
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    print('Sensitivity', (tp/(tp+fn)))
    print('Specificity', (tn/(tn+fp)))
    print('PPV', precision_score(y, y_pred))

    print('AUPRC', average_precision_score(y, probs))
    print('AUROC', roc_auc_score(y, probs))