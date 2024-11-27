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


class ModelOutput:
    def __init__(self, loader, testing_clf, max_visits):
        self.loader = loader
        self.testing_clf = testing_clf
        self.max_visits = max_visits
        
    def get_output_vals(self, timestep_ind = -1):
        """
        This takes in a trained classifier and a dataloader object 
        and returns the predicted probabilities and the actual outputs
        (y_pred and y_true)

        by default, it will return the value based on the full x trajectory
        but if timestep_ind is not -1, then it will return the value based on
        the x trajectory from 0 to timestep_ind + 1 
        """
        y_pred = []
        y_true = []
        device = torch.device("cuda:0") 
        self.testing_clf.to(device)
        for i, (x, y) in enumerate(self.loader):
            if timestep_ind != -1:
                x = x[:, 0:timestep_ind+1, :]

            x, y = torch.Tensor(x.float()).to(device), torch.Tensor(y.float()).to(device)
            out = self.testing_clf(x)
            y_pred.append(out.detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())

        y_pred_flattened = np.concatenate(y_pred).ravel()
        y_true_flattened = np.concatenate(y_true).ravel()

        return y_true_flattened, y_pred_flattened

    def get_all_output_vals(self):
        """
        This takes in a trained classifier and a dataloader object 
        and returns the predicted probabilities and the actual outputs
        (y_pred and y_true)

        It iterates through each timestep and returns the probability of 
        a positive output at each step
        """

        list_probs = []
        list_true = []
        for i in tqdm(range(0, self.max_visits)):
            y_true, y_pred = self.get_output_vals(timestep_ind = i)
            list_true.append(y_true)
            list_probs.append(y_pred)

        y_true = np.asarray(list_true)
        y_pred = np.asarray(list_probs)
        return y_true, y_pred
    
    def get_cutoff_prob(self, y_true, y_pred, stored_data_path = None, save_filename = None):
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
    
    def create_output_df(self, y_true, y_pred, df_pop, df_split, split_type):
        """
        y_true and y_pred should be the output of get_all_output_vals
            Their shape should be (n_iters, n_patients)

        df_pop should have the n_visits column
        df_split should have a split column where split_type is one e
        """
        # reshape to be person * iter, 1
        reshaped_y_true = y_true.T.reshape(-1)
        reshaped_y_pred = y_pred.T.reshape(-1)

        # get patient ids and sort them from smallest to largest
        pids = df_split.loc[df_split['split']==split_type, 'person_id'].values
        pids.sort()

        # create output with n_patients * n_iters
        df_outputs = pd.DataFrame(data = np.repeat(pids, y_true.shape[0]), columns = ['person_id'])
        df_outputs['iteration'] = np.tile(np.arange(1,y_true.shape[0]+1), len(pids))
        df_outputs['y_true'] = reshaped_y_true
        df_outputs['y_pred'] = reshaped_y_pred
        # checked by merge that the y_trues line up
        df_outputs = df_outputs.merge(df_pop[['person_id', 'n_visits']], how='inner', left_on = 'person_id', right_on = 'person_id')

        # restrict to only where there are no all-0 iterations
        df_outputs = df_outputs.loc[df_outputs['iteration'] <= df_outputs['n_visits']]

        return df_outputs
    
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
def ppv_score(y_true, y_pred_binary):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    return tp/(tp+fp)




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

def run_model_sequentially(model, input_data, save_cols, device = torch.device("cuda:0")):
    """
    input_data is a dataframe that is multi-indexed on person_id and timestep (both ranked and absolute).
    The ranked index tells us how many prior timesteps are recorded and the absolute iteration
    tells us if this was pre- (negative) or post (>=0) psychosis dx
    
    the model returns a dataframe that has patients as the index and the timestep as the columns, 
    and the output probability for each patient/timestep. will be NaN if the patient doesn't have a 
    datapoint at that timestep
    
    """
    ranked_iters = input_data.index.get_level_values(1).unique()
    prob_outcome_df = pd.DataFrame(index=input_data.index.get_level_values(0).unique(), columns = ranked_iters)
    
    # get dataset
    for ind in ranked_iters:
        patients_in_iter = input_data.loc[(input_data.index.get_level_values(1) == ind)&(input_data.index.get_level_values(2) >= 0)].index.get_level_values(0)
        iter_df = input_data.loc[(input_data.index.get_level_values(0).isin(patients_in_iter))&(input_data.index.get_level_values(1) <= ind)]

        iter_mat = iter_df.values.reshape(len(patients_in_iter), int(ind), len(save_cols))


        # run model on this data
        minitest_dataset = torch.utils.data.TensorDataset(torch.Tensor(iter_mat), torch.Tensor(np.zeros(len(patients_in_iter))))
        minitest_loader = torch.utils.data.DataLoader(minitest_dataset, batch_size=256)
        pred_ys = []
        for i, (x, y) in enumerate(minitest_loader):
            x, y = torch.Tensor(x.float()).to(device), torch.Tensor(y.float()).to(device)
            out = model(x)

            pred_ys.append(out.detach().cpu().numpy())
        prob_outcome_df.loc[patients_in_iter, ind] = np.concatenate(pred_ys).ravel()


    return prob_outcome_df