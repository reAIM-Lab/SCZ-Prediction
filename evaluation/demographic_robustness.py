import numpy as np
import os
import pandas as pd
import pyodbc
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import sys
import pickle
from joblib import load, dump
import matplotlib
import xgboost as xgb
from sklearn.metrics import *

sys.path.append('../utils')
from eval_utils import *

data_path = '../prediction_data/'

# read in population dataframe
num_days_prediction = 90
df_pop = pd.read_csv(data_path+"population.csv")
df_pop['psychosis_diagnosis_date'] = pd.to_datetime(df_pop['psychosis_diagnosis_date'], format="%Y-%m-%d")
df_pop['cohort_start_date'] = pd.to_datetime(df_pop['cohort_start_date'])
df_pop = df_pop.loc[(df_pop['cohort_start_date']-df_pop['psychosis_diagnosis_date']).dt.days >= num_days_prediction]

test_labels = pd.read_csv('stored_data/5_21_model_test_output.csv')
test_labels = test_labels.merge(df_pop[['person_id', 'gender_concept_id', 'race_concept_id']], how='left', right_on = 'person_id', left_on = 'person_id')
true_cutoff = load('stored_data/xgb_8_visits_5_21_cutoff')

test_black = test_labels.loc[test_labels['race_concept_id']==8516]
test_white = test_labels.loc[test_labels['race_concept_id']==8527]
test_missing = test_labels.loc[test_labels['race_concept_id']==0]
test_male = test_labels.loc[test_labels['gender_concept_id']==8507]
test_female = test_labels.loc[test_labels['gender_concept_id']==8532]

all_cutoff, all_cutoff_ci, all_pval = bootstrapp_cutoff(test_labels, label_col='sz_flag', prob_col='prob_1', true_cutoff=true_cutoff)
print('All', all_cutoff, all_cutoff_ci)
black_cutoff, black_cutoff_ci, black_pval = bootstrapp_cutoff(test_black, label_col='sz_flag', prob_col='prob_1', true_cutoff=true_cutoff)
print('Black', black_cutoff, black_cutoff_ci)
white_cutoff, white_cutoff_ci, white_pval = bootstrapp_cutoff(test_white, label_col='sz_flag', prob_col='prob_1', true_cutoff=true_cutoff)
print('White', white_cutoff, white_cutoff_ci)
missing_cutoff, missing_cutoff_ci, missing_pval = bootstrapp_cutoff(test_missing, label_col='sz_flag', prob_col='prob_1', true_cutoff=true_cutoff)
print('Missing', missing_cutoff, missing_cutoff_ci)
male_cutoff, male_cutoff_ci, male_pval = bootstrapp_cutoff(test_male, label_col='sz_flag', prob_col='prob_1', true_cutoff=true_cutoff)
print('Male', male_cutoff, male_cutoff_ci)
female_cutoff, female_cutoff_ci, female_pval = bootstrapp_cutoff(test_female, label_col='sz_flag', prob_col='prob_1', true_cutoff=true_cutoff)
print('Female', female_cutoff, female_cutoff_ci)
print('True Cutoff', true_cutoff)
