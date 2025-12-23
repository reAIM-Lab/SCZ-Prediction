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
featuretype = 'fullfeatures'
output_folder = f'figures_skiptrain_6mo_{featuretype}/'

test_output = pd.read_csv(f'{output_folder}/test_outputs.csv')
test_output = test_output.loc[test_output['days_since_start'] >= 365]

# Figure 1: performance as a function of time since psychosis
# A: Overall, B: By Gender, C: By Race
list_times_since_psychosis = np.arange(0, test_output['days_since_psychosis'].max()+1, 90)

list_eval_psych_overall = calc_subtime_performances(test_output, list_times_since_psychosis, 'days_since_psychosis', cutoff_prob = None, y_pred_binary = test_output['y_binary_pred'])
black_test = test_output.loc[test_output['race']=='Black']
list_eval_psych_black = calc_subtime_performances(black_test, list_times_since_psychosis, 'days_since_psychosis', cutoff_prob = None, y_pred_binary = black_test['y_binary_pred'])
white_test = test_output.loc[test_output['race']=='White']
list_eval_psych_white = calc_subtime_performances(white_test, list_times_since_psychosis, 'days_since_psychosis', cutoff_prob = None, y_pred_binary = white_test['y_binary_pred'])
missing_test = test_output.loc[test_output['race']=='Missing Race']
list_eval_psych_missing = calc_subtime_performances(missing_test, list_times_since_psychosis, 'days_since_psychosis', cutoff_prob = None, y_pred_binary = missing_test['y_binary_pred'])
female_test = test_output.loc[test_output['gender']=='Women']
list_eval_psych_women = calc_subtime_performances(female_test, list_times_since_psychosis, 'days_since_psychosis', cutoff_prob = None, y_pred_binary = female_test['y_binary_pred'])
male_test = test_output.loc[test_output['gender']=='Men']
list_eval_psych_men = calc_subtime_performances(male_test, list_times_since_psychosis, 'days_since_psychosis', cutoff_prob = None, y_pred_binary = male_test['y_binary_pred'])

list_times_since_psychosis = list_times_since_psychosis[0:-1]/365

metric = 'AUROC'
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(26,7))
font = {'weight' : 'bold',
        'size'   : 24}
matplotlib.rc('font', **font)
x_label_coord = -0
y_label_coord = 1.02

plt.subplot(1,3,1)
matplotlib.rc('font', **font)
ax = axes[0]
plt.plot(list_times_since_psychosis, [df.loc[metric, "mean"] for df in list_eval_psych_overall], c = 'k')
plt.fill_between(list_times_since_psychosis, [df.loc[metric, "CI_low"] for df in list_eval_psych_overall], [df.loc[metric, "CI_high"] for df in list_eval_psych_overall], color = 'k', alpha=0.5)
plt.xlabel('Years since psychosis', fontdict=font)
plt.ylabel('AUROC', fontdict=font)
plt.xticks(fontsize=font["size"], fontweight=font["weight"])
plt.yticks(fontsize=font["size"], fontweight=font["weight"])
plt.title('Overall performance')
ax.text(x_label_coord, y_label_coord, 'A', transform=ax.transAxes, size=24, weight='bold')
plt.ylim([0.75, 0.85])

plt.subplot(1,3,2)
matplotlib.rc('font', **font)
ax = axes[1]
plt.plot(list_times_since_psychosis, [df.loc[metric, "mean"] for df in list_eval_psych_black], c = 'b', label = 'Black')
plt.fill_between(list_times_since_psychosis, [df.loc[metric, "CI_low"] for df in list_eval_psych_black], [df.loc[metric, "CI_high"] for df in list_eval_psych_black], color = 'b', alpha=0.3)

plt.plot(list_times_since_psychosis, [df.loc[metric, "mean"] for df in list_eval_psych_white], c = 'r', label = 'White')
plt.fill_between(list_times_since_psychosis, [df.loc[metric, "CI_low"] for df in list_eval_psych_white], [df.loc[metric, "CI_high"] for df in list_eval_psych_white], color = 'r', alpha=0.3)

plt.plot(list_times_since_psychosis, [df.loc[metric, "mean"] for df in list_eval_psych_missing], c = 'y', label = 'Missing Race')
plt.fill_between(list_times_since_psychosis, [df.loc[metric, "CI_low"] for df in list_eval_psych_missing], [df.loc[metric, "CI_high"] for df in list_eval_psych_missing], color = 'y', alpha=0.3)
plt.xlabel('Years since psychosis', fontdict=font)
plt.ylabel('AUROC', fontdict=font)
plt.xticks(fontsize=font["size"], fontweight=font["weight"])
plt.yticks(fontsize=font["size"], fontweight=font["weight"])
plt.title('Performance across races')
ax.text(x_label_coord, y_label_coord, 'B', transform=ax.transAxes, size=24, weight='bold')
matplotlib.rc('font', **font)
plt.ylim([0.75, 0.85])
plt.legend(loc='upper right')

plt.subplot(1,3,3)
matplotlib.rc('font', **font)
ax = axes[2]
plt.plot(list_times_since_psychosis, [df.loc[metric, "mean"] for df in list_eval_psych_women], c = 'b', label = 'Women')
plt.fill_between(list_times_since_psychosis, [df.loc[metric, "CI_low"] for df in list_eval_psych_women], [df.loc[metric, "CI_high"] for df in list_eval_psych_women], color = 'b', alpha=0.3)

plt.plot(list_times_since_psychosis, [df.loc[metric, "mean"] for df in list_eval_psych_men], c = 'r', label = 'Men')
plt.fill_between(list_times_since_psychosis, [df.loc[metric, "CI_low"] for df in list_eval_psych_men], [df.loc[metric, "CI_high"] for df in list_eval_psych_men], color = 'r', alpha=0.3)

plt.xlabel('Years since psychosis', fontdict=font)
plt.ylabel('AUROC', fontdict=font)
plt.xticks(fontsize=font["size"], fontweight=font["weight"])
plt.yticks(fontsize=font["size"], fontweight=font["weight"])
plt.title('Performance across genders')
ax.text(x_label_coord, y_label_coord, 'C', transform=ax.transAxes, size=24, weight='bold')
matplotlib.rc('font', **font)
plt.ylim([0.72, 0.82])
plt.legend(loc = 'lower right')

plt.tight_layout()
plt.savefig(f'{output_folder}/auroc_over_time_since_psychosis_withdemo.pdf', dpi = 300)

# Figure 2: performance as a function of time since first visit
# A: Overall, B: By Gender, C: By Race
list_times_since_start = np.arange(365, np.percentile(test_output['days_since_start'], 90), 180)
print(list_times_since_start)

print(len(test_output))
list_eval_start_overall = calc_subtime_performances(test_output, list_times_since_start, 'days_since_start', cutoff_prob = None, y_pred_binary = test_output['y_binary_pred'])
list_eval_start_black = calc_subtime_performances(black_test, list_times_since_start, 'days_since_start', cutoff_prob = None, y_pred_binary = black_test['y_binary_pred'])
list_eval_start_white = calc_subtime_performances(white_test, list_times_since_start, 'days_since_start', cutoff_prob = None, y_pred_binary = white_test['y_binary_pred'])
list_eval_start_missing = calc_subtime_performances(missing_test, list_times_since_start, 'days_since_start', cutoff_prob = None, y_pred_binary = missing_test['y_binary_pred'])
list_eval_start_women = calc_subtime_performances(female_test, list_times_since_start, 'days_since_start', cutoff_prob = None, y_pred_binary = female_test['y_binary_pred'])
list_eval_start_men = calc_subtime_performances(male_test, list_times_since_start, 'days_since_start', cutoff_prob = None, y_pred_binary = male_test['y_binary_pred'])

list_times_since_start = list_times_since_start[0:-1]/365

metric = 'AUROC'
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(26,7))
font = {'weight' : 'bold',
        'size'   : 24}
matplotlib.rc('font', **font)
x_label_coord = -0
y_label_coord = 1.02

plt.subplot(1,3,1)
matplotlib.rc('font', **font)
ax = axes[0]
plt.plot(list_times_since_start, [df.loc[metric, "mean"] for df in list_eval_start_overall], c = 'k')
plt.fill_between(list_times_since_start, [df.loc[metric, "CI_low"] for df in list_eval_start_overall], [df.loc[metric, "CI_high"] for df in list_eval_start_overall], color = 'k', alpha=0.5)
plt.xlabel('Years since first visit', fontdict=font)
plt.ylabel('AUROC', fontdict=font)
plt.xticks(fontsize=font["size"], fontweight=font["weight"])
plt.yticks(fontsize=font["size"], fontweight=font["weight"])
plt.title('Overall performance')
ax.text(x_label_coord, y_label_coord, 'A', transform=ax.transAxes, size=24, weight='bold')
plt.ylim([0.70, 0.85])

plt.subplot(1,3,2)
matplotlib.rc('font', **font)
ax = axes[1]
plt.plot(list_times_since_start, [df.loc[metric, "mean"] for df in list_eval_start_black], c = 'b', label = 'Black')
plt.fill_between(list_times_since_start, [df.loc[metric, "CI_low"] for df in list_eval_start_black], [df.loc[metric, "CI_high"] for df in list_eval_start_black], color = 'b', alpha=0.3)

plt.plot(list_times_since_start, [df.loc[metric, "mean"] for df in list_eval_start_white], c = 'r', label = 'White')
plt.fill_between(list_times_since_start, [df.loc[metric, "CI_low"] for df in list_eval_start_white], [df.loc[metric, "CI_high"] for df in list_eval_start_white], color = 'r', alpha=0.3)

plt.plot(list_times_since_start, [df.loc[metric, "mean"] for df in list_eval_start_missing], c = 'y', label = 'Missing Race')
plt.fill_between(list_times_since_start, [df.loc[metric, "CI_low"] for df in list_eval_start_missing], [df.loc[metric, "CI_high"] for df in list_eval_start_missing], color = 'y', alpha=0.3)
plt.xlabel('Years since first visit', fontdict=font)
plt.ylabel('AUROC', fontdict=font)
plt.xticks(fontsize=font["size"], fontweight=font["weight"])
plt.yticks(fontsize=font["size"], fontweight=font["weight"])
plt.title('Performance across races')
ax.text(x_label_coord, y_label_coord, 'B', transform=ax.transAxes, size=24, weight='bold')
matplotlib.rc('font', **font)
plt.ylim([0.70, 0.85])
plt.legend(loc=2)

plt.subplot(1,3,3)
matplotlib.rc('font', **font)
ax = axes[2]
plt.plot(list_times_since_start, [df.loc[metric, "mean"] for df in list_eval_start_women], c = 'b', label = 'Women')
plt.fill_between(list_times_since_start, [df.loc[metric, "CI_low"] for df in list_eval_start_women], [df.loc[metric, "CI_high"] for df in list_eval_start_women], color = 'b', alpha=0.3)

plt.plot(list_times_since_start, [df.loc[metric, "mean"] for df in list_eval_start_men], c = 'r', label = 'Men')
plt.fill_between(list_times_since_start, [df.loc[metric, "CI_low"] for df in list_eval_start_men], [df.loc[metric, "CI_high"] for df in list_eval_start_men], color = 'r', alpha=0.3)

plt.xlabel('Years since first visit', fontdict=font)
plt.ylabel('AUROC', fontdict=font)
plt.xticks(fontsize=font["size"], fontweight=font["weight"])
plt.yticks(fontsize=font["size"], fontweight=font["weight"])
plt.title('Performance across genders')
ax.text(x_label_coord, y_label_coord, 'C', transform=ax.transAxes, size=24, weight='bold')
matplotlib.rc('font', **font)
plt.ylim([0.70, 0.85])
plt.legend(loc = 2)

plt.tight_layout()
plt.savefig(f'{output_folder}/auroc_over_time_since_start_withdemo.pdf', dpi = 300)
