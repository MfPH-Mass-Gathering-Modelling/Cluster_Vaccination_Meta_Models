"""
Creation:
    Author: Martin Grunnill
    Date: 2022-11-14
Description: Performing PRCC and producing figures.
    
"""
#%%
import pandas as pd
import os
from LH_sampling.gen_LHS_and_simulate_serrialy import calucate_PRCC
from LH_sampling.load_variables_and_parameters import load_parameters, load_repeated_sample
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

sample_size = 12375
data_dir =  'C:/Users/mdgru/OneDrive - York University/Data/World_cup_modelling'
data_dir = data_dir + '/Assesing testing regimes with LH sample Size'+str(sample_size)+'/'



#%%
# loading data
# LH sample
LH_sample = pd.read_csv(data_dir+'LH sample.csv')
testing_regimes = ['No Testing',
                   'Pre-travel RTPCR',
                   'Pre-match RTPCR',
                   'Post-match RTPCR',
                   'Pre-travel RA',
                   'Pre-match RA',
                   'Post-match RA'
                   ]

results_dfs = {}
for testing_regime in tqdm(testing_regimes, desc='loading data from testing regime'):
    regime_data_dir = data_dir + testing_regime + '/'
    all_data_csv = regime_data_dir + 'All Focused Outputs and Samples.csv'
    if os.path.isfile(all_data_csv):
        results_dfs[testing_regime] = pd.read_csv(all_data_csv)
    else:
        csvs = [regime_data_dir+'Focused Outputs and Sample ' + str(index)+ '.csv' for index in range(sample_size)]
        df = pd.concat(map(pd.read_csv, csvs), ignore_index=True)
        df.to_csv(all_data_csv, index=False)
        results_dfs[testing_regime] = df

#%%
# Calculating PRCCs
parameters_sampled = LH_sample.columns.to_list()
parameters_sampled.remove('Sample Number')
outputs = ['peak infected',
           'total infections',
           'peak hospitalised',
           'total hospitalisations']#, 'total positive tests']
prcc_args = []
for parameter in parameters_sampled:
    covariables = [item
                   for item in parameters_sampled
                   if item != parameter]
    for output in outputs:
        prcc_args.append((parameter, output, covariables))

PRCCs = []
for testing_regime, results_df in tqdm(results_dfs.items(), desc='PRCCs for testing regime'):
    prccs_in_regime = []
    for parameter, output, covariables in tqdm(prcc_args, desc='PRCC for parameter'):
        prccs_in_regime.append(calucate_PRCC(results_df, parameter, output, covariables))
    prccs_in_regime = pd.concat(prccs_in_regime)
    prccs_in_regime.sort_index(inplace=True)
    prccs_in_regime['Test Regime'] = testing_regime
    if testing_regime == 'No Testing':
        prccs_in_regime['Test Time'] = 'None'
        prccs_in_regime['Test Type'] = 'None'
    else:
        testing_regime_split = testing_regime.split(' ')
        prccs_in_regime['Test Time'] = testing_regime_split[0]
        prccs_in_regime['Test Type'] = testing_regime_split[1]
    PRCCs.append(prccs_in_regime)

PRCCs_df = pd.concat(PRCCs)
PRCCs_df.reset_index(inplace=True)
PRCC_measure = PRCCs_df['index'].str.split(" on ", n = 1, expand = True)
PRCCs_df['Parameter'] = PRCC_measure[0]
PRCCs_df['Output'] = PRCC_measure[1]
PRCCs_df.drop(columns=['index'],inplace=True)
PRCCs_df[['lower_CI_0.95','upper_CI_0.95']] = pd.DataFrame(PRCCs_df['CI95%'].tolist())
PRCCs_df.drop(columns=['CI95%'], inplace=True)

#%%
# Figures
total_inf_prccs = PRCCs_df[PRCCs_df.Output=='total infections']
ax = sns.stripplot(data=total_inf_prccs, x='Parameter', y="r", hue='Test Regime')
sns.move_legend(
    ax, "upper left", bbox_to_anchor=(1, 1),
    title=None, frameon=False)
plt.xticks(rotation=75)
#ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()