'''
Creation:
    Author: Martin Grunnill
    Date: 2022-11-14
Description: Performing PRCC and producing figures.
    
'''
#%%
import pandas as pd
import copy
import os
from LH_sampling.gen_LHS_and_simulate_serrialy import calucate_PCC
from LH_sampling.load_variables_and_parameters import load_parameters, load_repeated_sample
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

sample_size = 12375
data_dir =  'C:/Users/mdgru/OneDrive - York University/Data/World_cup_modelling'
data_dir = data_dir + '/Assesing testing regimes with LH sample Size ' + str(sample_size)+'/'
fig_dir = data_dir +'/Figures'
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)
fig_dir = fig_dir +'/'



#%%
# loading data
# LH sample
LH_sample = pd.read_csv(data_dir+'LH sample.csv')
testing_regimes = ['No Testing',
                   'Pre-travel RTPCR',
                   'Pre-travel RA',
                   'Pre-match RTPCR',
                   'Pre-match RA',
                   'Post-match RTPCR',
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
# Calculating PCCs
parameters_sampled = LH_sample.columns.to_list()
parameters_sampled.remove('Sample Number')
outputs = ['peak infected',
           'total infections',
           'peak hospitalised',
           'total hospitalisations',
           'total positive tests']
pcc_args = []
for parameter in parameters_sampled:
    covariables = [item
                   for item in parameters_sampled
                   if item != parameter]
    for output in outputs:
        pcc_args.append((parameter, output, covariables))

PCCs = []

for testing_regime, results_df in tqdm(results_dfs.items(), desc='PCCs for testing regime'):
    pccs_in_regime = []
    for parameter, output, covariables in tqdm(pcc_args, desc='PCC for parameter'):
        PCC = calucate_PCC(results_df, parameter, output, covariables, method='spearman')
        pccs_in_regime.append(PCC)


    pccs_in_regime = pd.concat(pccs_in_regime)
    pccs_in_regime.sort_index(inplace=True)
    pccs_in_regime['Test Regime'] = testing_regime
    if testing_regime == 'No Testing':
        pccs_in_regime['Test Time'] = 'None'
        pccs_in_regime['Test Type'] = 'None'
    else:
        testing_regime_split = testing_regime.split(' ')
        pccs_in_regime['Test Time'] = testing_regime_split[0]
        pccs_in_regime['Test Type'] = testing_regime_split[1]
    PCCs.append(pccs_in_regime)

PCCs_df = pd.concat(PCCs)
PCCs_df.reset_index(inplace=True)
PCC_measure = PCCs_df['index'].str.split(' on ', n = 1, expand = True)
PCCs_df['Parameter'] = PCC_measure[0]
PCCs_df['Output'] = PCC_measure[1]
PCCs_df.drop(columns=['index'],inplace=True)
PCCs_df[['lower_CI_0.95','upper_CI_0.95']] = pd.DataFrame(PCCs_df['CI95%'].tolist())
PCCs_df.drop(columns=['CI95%'], inplace=True)

#%%
# Figures
order = ['Capacity', 'eta_{spectators}', 'N_{staff}',
         'v_A', 'v_B',
         'Team A prevalence', 'Team B prevalence',
         'sigma_A', 'sigma_B', 'sigma_{host}',
         'R_0', 'b',
         'l_effective', 'VE_{hos}',
         'epsilon_H', 'gamma_H',
         'p_s',  'p_h',
         'kappa', 'theta']
paramater_legend = {'$N_C$': 'Arena Capacity', '$N^*_{Q}$': 'Proportion of host tickets',
                    '$N_{S}$': 'Staff Population',
                    '$v_A$':'Proportion Recently Vaccinated  Team A', '$v_B$':'Proportion Recently Vaccinated  Team B',
                   '$\\rho_A$':'Reported Prevalence Supporters A', '$\\rho_B$':'Reported Prevalence Supporters B',
                   '$\sigma_A$': 'Actual/Reported Prevalence Supporters A',
                    '$\sigma_B$': 'Actual/Reported Prevalence Supporters B',
                    '$\sigma_H$': 'Actual/Reported Prevalence Hosts',
                   '$R_0$': 'Basic Reproduction Number', '$b$': 'Increase in Transmission Match Day',
                   '$l_E$':'Recent Vaccine Efficacy against infection', '$VE_{h}$':'Recent Vaccine Efficacy against hospitalising infection',
                   '$\epsilon_h$':'Rate of Hospitalisation', '$\gamma_h$':'Rate of Hospital Recovery',
                   '$p_s$':'Proportion Symptomatic', '$p_{h|s}$':'Proportion Hospitalised given Symptomatic',
                   '$\kappa$':'Asymptomatic Transmission Modifier', '$\\theta$': 'Isolated Transmission Modifier'}

hue_order=testing_regimes
for output in outputs:
    plt.figure() # clear any prevously plotted figures
    selected_output_df = PCCs_df[PCCs_df.Output == output]
    fig = sns.stripplot(selected_output_df, x="r", y="Parameter", order=order, hue='Test Regime', hue_order=hue_order)
    fig.set_yticklabels(paramater_legend.keys())
    fig.set_xlim(-1,1)
    sns.move_legend(fig, 'upper left', bbox_to_anchor=(1, 0.75), ncol=1)

    plt.tight_layout()
    plt.grid(visible='both')
    plt.savefig(fig_dir+'PRCCs '+ output+ '.tiff')

#%%
# Test regime as parameter
actual_testing_regimes = copy.deepcopy(testing_regimes)
actual_testing_regimes.remove('No Testing')
test_PCC = []
for testing_regime in actual_testing_regimes:
    no_testing_results_df = copy.deepcopy(results_dfs['No Testing'])
    no_testing_results_df[testing_regime] = 0
    testing_regime_results = copy.deepcopy(results_dfs[testing_regime])
    testing_regime_results[testing_regime] = 1
    comparison_df = pd.concat([no_testing_results_df,testing_regime_results])
    for output in outputs:
        test_PCC.append(calucate_PCC(comparison_df,
                                     parameter=testing_regime,
                                     output=output,
                                     covariables=parameters_sampled,
                                     method='spearman')
                        )

test_PCC_df = pd.concat(test_PCC)
test_PCC_df.reset_index(inplace=True)
PCC_measure = test_PCC_df['index'].str.split(' on ', n = 1, expand = True)
test_PCC_df['Test Regime'] = PCC_measure[0]
test_PCC_df['Output'] = PCC_measure[1]
test_PCC_df.drop(columns=['index'],inplace=True)
test_PCC_df[['lower_CI_0.95','upper_CI_0.95']] = pd.DataFrame(test_PCC_df['CI95%'].tolist())
test_PCC_df.drop(columns=['CI95%'], inplace=True)

plt.figure()  # clear any prevously plotted figures
fig = sns.stripplot(test_PCC_df, x="r", y="Output", hue='Test Regime', hue_order=actual_testing_regimes)
fig.set_xlim(-1, 1)
fig.set_yticklabels([entry.replace(' ','\n') for entry in outputs])
sns.move_legend(fig, 'upper left', bbox_to_anchor=(1, 0.75), ncol=1)
plt.tight_layout()
plt.grid(visible='both')
plt.savefig(fig_dir + 'Test Regime PRCCs against no testing.tiff')

#%%
