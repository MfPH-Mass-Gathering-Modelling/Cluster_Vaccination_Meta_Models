"""
Creation:
    Author: Martin Grunnill
    Date: 01/04/2022
Description: Example code for running piecewise vaccination mass gathering model.
    
"""

#%%
# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


# %%
# import model
from CVM_models.pure_python_models.mass_gathering_piecewise_vaccination import MassGatheringModel

#%%
# get model meta population structure
structures_dir = ('C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/'+
                  'Mass Gathering work/Compartment based models/Cluster_Vaccination_Meta_Models/'+
                  'CVM_models/Model meta population structures/')
with open(structures_dir + "general population with testing 3 dose model.json", "r") as json_file:
    group_info=json_file.read()

group_info = json.loads(group_info)


#%%
# Initialise model
# Ontario’s population reached 14,755,211 on January 1, 2021
# https://www.ontario.ca/page/ontario-demographic-quarterly-highlights-fourth-quarter-2020#:~:text=Ontario’s%20population%20reached%2014%2C755%2C211%20on,quarter%20of%20the%20previous%20year..
starting_population = 10000



#%%
mg_model = MassGatheringModel(starting_population, group_info)

#%%
# If available attach Jacobian. Note this has to avaluated for every different meta-population structure.
# See CVM_models/pure_python_models/deriving_MG_model_jacobian.py
# cvm_dir = ('C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/Mass Gathering work/'+
#            'Compartment based models/Cluster_Vaccination_Meta_Models/CVM_models/pure_python_models/')
#
# json_file = cvm_dir+"MG_model_jacobian.json"
#
# mg_model.load_dok_jacobian(json_file)

# %%
# Test parameter values
beta = 1.759782,  # may change conform to Ontario's 1st wave see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8454019/pdf/CCDR-47-329.pdf
kappa = 0.25
beta_for_test_positive = beta*kappa
test_params = {
    # Test parameter values - Vaccination parameters
    # These are available at https://docs.google.com/spreadsheets/d/1_XQn8bfPXKA8r1D_rz6Y1LeMTR_Y0RK5Rh4vnxDcQSo/edit?usp=sharing
    # These are placed in a list corresponding to the order of vaccination groups (enter MassGatheringModel.vaccination_groups)
    # For now we will assume that the 3rd dose is effective as the 2nd dose before wanning immunity.
    'l_unvaccinated':0,
    'l_first_dose_delay':0,
    'l_first_dose_effective':0.634,
    'l_second_dose_delay':0.634,
    'l_second_dose_effective':0.7443333333,
    'l_waned':0.2446666667,
    'l_third_dose_delay':0.2446666667,
    'l_third_dose_effective':0.7443333333,

    's_unvaccinated':0,
    's_first_dose_delay':0,
    's_first_dose_effective':0.479,
    's_second_dose_delay':0.479,
    's_second_dose_effective':0.7486666667,
    's_waned':0.274,
    's_third_dose_delay':0.274,
    's_third_dose_effective':0.7486666667,

    'h_unvaccinated':0,
    'h_first_dose_delay':0,
    'h_first_dose_effective':0.661,
    'h_second_dose_delay':0.661,
    'h_second_dose_effective':0.957,
    'h_waned':0.815,
    'h_third_dose_delay':0.815,
    'h_third_dose_effective':0.957,

    'beta_general_population': beta,
    'beta_PCR_positive_waiting': beta,
    'beta_PCR_positive': beta_for_test_positive,
    'beta_LFD_true_positive': beta_for_test_positive,
    'theta':0.342,
    'nu_d':1/14,
    'nu_w': 1/84,
    'epsilon_1':0.5988023952,
    'epsilon_2':0.4291845494,
    'epsilon_3':1,
    'p_s':0.724,
    'gamma_A_1':0.139*2,
    'gamma_A_2':0.139*2,
    'gamma_I_1':0.1627*2,
    'gamma_I_2':0.1627*2,
    'p_d':0.8,
    'p_h':0.05,
    'gamma_H':0.0826446281,
    'alpha':1/(6*28), # assumin 6 months natural immunity
    'kappa_D':kappa,
    'omega':1
}


#%%
# Get Ontario's vaccination data
from data_extraction.vaccination_data import ontario
vac_data = ontario()
first_vac_dose = vac_data.previous_day_at_least_one
second_vac_dose = vac_data.previous_day_fully_vaccinated
third_vac_dose = vac_data.previous_day_3doses



#%%
# Create Test population lets assume no infections prior to vaccination program and 10 infected arrive.
y0 = np.zeros(mg_model.num_state)
infection_seed = 1
y0[0] = starting_population-infection_seed
y0[1] = infection_seed
t = range(len(third_vac_dose)-5)

#%%
# Runninng mg_model
sol = mg_model.integrate(y0, t, **test_params)

#%%
# Checking mg_model is working as it should.

multi_columns = []
for cluster, sub_dict in mg_model.state_index.items():
    if cluster != 'observed_states':
        for vaccine_group, state_dict in sub_dict.items():
            for state in state_dict.keys():
                multi_columns.append((cluster, vaccine_group, state))
    else:
        for state in sub_dict.keys():
            multi_columns.append((cluster, None, state))

sol_df = pd.DataFrame(sol, index=vac_data['report_date'][t].tolist())
sol_df.columns = pd.MultiIndex.from_tuples(multi_columns)

#%% Conversion of dataframe for use in seaborn plotting package
sol_melted = pd.melt(sol_df, ignore_index=False)
sol_line_list = sol_melted.reset_index()
sol_line_list.columns = ['date','cluster', 'vaccine_group','state','population']
sol_line_list.replace({'observed_states':'acumelated_totals'},inplace=True)


#%%
#Plotting graph of states accross vaccine groups
graph_states_accross_groups = sns.FacetGrid(sol_line_list, col='vaccine_group',  hue='state', col_wrap=3)
graph_states_accross_groups.map(sns.lineplot, 'date', 'population')
graph_states_accross_groups.add_legend()
plt.show()

#%%
#Plotting graph of vaccine groups accross states
graph_groups_accross_states = sns.FacetGrid(sol_line_list, hue='vaccine_group',  col='state', col_wrap=3)
graph_groups_accross_states.map(sns.lineplot, 'date', 'population')
graph_groups_accross_states.add_legend()
plt.show()



#%%
# Plotting graph of vaccination totals
sol_vaccination_totals = sol_df.groupby(level=1,axis=1).sum()
data_df = vac_data.iloc[t]
plot, axes = plt.subplots(4, 1, sharex=True)
plt.xticks(rotation=45)
axes[0].plot(sol_vaccination_totals.index, sol_vaccination_totals.unvaccinated,color='black')
axes[0].scatter(data_df.report_date,data_df.unvaccinated,color='red')
axes[1].plot(sol_vaccination_totals.index, sol_vaccination_totals[['first_dose_delay',
                                                                   'first_dose_effective']].sum(axis=1),color='black')
axes[1].scatter(data_df.report_date,data_df.total_individuals_partially_vaccinated,color='yellow')
axes[2].plot(sol_vaccination_totals.index, sol_vaccination_totals[['second_dose_delay',
                                                                   'second_dose_effective',
                                                                   'waned']].sum(axis=1),color='black')
axes[2].scatter(data_df.report_date,data_df.total_individuals_fully_vaccinated,color='blue')
axes[3].plot(sol_vaccination_totals.index, sol_vaccination_totals[['third_dose_delay',
                                                                   'third_dose_effective']].sum(axis=1),color='black')
axes[3].scatter(data_df.report_date,data_df.total_individuals_3doses,color='green')
plt.show()
# In terms of progression through vaccination groups everything seems to work.

#%%
#Plotting graph of state totals
sol_state_totals = sol_df.groupby(level=2,axis=1).sum()
plot, axes = plt.subplots(len(mg_model.states), 1, sharex=True)
plt.xticks(rotation=45)
axes_index = 0
for state in mg_model.states:
    axes[axes_index].plot(sol_state_totals.index, sol_state_totals[state],color='black')
    axes_index += 1
plt.show()
