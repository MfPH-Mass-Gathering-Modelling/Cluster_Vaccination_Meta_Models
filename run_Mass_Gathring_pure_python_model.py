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
import copy


#%%
# Get Ontario's vaccination data
from data_extraction.vaccination_data import ontario
vac_data = ontario()
first_vac_dose = vac_data.previous_day_at_least_one
second_vac_dose = vac_data.previous_day_fully_vaccinated
third_vac_dose = vac_data.previous_day_3doses


# %%
# import model
from CVM_models.pure_python_models.mass_gathering_piecewise_vaccination import MassGatheringModel

#%% 
# Test parameter values - Vaccination parameters
# These are available at https://docs.google.com/spreadsheets/d/1_XQn8bfPXKA8r1D_rz6Y1LeMTR_Y0RK5Rh4vnxDcQSo/edit?usp=sharing
# These are placed in a list corresponding to the order of vaccination groups (enter MassGatheringModel.vaccination_groups)
# For now we will assume that the 3rd dose is effective as the 2nd dose before wanning immunity.
ve_dict = {}
ve_dict['ve_infection'] = {
    'unvaccinated': 0, 
    'first_dose_delay': 0,
    'first_dose': 0.634,
    'second_dose_delay': 0.634,
    'second_dose': 0.7443333333,
    'second_dose_waned': 0.2446666667,
    'third_dose_delay': 0.2446666667,
    'third_dose': 0.7443333333}
ve_dict['ve_symptoms'] = {
    'unvaccinated': 0,
    'first_dose_delay': 0,
    'first_dose': 0.479,
    'second_dose_delay': 0.479,
    'second_dose': 0.7486666667,
    'second_dose_waned': 0.274,
    'third_dose_delay': 0.274,
    'third_dose': 0.7486666667}
ve_dict['ve_hospitalisation'] = {
    'unvaccinated': 0,
    'first_dose_delay': 0,
    'first_dose': 0.661,
    'second_dose_delay': 0.661,
    'second_dose': 0.957,
    'second_dose_waned': 0.815,
    'third_dose_delay': 0.815,
    'third_dose': 0.957}
# assume that VE against hospitalisation and death are the same.
ve_dict['ve_mortality'] = copy.deepcopy(ve_dict['ve_hospitalisation'])

#%%
# Initialise model
# Ontario’s population reached 14,755,211 on January 1, 2021
# https://www.ontario.ca/page/ontario-demographic-quarterly-highlights-fourth-quarter-2020#:~:text=Ontario’s%20population%20reached%2014%2C755%2C211%20on,quarter%20of%20the%20previous%20year..
starting_population = 14755211
groups_loss_via_vaccination = {'unvaccinated': first_vac_dose,
                               'first_dose': second_vac_dose,
                               'second_dose': third_vac_dose}

#%%
mg_model = MassGatheringModel(starting_population,groups_loss_via_vaccination, ve_dict)



# %%
# Test parameter values
beta = 0.5  # may change conform to Ontario's 1st wave see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8454019/pdf/CCDR-47-329.pdf
asymptomatic_tran_mod = 0.342
inverse_effective_delay = 1/14
inverse_waning_immunity =1/84
epsilon_1 = 0.5988023952
epsilon_2 = 0.4291845494
epsilon_3 = 1
rho = 0.724
gamma_a_1 = 0.139*2
gamma_a_2 = 0.139*2
gamma_i_1 = 0.1627*2
gamma_i_2 = 0.1627*2
eta = 0.038
mu = 0.0829
psi = 0.0826446281
alpha = 1/(6*28) # assumin 6 months natural immunity
parameters = (
    beta,
    asymptomatic_tran_mod,
    inverse_effective_delay,
    inverse_waning_immunity,
    epsilon_1,
    epsilon_2,
    epsilon_3,
    rho,
    gamma_a_1,
    gamma_a_2,
    gamma_i_1,
    gamma_i_2,
    eta,
    mu,
    psi,
    alpha
)


#%%
# Create Test population lets assume no infections prior to vaccination program and 10 infected arrive.
y0 = np.zeros(mg_model.num_state)
infection_seed = 1
y0[0] = starting_population-infection_seed
y0[1] = infection_seed
t = range(len(vac_data.previous_day_at_least_one)-5)

#%%
# Runninng mg_model
sol = mg_model.integrate(y0, t, parameters)

#%%
# Checking mg_model is working as it should.
multi_columns = [(key_l1, key_l2)
                 for key_l1, sub_dict in mg_model.state_index.items()
                 for key_l2 in sub_dict.keys()]

sol_df = pd.DataFrame(sol, index=vac_data['report_date'][t].tolist())
sol_df.columns = pd.MultiIndex.from_tuples(multi_columns)

#%% Conversion of dataframe for use in seaborn plotting package
sol_melted = pd.melt(sol_df, ignore_index=False)
sol_line_list = sol_melted.reset_index()
sol_line_list.columns = ['date','vaccine_group','state','population']
sol_line_list.replace({'H_T':'H','D_T':'D','observed_states':'accumelated_totals'},inplace=True)


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
sol_vaccination_totals = sol_df.groupby(level=0,axis=1).sum()
data_df = vac_data.iloc[t]
plot, axes = plt.subplots(4, 1, sharex=True)
plt.xticks(rotation=45)
axes[0].plot(sol_vaccination_totals.index, sol_vaccination_totals.unvaccinated,color='black')
axes[0].scatter(data_df.report_date,data_df.unvaccinated,color='red')
axes[1].plot(sol_vaccination_totals.index, sol_vaccination_totals[['first_dose_delay',
                                                       'first_dose']].sum(axis=1),color='black')
axes[1].scatter(data_df.report_date,data_df.total_individuals_partially_vaccinated,color='yellow')
axes[2].plot(sol_vaccination_totals.index, sol_vaccination_totals[['second_dose_delay',
                                                       'second_dose',
                                                       'second_dose_waned']].sum(axis=1),color='black')
axes[2].scatter(data_df.report_date,data_df.total_individuals_fully_vaccinated,color='blue')
axes[3].plot(sol_vaccination_totals.index, sol_vaccination_totals[['third_dose_delay',
                                                       'third_dose']].sum(axis=1),color='black')
axes[3].scatter(data_df.report_date,data_df.total_individuals_3doses,color='green')
plt.show()
# In terms of progression through vaccination groups everything seems to work.

#%%
#Plotting graph of state totals
sol_state_totals = sol_df.groupby(level=1,axis=1).sum()
sol_state_totals.drop(columns=['H_T','D_T'], inplace=True)
plot, axes = plt.subplots(12, 1, sharex=True)
plt.xticks(rotation=45)
axes_index = 0
for state in mg_model.states:
    axes[axes_index].plot(sol_state_totals.index, sol_state_totals[state],color='black')
    axes_index += 1
plt.show()
