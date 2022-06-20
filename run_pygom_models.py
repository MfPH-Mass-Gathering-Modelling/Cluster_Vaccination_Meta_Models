"""
Creation:
    Author: Martin Grunnill
    Date: 11/04/2022
Description: Run pygom models with single clusters and Ontarian vaccination data.
    
"""

#%%
# import packages
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy

#%%
# get model meta population structure
dir_name = ('C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/'+
            'Mass Gathering work/Compartment based models/Cluster_Vaccination_Meta_Models/'+
            'CVM_models/Model meta population structures/')
with open(dir_name + "single cluster 3 dose model.json", "r") as json_file:
    group_info=json_file.read()

group_info = json.loads(group_info)

#%%

# Import  model constructor
from CVM_models.pygom_models.mass_gathering_vaccination import MGModelConstructor

# Update model constructor
mg_model_constructor = MGModelConstructor(group_info)
#%%
# and generate model
mg_model = mg_model_constructor.generate_model(variety='deterministic')

#%%
# Check ODE model
mg_ode = mg_model.get_ode_eqn()
#%%
mg_ode[1]

#%%
# updating fixed_parameters
mg_model_constructor.all_parameters

ve_dict = {} # V.E. = vaccine efficacy. This has been cut and pasted from run_Mass_Gathring_pure_python_model.py
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
# Putting VE dict in fixed_params
for param, value in ve_dict['ve_infection'].items():
    fixed_params['l_'+ param ] = value

for param, value in ve_dict['ve_symptoms'].items():
    fixed_params['r_'+ param ] = value

for param, value in ve_dict['ve_hospitalisation'].items():
    fixed_params['h_'+ param ] = value

for param, value in ve_dict['ve_mortality'].items():
    fixed_params['m_'+ param ] = value

#%%
# Updating rest of fixed_params.
fixed_params.update({
    'theta': 0.342,
    'epsilon_1': 0.5988023952,
    'epsilon_2': 0.4291845494,
    'epsilon_3': 1,
    'gamma_I_1': 0.1627*2,
    'gamma_I_2': 0.1627*2,
    'gamma_A_1': 0.139*2,
    'gamma_A_2': 0.139*2,
    'psi': 0.0826446281,
    'rho': 0.724,
    'beta__': 0.5, # may change conform to Ontario's 1st wave see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8454019/pdf/CCDR-47-329.pdf
    'eta_': 0.038,
    'mu_': 0.0829,
    'alpha': 1/(6*28), # assumin 6 months natural immunity
})

#%%
# Setting fixed and piecewise params
mg_model.fixed_parameters = fixed_params
mg_model.piecwise_params = piecewise_params

#%%
# Create Test population lets assume no infections prior to vaccination program and 10 infected arrive.
number_exposed = 1
y0 = np.zeros(mg_model.num_state)
index_S__unvaccinated = mg_model.get_state_index('S__unvaccinated')
y0[index_S__unvaccinated] = starting_population-number_exposed
index_E__unvaccinated = mg_model.get_state_index('E__unvaccinated')
y0[index_E__unvaccinated] = number_exposed
mg_model.initial_values = (y0, t[0])

#%%
# Running model
sol = mg_model.integrate(t[1::])

#%%
# Checking model is working as it should.
multi_columns = [(key_l1, key_l2)
                 for key_l1, sub_dict in mg_model_constructor.cluste_vaccine_group_state_index[""].items()
                 for key_l2 in sub_dict.keys()]

sol_df = pd.DataFrame(sol, index=vac_data['report_date'][t].tolist())
sol_df.columns = pd.MultiIndex.from_tuples(multi_columns)

#%% Conversion of dataframe for use in seaborn plotting package
sol_melted = pd.melt(sol_df, ignore_index=False)
sol_line_list = sol_melted.reset_index()
sol_line_list.columns = ['date','vaccine_group','state','population']
sol_line_list.replace({'H_total':'H','D_total':'D','observed_states':'accumelated_totals'},inplace=True)


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
sol_state_totals.drop(columns=['H_total','D_total'], inplace=True)
plot, axes = plt.subplots(12, 1, sharex=True)
plt.xticks(rotation=45)
axes_index = 0
for state in mg_model.states:
    axes[axes_index].plot(sol_state_totals.index, sol_state_totals[state],color='black')
    axes_index += 1
plt.show()
