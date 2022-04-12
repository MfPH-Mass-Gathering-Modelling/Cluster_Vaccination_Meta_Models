"""
Creation:
    Author: Martin Grunnill
    Date: 11/04/2022
Description: Run pygom models with single clusters and Ontarian vaccination data.
    
"""

#%%
# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy

#%% Setup vaccination groups and clusters
clusters = ['']
vaccine_groups = [
    'unvaccinated',
    'first_dose',
    'second_dose',
    'third_dose'
]

#%%
# Start with base model, i.e. no infections just susceptibles running through vaccination groups
# Import base model constructor
from CVM_models.pygom_models.base_vaccination import BaseMultiClusterVacConstructor

# Initalise constructor.
base_model_constructor = BaseMultiClusterVacConstructor(clusters, vaccine_groups)

#%%
# Get Ontario's vaccination data.
from data_extraction.vaccination_data import ontario
vac_data = ontario()
piecewise_params = {'nu__unvaccinated': {'targets': vac_data.previous_day_at_least_one,  # nu_1 is the rate of vaccination rate for unvaccinated group.
                                         'source states': [state + '__' + 'unvaccinated'
                                                           for state in base_model_constructor.vaccinable_states]},
                    'nu__first_dose' : {'targets': vac_data.previous_day_fully_vaccinated, # nu_3 is the rate of vaccination rate for first_dose group.
                                        'source states': [state + '__' + 'first_dose'
                                                          for state in base_model_constructor.vaccinable_states]},
                    'nu__second_dose' : {'targets': vac_data.previous_day_3doses, # nu_1 is the rate of vaccination rate for second_dose_waned group.
                                         'source states': [state + '__' + 'second_dose'
                                                           for state in base_model_constructor.vaccinable_states]}}

#%%
# Ontario’s population reached 14,755,211 on January 1, 2021
# https://www.ontario.ca/page/ontario-demographic-quarterly-highlights-fourth-quarter-2020#:~:text=Ontario’s%20population%20reached%2014%2C755%2C211%20on,quarter%20of%20the%20previous%20year..
starting_population = 14755211
fixed_params = {'N_':starting_population}




#%%
# generate pygom model
base_model = base_model_constructor.generate_model(variety='piecewise parameter estimation')

#%%
# Check ODE model
base_model.get_ode_eqn()


#%%
# update piecewise param estimation dictionary with source states
for vaccine_group in ['unvaccinated','first_dose','second_dose']:
    piecewise_params['nu__'+vaccine_group]['source states'] = [state + '__' +vaccine_group
                                                               for state in base_model_constructor.vaccinable_states]

#%%
# set fixed and piecewise params
base_model.fixed_parameters = fixed_params
base_model.piecwise_params = piecewise_params

# Create Test population lets assume no infections prior to vaccination program and 10 infected arrive.
y0 = np.zeros(base_model.num_state)
index_S__unvaccinated = base_model.get_state_index('S__unvaccinated')
y0[index_S__unvaccinated ] = starting_population
t = np.arange(len(vac_data.previous_day_at_least_one)-5)
base_model.initial_values = (y0, t[0])

#%%
# Run Model
sol = base_model.integrate(t[1::])

#%%
# Checking results.

sol_df = pd.DataFrame(sol, index=vac_data['report_date'][t].tolist(), columns=base_model._stateDict.keys())


#%%
# Plotting graph of vaccination totals
sol_df  = sol_df.groupby(level=0,axis=1).sum()
data_df = vac_data.iloc[t]
plot, axes = plt.subplots(4, 1, sharex=True)
plt.xticks(rotation=45)
axes[0].plot(sol_df .index, sol_df.S__unvaccinated,color='black')
axes[0].scatter(data_df.report_date,data_df.unvaccinated,color='red')
axes[1].plot(sol_df .index, sol_df.S__first_dose,color='black')
axes[1].scatter(data_df.report_date,data_df.total_individuals_partially_vaccinated,color='yellow')
axes[2].plot(sol_df .index, sol_df.S__second_dose,color='black')
axes[2].scatter(data_df.report_date,data_df.total_individuals_fully_vaccinated,color='blue')
axes[3].plot(sol_df .index, sol_df.S__third_dose,color='black')
axes[3].scatter(data_df.report_date,data_df.total_individuals_3doses,color='green')
plt.show()
# In terms of progression through vaccination groups everything seems to work.

#%%
# Basic model with delay in vaccine eficacy and wanning immunity

# Need to update vaccine groups,
vaccine_groups = [
    'unvaccinated',
    'first_dose_delay',
    'first_dose',
    'second_dose_delay',
    'second_dose',
    'second_dose_waned',
    'third_dose_delay',
    'third_dose'
]
#%%
# Update model constructor
vac_delay_and_wanning_model_constructor = BaseMultiClusterVacConstructor(clusters, vaccine_groups)

#%%
# and generate model
vac_delay_and_wanning_model = vac_delay_and_wanning_model_constructor.generate_model(variety='piecewise parameter estimation')

#%%
# Check ODE model
vac_delay_and_wanning_model.get_ode_eqn()

#%%
# Update parameters
# piecewise_params
piecewise_params = {'nu__unvaccinated': {'targets': vac_data.previous_day_at_least_one,  # nu_1 is the rate of vaccination rate for unvaccinated group.
                                         'source states': [state + '__' + 'unvaccinated'
                                                           for state in base_model_constructor.vaccinable_states]},
                    'nu__first_dose' : {'targets': vac_data.previous_day_fully_vaccinated, # nu_3 is the rate of vaccination rate for first_dose group.
                                        'source states': [state + '__' + 'first_dose'
                                                          for state in base_model_constructor.vaccinable_states]},
                    'nu__second_dose_waned' : {'targets': vac_data.previous_day_3doses, # nu_1 is the rate of vaccination rate for second_dose_waned group.
                                               'source states': [state + '__' + 'second_dose_waned'
                                                                 for state in base_model_constructor.vaccinable_states]}}


# and fixed_params
inverse_delay_ve = 1/14
inverse_wanning_ve = 1/(3*30)
fixed_params.update({'nu__first_dose_delay':inverse_delay_ve,
                     'nu__second_dose_delay':inverse_delay_ve,
                     'nu__second_dose': inverse_wanning_ve,
                     'nu__third_dose_delay': inverse_delay_ve})
vac_delay_and_wanning_model.fixed_parameters = fixed_params
vac_delay_and_wanning_model.piecwise_params = piecewise_params


# Create Test population lets assume no infections prior to vaccination program and 10 infected arrive.
y0 = np.zeros(vac_delay_and_wanning_model.num_state)
index_S__unvaccinated = vac_delay_and_wanning_model.get_state_index('S__unvaccinated')
y0[index_S__unvaccinated] = starting_population
t = np.arange(len(vac_data.previous_day_at_least_one)-5)
vac_delay_and_wanning_model.initial_values = (y0, t[0])

#%%
# Run Model
sol = vac_delay_and_wanning_model.integrate(t[1::])

#%%
# Checking results

sol_df = pd.DataFrame(sol, index=vac_data['report_date'][t].tolist(), columns=vac_delay_and_wanning_model._stateDict.keys())

# Plotting graph of vaccination totals
sol_df  = sol_df.groupby(level=0,axis=1).sum()
data_df = vac_data.iloc[t]
plot, axes = plt.subplots(4, 1, sharex=True)
plt.xticks(rotation=45)
axes[0].plot(sol_df .index, sol_df.S__unvaccinated,color='black')
axes[0].scatter(data_df.report_date,data_df.unvaccinated,color='red')
axes[1].plot(sol_df.index, sol_df[['S__first_dose_delay',
                                   'S__first_dose']].sum(axis=1),color='black')
axes[1].scatter(data_df.report_date,data_df.total_individuals_partially_vaccinated,color='yellow')
axes[2].plot(sol_df.index, sol_df[['S__second_dose_delay',
                                   'S__second_dose',
                                   'S__second_dose_waned']].sum(axis=1),color='black')
axes[2].scatter(data_df.report_date,data_df.total_individuals_fully_vaccinated,color='blue')
axes[3].plot(sol_df.index, sol_df[['S__third_dose_delay',
                                   'S__third_dose']].sum(axis=1),color='black')
axes[3].scatter(data_df.report_date,data_df.total_individuals_3doses,color='green')
plt.show()
# In terms of progression through vaccination groups everything seems to work.

#%%
# Now for an epidimiological model. For now lets go with the mass gathering model.
from CVM_models.pygom_models.mass_gathering_vaccination import MGModelConstructor

# Update model constructor
mg_model_constructor = MGModelConstructor(clusters, vaccine_groups)

#%%
# and generate model
mg_model = mg_model_constructor.generate_model(variety='piecewise parameter estimation')

#%%
# Check ODE model
mg_model.get_ode_eqn()