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
# Get Ontario's vaccination data.
from data_extraction.vaccination_data import ontario
vac_data = ontario()
piecewise_params = {'nu__unvaccinated': {'targets': vac_data.previous_day_at_least_one}, # nu_1 is the rate of vaccination rate for unvaccinated group.
                    'nu__first_dose' : {'targets': vac_data.previous_day_fully_vaccinated}, # nu_3 is the rate of vaccination rate for first_dose group.
                    'nu__second_dose' : {'targets': vac_data.previous_day_3doses}} # nu_1 is the rate of vaccination rate for second_dose_waned group.

#%%
# Ontario’s population reached 14,755,211 on January 1, 2021
# https://www.ontario.ca/page/ontario-demographic-quarterly-highlights-fourth-quarter-2020#:~:text=Ontario’s%20population%20reached%2014%2C755%2C211%20on,quarter%20of%20the%20previous%20year..
starting_population = 14755211
fixed_params = {'N_':starting_population}


#%%
# Start with base model, i.e. no infections just susceptibles running through vaccination groups
# Import base model constructor
from CVM_models.pygom_models.base_vaccination import BaseMultiClusterVacConstructor

# Initalise constructor.
base_model_constructor = BaseMultiClusterVacConstructor(clusters, vaccine_groups)

#%%
# generate pygom model
base_model = base_model_constructor.generate_model(variety='piecewise parameter estimation')

#%%
# Check ODE model
#base_model.get_ode_eqn()


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
y0[0] = starting_population
t = np.arange(len(vac_data.previous_day_at_least_one)-5)
base_model.initial_values = (y0, t[0])

#%%
# Run Model
sol = base_model.integrate(t[1::])

#%%
# Checking model is working as it should.

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