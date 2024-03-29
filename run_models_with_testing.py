"""
Creation:
    Author: Martin Grunnill
    Date: 01/04/2022
Description: Example code for running piecewise vaccination mass gathering model.
    
"""

#%%
# import packages

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import copy
import pandas as pd


# %%
# import model
from CVM_models.pure_python_models.mass_gathering_piecewise_vaccination import MassGatheringModel
# import transfer class
from event_handling.event_que import EventQueue
from cleaning_up_results.results_to_dfs import results_array_to_df, results_df_pivoted
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


mg_model = MassGatheringModel(group_info)

#%%
# If available attach Jacobian. Note this has to avaluated for every different meta-population structure.
# See CVM_models/pure_python_models/deriving_MG_model_jacobian.py
pure_py_mods_dir = ('C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/Mass Gathering work/'+
                    'Compartment based models/Cluster_Vaccination_Meta_Models/CVM_models/pure_python_models/')

json_file = pure_py_mods_dir+"testing_3_dose_model_jacobian.json"

mg_model.load_dok_jacobian(json_file)

# %%
# Test parameter values
beta = 1.759782  # may change conform to Ontario's 1st wave see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8454019/pdf/CCDR-47-329.pdf
kappa = 0.25
nu = 0 # Assume no new vaccinations.
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
    'nu_1':nu,
    'nu_2':nu,
    'nu_3':nu,
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
    'omega': 1,
    'tau_A': 0,
    'tau_G': 0,
    'iota':1/10
}


mg_model.parameters = test_params

#%%
# Setting up population
# Qatar's population 2021
# https://data.worldbank.org/indicator/SP.POP.TOTL?locations=QA
qatars_population = 2930524
# From Qatars vaccination data enter populations
full_dose = 2751485 # https://coronavirus.jhu.edu/region/qatar
booster_dose = 1870034 # https://covid19.moph.gov.qa/EN/Pages/Vaccination-Program-Data.aspx
unvaccinated = qatars_population - full_dose

y0 = np.zeros(mg_model.num_state)
# for simplicity lets assume no one has had infection in the last 6 months 
state_being_populated = 'S'
y0[mg_model.state_index['general_population']['unvaccinated'][state_being_populated ]] = unvaccinated
# Lets assume that everybody who has had a full dose is waned.
y0[mg_model.state_index['general_population']['waned'][state_being_populated ]] = full_dose
# And that all booster doses have taken effect.
y0[mg_model.state_index['general_population']['third_dose_effective'][state_being_populated]] = booster_dose 

# Add some early stage infecteds to unvaccinated population
infection_seed = 100
y0[mg_model.state_index['general_population']['unvaccinated']['E']] = infection_seed


#%%
# Setting up transfer infomation for people being tested
# need to know model end time
end_time = 100
lfd_sensitivity = 0.78 # https://bmcinfectdis.biomedcentral.com/articles/10.1186/s12879-021-06528-3#:~:text=This%20systematic%20review%20has%20identified,one%20single%20centred%20trial%20(BD
lfd_times = range(0,end_time,3)
lfd_transfer_info = mg_model.group_transition_params_dict['tau_A']
lfd_from_index = []
lfd_to_index = []
for vaccine_group_transfer_info in lfd_transfer_info:
    lfd_from_index += vaccine_group_transfer_info['from_index']
    lfd_to_index +=  vaccine_group_transfer_info['to_index']
transfer_info_dict = {'LFD test': {'proportion': lfd_sensitivity,
                                   'from_index': lfd_from_index,
                                   'to_index': lfd_to_index,
                                   'times': lfd_times}}
rtpcr_sensitivity = 0.96 # https://pubmed.ncbi.nlm.nih.gov/34856308/
rtpcr_transfer_info = mg_model.group_transition_params_dict['tau_G']
rtpcr_times = range(0,end_time,7)
rtpcr_from_index = []
rtpcr_to_index = []
for vaccine_group_transfer_info in rtpcr_transfer_info:
    rtpcr_from_index += vaccine_group_transfer_info['from_index']
    rtpcr_to_index += vaccine_group_transfer_info['to_index']
transfer_info_dict['RTPCR test'] = {'proportion': rtpcr_sensitivity,
                                    'from_index': rtpcr_from_index,
                                    'to_index': rtpcr_to_index,
                                    'times': rtpcr_times}

transfers_scafold = EventQueue(transfer_info_dict)

#%%
# Runninng mg_model
solution, transfers_df = transfers_scafold.run_simulation(mg_model.integrate, y0, end_time)

#%%
# Checking mg_model is working as it should.


sol_df = results_array_to_df(solution, mg_model.state_index)


#%% Conversion of dataframe for use in seaborn plotting package

sol_pivoted = results_df_pivoted(sol_df)



#%%
#Plotting graph of states accross vaccine groups
graph_states_accross_groups = sns.FacetGrid(sol_pivoted, col='cluster', row='state', sharey=False)
graph_states_accross_groups.map(sns.lineplot, 'time', 'population')
graph_states_accross_groups.add_legend()
plt.show()


#%%
# Lets check that susceptible and early stage infections for tested cluster seen in the later part of the simulation
# are those who have lost natural immunity. This can be done by setting wanning of immunity to 0.
params_0_alpha = copy.deepcopy(test_params)
params_0_alpha['alpha'] = 0
mg_model.parameters = params_0_alpha
solution_0_alpha, transfers_df = transfers_scafold.run_simulation(mg_model.integrate, y0, end_time)
sol_0_alpha_df = results_array_to_df(solution_0_alpha, mg_model.state_index)
sol_0_alpha_pivoted = results_df_pivoted(sol_0_alpha_df)
graph_states_accross_groups = sns.FacetGrid(sol_0_alpha_pivoted, col='cluster', row='state', sharey=False)
graph_states_accross_groups.map(sns.lineplot, 'time', 'population')
graph_states_accross_groups.add_legend()
plt.show()


# Great now one in the test clusters who are susceptible or in an early stage of infection.

#%%
# Check that without people being transfered between compartments does TranfersAtTsScafold.run_simulation produce
# the same restuls as model.intergrate.

# Set all transfer events to 0 being transfered.
transfers_scafold.make_events_nullevents('all')

mg_model.parameters = test_params
scafold_solution, transfers_df, scafold_info_dict = transfers_scafold.run_simulation(mg_model.integrate,
                                                                                     y0,
                                                                                     end_time,
                                                                                     full_output=True)

time = np.arange(0,end_time+1,1)
intergrate_solution, intergrate_info_dict = mg_model.integrate(y0,time, full_output=True)

intergrate_solution_df = results_array_to_df(intergrate_solution, mg_model.state_index)
intergrate_solution_df['time'] = time
scafold_solution_df = results_array_to_df(scafold_solution, mg_model.state_index)
scafold_solution_df['time'] = time

# compare step sizes
intergrate_info_step_size = pd.Series(intergrate_info_dict['hu'])
intergrate_step_size_desciribe = intergrate_info_step_size.describe()


scafold_step_sizes = []
for scafold_info_sub_dict in scafold_info_dict.values():
    scafold_step_sizes.append(scafold_info_sub_dict['hu'])
scafold_step_sizes = np.concatenate(scafold_step_sizes,axis=0)


scafold_step_sizes = pd.Series(scafold_step_sizes)
scafold_step_sizes_describe = scafold_step_sizes.describe()

intergrate_step_size_desciribe
scafold_step_sizes_describe
# Step sizes seem to be much smaller for scafold method. I beleive this means that the safold method might be more precise.

#%% Lets check if the intergrator run on its own with same time chunks produces the same results as the scafold.
time_chunks = list(scafold_info_dict.keys())
simulation_step = 1
intergrate_chunk_solutions = []
info_dict = {}
solution_at_t = y0
for time_chunk in time_chunks:
    start_time, end_time = time_chunk
    current_t_range = np.arange(start_time, end_time+simulation_step, simulation_step)
    y_over_current_t_range, info_sub_dict = mg_model.integrate(solution_at_t, current_t_range, full_output=True)
    range_as_list = current_t_range.tolist()
    time_points = (range_as_list[0], range_as_list[-1])
    info_dict[time_points] = info_sub_dict
    solution_at_t = y_over_current_t_range[-1, :]
    if time_chunk != time_chunks[-1]:
        intergrate_chunk_solutions.append(y_over_current_t_range[:-1, :])
    else:
        intergrate_chunk_solutions.append(y_over_current_t_range)

intergrate_chunk_solutions = np.vstack(intergrate_chunk_solutions)

# Compare intergrate_chunk_solutions with scafold_solution_df to see that results are the same.