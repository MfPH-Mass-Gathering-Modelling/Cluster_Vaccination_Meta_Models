"""
Creation:
    Author: Martin Grunnill
    Date: 8/8/2022
Description: Sets up metapopulation structure as outlined in https://www.overleaf.com/read/bsrfhxqhtzkk.
 Metapopulation is then saved as json.
    
"""
#%%
# import packages
import json


#%%
# Setting up states that can be transfered between metapopulations

vaccinable_states = ['S', 'E', 'G_I', 'G_A', 'P_I', 'P_A', 'M_A', 'F_A', 'R']
delay_vaccine_efficacy_states = 'all'
wanning_vaccine_efficacy_states = 'all'
LFD_true_positive_states = ['P_I', 'P_A', 'M_A', 'F_A']
PCR_true_positive_states = ['G_I', 'G_A', 'P_I', 'P_A', 'M_A', 'F_A']
PCR_delay = 'all'

#%%
# Lets start off by a general population cluster
general_population = 'general_population'
to_and_from_cluster = {'cluster': general_population, 'to cluster': general_population}
first_dose = {**to_and_from_cluster,
              'vaccine group': 'unvaccinated', 'to vaccine group': 'first_dose_delay',
              'parameter': 'nu_1', 'states': vaccinable_states}
first_dose_effective = {**to_and_from_cluster,
                        'vaccine group': 'first_dose_delay', 'to vaccine group': 'first_dose_effective',
                        'parameter': 'nu_d', 'states': delay_vaccine_efficacy_states}

second_dose = {**to_and_from_cluster,
               'vaccine group': 'first_dose_effective', 'to vaccine group': 'second_dose_delay',
               'parameter': 'nu_2', 'states': vaccinable_states}
second_dose_effective = {**to_and_from_cluster,
                         'vaccine group': 'second_dose_delay', 'to vaccine group': 'second_dose_effective',
                         'parameter': 'nu_d', 'states': delay_vaccine_efficacy_states}
second_dose_waned = {**to_and_from_cluster,
                     'vaccine group': 'second_dose_effective', 'to vaccine group': 'waned',
                     'parameter': 'nu_w', 'states': wanning_vaccine_efficacy_states}
third_dose = {**to_and_from_cluster,
              'vaccine group': 'waned', 'to vaccine group': 'third_dose_delay',
              'parameter': 'nu_3', 'states': vaccinable_states}
third_dose_effective = {**to_and_from_cluster,
                        'vaccine group': 'third_dose_delay', 'to vaccine group': 'third_dose_effective',
                        'parameter': 'nu_d', 'states': delay_vaccine_efficacy_states}

group_info = [first_dose, first_dose_effective,
              second_dose, second_dose_effective, second_dose_waned,
              third_dose, third_dose_effective
              ]

#%%
# LFD test information
LFD_true_positives = 'LFD_true_positive'
to_and_from_cluster = {'cluster': general_population, 'to cluster': LFD_true_positives}
vaccine_groups = ['unvaccinated',
                  'first_dose_delay', 'first_dose_effective',
                  'second_dose_delay', 'second_dose_effective', 'waned',
                  'third_dose_delay', 'third_dose_effective']

for vaccine_group in vaccine_groups:
    entry = {**to_and_from_cluster,
             'vaccine group': vaccine_group, 'to vaccine group': vaccine_group,
             'parameter': 'tau_A', 'states': LFD_true_positive_states}
    group_info.append(entry)

#%%
# RT_PCR test information
PCR_positive_waiting = 'PCR_positive_waiting'
to_and_from_cluster = {'cluster': general_population, 'to cluster': PCR_positive_waiting}

for vaccine_group in vaccine_groups:
    entry = {**to_and_from_cluster,
             'vaccine group': vaccine_group, 'to vaccine group': vaccine_group,
             'parameter': 'tau_G', 'states': PCR_true_positive_states}
    group_info.append(entry)

PCR_positive = 'PCR_positive'
to_and_from_cluster = {'cluster': PCR_positive_waiting, 'to cluster': PCR_positive}

for vaccine_group in vaccine_groups:
    entry = {**to_and_from_cluster,
             'vaccine group': vaccine_group, 'to vaccine group': vaccine_group,
             'parameter': 'omega', 'states': 'all'}
    group_info.append(entry)



#%%
# Save structure as json file
dir_name = ('C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/'+
            'Mass Gathering work/Compartment based models/Cluster_Vaccination_Meta_Models/'+
            'CVM_models/Model meta population structures/')
with open(dir_name + "general population with testing 3 dose model.json", "w") as outfile:
    json.dump(group_info, outfile)