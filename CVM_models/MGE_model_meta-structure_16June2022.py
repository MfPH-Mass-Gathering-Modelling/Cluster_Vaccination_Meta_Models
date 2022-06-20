"""
Creation:
    Author: Martin Grunnill
    Date: 16/06/2022
Description: Sets up metapopulation structure as outlined in https://www.overleaf.com/read/bsrfhxqhtzkk.
 Metapopulation is then saved as json.
    
"""
#%%
# import packages
import json


#%%
# For now we will use Ontario's vaccination data
from data_extraction.vaccination_data import ontario
vac_data = ontario()
first_vac_dose = vac_data.previous_day_at_least_one.tolist() # needs to be a list to save as json later
second_vac_dose = vac_data.previous_day_fully_vaccinated.tolist() # needs to be a list to save as json later
third_vac_dose = vac_data.previous_day_3doses.tolist() # needs to be a list to save as json later

all_states = 'all'
vaccinable_states = ['S', 'E', 'G_I', 'G_A', 'P_I', 'P_A', 'M_A', 'F_A', 'R']

#%%
# Lets start off by considering only a single cluster
only_cluster = ''
to_and_from_cluster = {'cluster': only_cluster, 'to cluster': only_cluster}
first_dose = {**to_and_from_cluster,
              'vaccine group': 'unvaccinated', 'to vaccine group': 'first_dose_delay',
              'parameter': 'nu_1', 'states': vaccinable_states,
              'piecewise targets': first_vac_dose}
first_dose_effective = {**to_and_from_cluster,
                        'vaccine group': 'first_dose_delay', 'to vaccine group': 'first_dose_effective',
                        'parameter': 'nu_d', 'states': all_states}

second_dose = {**to_and_from_cluster,
               'vaccine group': 'first_dose_effective', 'to vaccine group': 'second_dose_delay',
               'parameter': 'nu_2', 'states': vaccinable_states,
               'piecewise targets': second_vac_dose}
second_dose_effective = {**to_and_from_cluster,
                         'vaccine group': 'second_dose_delay', 'to vaccine group': 'second_dose_effective',
                         'parameter': 'nu_d', 'states': all_states}
second_dose_waned = {**to_and_from_cluster,
                     'vaccine group': 'second_dose_effective', 'to vaccine group': 'waned',
                     'parameter': 'nu_w', 'states': all_states}
third_dose = {**to_and_from_cluster,
              'vaccine group': 'waned', 'to vaccine group': 'third_dose_delay',
              'parameter': 'nu_3', 'states': vaccinable_states,
              'piecewise targets': third_vac_dose}
third_dose_effective = {**to_and_from_cluster,
                        'vaccine group': 'third_dose_delay', 'to vaccine group': 'third_dose_effective',
                        'parameter': 'nu_d', 'states': all_states}

group_info = [first_dose, first_dose_effective,
              second_dose, second_dose_effective, second_dose_waned,
              third_dose, third_dose_effective
              ]


#%%
# Save structure as json file
dir_name = ('C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/'+
            'Mass Gathering work/Compartment based models/Cluster_Vaccination_Meta_Models/'+
            'CVM_models/Model meta population structures/')
with open(dir_name + "single cluster 3 dose model.json", "w") as outfile:
    json.dump(group_info, outfile)