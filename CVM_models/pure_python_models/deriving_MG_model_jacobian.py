"""
Creation:
    Author: Martin Grunnill
    Date: 05/04/2022
Description: Derivation of the Jacobian matrix of ODEs for the mass gathering model
             for a single cluster. This needed for using faster intergration methods
             for the single cluster version of the model. In order to do this I am
             going to construct a single cluster pygom version of the model.

"""

#%%
# import packages
import pygom
from CVM_models.pygom_models.mass_gathering_vaccination import MGModelConstructor
import json
import os
#abspath = os.path.abspath(__file__)
#dir_name = os.path.dirname(abspath) +'/'
dir_name = 'C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/Mass Gathering work/Compartment based models/Cluster_Vaccination_Meta_Models/CVM_models/pure_python_models/'

#%%
# Create constructor
clusters = ['']
vaccine_groups = [
    'unvaccinated',
    'first_dose_delay',
    'first_dose',
    'second_dose_delay',
    'second_dose',
    'waned', # do not change this to
    'third_dose_delay',
    'third_dose'
]

single_cluster_constor = MGModelConstructor(clusters, vaccine_groups)

#%%
# Generate actual model.
single_cluster_model = single_cluster_constor.generate_model()

#%%
# Check model is correct - graph
# graph_dot = single_cluster_model.get_transition_graph()
# graph_dot.render(filename='mg_single_cluster_model_graph', format='pdf')

#%%
# Check model is correct - ODEs
odes = single_cluster_model.get_ode_eqn()


#%%
# Determine models Jacobian
jacobian = single_cluster_model.get_jacobian_eqn()
print(single_cluster_constor.all_states)

#%%
#Convert to dictionary of keys
def conv_sym_mat_to_dok_sym(sym_matrix):
    rows, cols = sym_matrix.shape
    dict_of_keys = {(row,col): str(sym_matrix[row,col])
                    for row in range(rows)
                    for col in range(cols)
                    if sym_matrix[row,col] !=0}
    return dict_of_keys

jacobian_dok = conv_sym_mat_to_dok_sym(jacobian)

#%%
# Modify dict of keys for saving as json and use in methods.
def mod_dok(dict_of_keys):
    dict_of_keys = {coords: value.replace('__', '_') for coords, value in dict_of_keys.items()}
    for symbol in MGModelConstructor.cluster_specific_params:
        dict_of_keys = {coords: value.replace(symbol + '_', symbol) for coords, value in  dict_of_keys.items()}

    replacement = {state + '_' + vaccine_group: 'state_value['+str(index)+']'
                   for vaccine_group, state_dict_index in single_cluster_constor.cluste_vaccine_group_state_index[''].items()
                   for state, index in state_dict_index.items()}

    new_dict_of_keys = {}
    for coords, value in dict_of_keys.items():
        for orignal, new in replacement.items():
            value = value.replace(orignal, new)
        new_dict_of_keys[coords] = value

    return {str(key): value for key, value in new_dict_of_keys.items()} # json keys cannot be tuples for this so convert to string:
jacobian_dok = mod_dok(jacobian_dok)

# %%
# Write output to json file.
save_location = dir_name
with open(save_location + "MG_model_jacobian.json", "w") as outfile:
    json.dump(jacobian_dok, outfile)


# %%

gradient = single_cluster_model.get_grad_eqn()

# %%
# convert and save gradient.
def collapse_and_mod_sym_mat(sym_matrix):
    dok = conv_sym_mat_to_dok_sym(sym_matrix)
    return mod_dok(dok)

# convert and save gradient.
gradient_dok = collapse_and_mod_sym_mat(gradient)

with open(save_location + "MG_model_ode_gradient.json", "w") as outfile2:
    json.dump(gradient_dok, outfile2)

#%%
gradient_jacobian = single_cluster_model.get_grad_jacobian_eqn()
#%%

# convert and save gradients jacobian.
gradient_jacobian_dok = collapse_and_mod_sym_mat(gradient_jacobian)

with open(save_location + "MG_model_ode_gradients_jacobian.json", "w") as outfile3:
    json.dump(gradient_jacobian_dok, outfile3)

#%%
diff_jacobian = single_cluster_model.get_diff_jacobian_eqn()
#%%

# convert and save gradients jacobian.
diff_jacobian_dok = collapse_and_mod_sym_mat(diff_jacobian)

with open(save_location + "MG_model_ode_diff_jacobian.json", "w") as outfile4:
    json.dump(diff_jacobian_dok, outfile4)