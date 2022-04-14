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

#%%
# Create constructor
clusters = ['']
vaccine_groups = [
    'unvaccinated',
    'first_dose_delay',
    'first_dose',
    'second_dose_delay',
    'second_dose',
    'waned',
    'third_dose_delay',
    'third_dose'
]

single_cluster_constor = MGModelConstructor(clusters, vaccine_groups)

#%%
# Generate actual model.
single_cluster_model = single_cluster_constor.generate_model()

#%%
# Check model is correct - graph
graph_dot = single_cluster_model.get_transition_graph()
graph_dot.render(filename='mg_single_cluster_model_graph', format='pdf')

#%%
# Check model is correct - ODEs
odes = single_cluster_model.get_ode_eqn()


#%%
# Determine models Jacobian
jacobian = single_cluster_model.get_jacobian_eqn()
print(single_cluster_constor.all_states)

#%%
#Convert to dictionary of keys
length = range(single_cluster_model.num_state)
dict_of_keys = {(row,col): str(jacobian[row,col])
                for row in length
                for col in length
                if jacobian[row,col] !=0}


#%%
# Modeify dict of keys for pasting into Jacobian method
dict_of_keys = {coords: value.replace('__', '_') for coords, value in  dict_of_keys.items()}
for symbol in MGModelConstructor.cluster_specific_params:
    dict_of_keys = {coords: value.replace(symbol + '_', symbol) for coords, value in  dict_of_keys.items()}

replacement = {state + '_' + vaccine_group: 'y['+str(index)+']'
               for vaccine_group, state_dict_index in single_cluster_constor.cluste_vaccine_group_state_index[''].items()
               for state, index in state_dict_index.items()}

new_dict_of_keys = {}
for coords, value in dict_of_keys.items():
    for orignal, new in replacement.items():
        value = value.replace(orignal, new)
    new_dict_of_keys[coords] = value

new_dict_of_keys = {coords: value.replace('_A_', '_a_') for coords, value in new_dict_of_keys.items()}
new_dict_of_keys = {coords: value.replace('_I_', '_i_') for coords, value in new_dict_of_keys.items()}
## Now you can copy new_dict_of_keys into your clipboard through your IDE variable viewer.
