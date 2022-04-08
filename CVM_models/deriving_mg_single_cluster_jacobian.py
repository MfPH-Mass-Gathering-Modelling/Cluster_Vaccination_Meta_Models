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
from CVM_models.mass_gathering_fixed_vaccination import MGModelConstructor

#%%
# Create constructor
clusters = ['']
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

f_of_index = single_cluster_constor.all_states.index('S__first_dose')
x_of_index =single_cluster_constor.all_states.index('S__unvaccinated')
test = jacobian[f_of_index,x_of_index]
print(test)