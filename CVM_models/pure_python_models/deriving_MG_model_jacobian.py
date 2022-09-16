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
from CVM_models.pygom_models.mass_gathering_vaccination import MGModelConstructor
from CVM_models.pure_python_models.derving_model_attributes import AttributeGetter
import json
save_dir = 'C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/Mass Gathering work/Compartment based models/Cluster_Vaccination_Meta_Models/CVM_models/pure_python_models/'


#%%
# Single population 3 dose model.

# get model meta population structure
structures_dir = ('C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/'+
                  'Mass Gathering work/Compartment based models/Cluster_Vaccination_Meta_Models/'+
                  'CVM_models/Model meta population structures/')
with open(structures_dir + "single cluster 3 dose model.json", "r") as json_file:
    group_info=json_file.read()

group_info_single_cluster_3_doses = json.loads(group_info)

#%%

standard_doses_attribs = AttributeGetter(MGModelConstructor, group_info_single_cluster_3_doses,
                                         '1_cluster_3_dose_model',save_dir,
                                         include_observed_states=True)


standard_doses_attribs.save_odes()
standard_doses_attribs.save_jacobian()

#%%
# Testing model.
# get model meta population structure
with open(structures_dir + "general population with testing 3 dose model.json", "r") as json_file:
    group_info=json_file.read()

group_info_testing_with_3_doses = json.loads(group_info)

#%%

testing_3_doses_attribs = AttributeGetter(MGModelConstructor, group_info_testing_with_3_doses,
                                          'testing_3_dose_model',save_dir,
                                          include_observed_states=True)


testing_3_doses_attribs.save_odes()
testing_3_doses_attribs.save_jacobian()

#%%
# Testing model.
# get model meta population structure
with open(structures_dir + "general population with testing 3 dose model.json", "r") as json_file:
    group_info=json_file.read()

group_info_testing_with_3_doses = json.loads(group_info)

#%%

testing_3_doses_attribs = AttributeGetter(MGModelConstructor, group_info_testing_with_3_doses,
                                          'testing_3_dose_model',save_dir,
                                          include_observed_states=True)


testing_3_doses_attribs.save_odes()
testing_3_doses_attribs.save_jacobian()

#%%
# MGE World cup match model.
# get model meta population structure
with open(structures_dir + "World cup MGE.json", "r") as json_file:
    group_info=json_file.read()

group_info_world_cup_MGE = json.loads(group_info)

#%%

world_cup_MGE_attribs = AttributeGetter(MGModelConstructor, group_info_world_cup_MGE,
                                          'World_cup_MGEmodel',save_dir,
                                          include_observed_states=True)


world_cup_MGE_attribs.save_odes()
world_cup_MGE_attribs.save_jacobian()