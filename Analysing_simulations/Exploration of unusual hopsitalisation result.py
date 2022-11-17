"""
Creation:
    Author: Martin Grunnill
    Date: 2022-11-17
Description:
    
"""
#%%
import pandas as pd
import numpy as np
import copy
import os


sample_size = 12375
data_dir =  'C:/Users/mdgru/OneDrive - York University/Data/World_cup_modelling'
data_dir = data_dir + '/Assesing testing regimes with LH sample Size ' + str(sample_size)+'/'
no_testing_dir = data_dir+'No testing/'
post_match_ra_dir = data_dir+'Post-match RA/'
post_match_rtpcr_dir = data_dir+'Post-match RTPCR/'
LH_sample = pd.read_csv(data_dir+'LH sample.csv')

#%%
min_hopsitalisation = LH_sample.epsilon_H.min()
max_hopsitalisation = LH_sample.epsilon_H.max()
min_hopsital_recovery = LH_sample.gamma_H.min()
max_hopsital_recovery = LH_sample.gamma_H.max()

sample_min_hospitalisation = LH_sample['Sample Number'][LH_sample.epsilon_H==min_hopsitalisation]
sample_min_hospitalisation = sample_min_hospitalisation.tolist()[0]
sample_min_recovery = LH_sample['Sample Number'][LH_sample.gamma_H==min_hopsital_recovery]
sample_min_recovery = sample_min_recovery.tolist()[0]

#%%
looking_at = sample_min_hospitalisation
days = np.arange(-2, 50+0.5, 0.5)
# comparing testing with no testing
no_testing = pd.read_csv(no_testing_dir+'Solution '+ str(looking_at)+'.csv',header=[0, 1,2])
no_testing.index = days
post_ra = pd.read_csv(post_match_ra_dir +'Solution '+ str(looking_at)+'.csv',header=[0, 1,2])
post_ra.index = days


post_ra_population = post_ra.iloc[:, :-2].sum(axis=1)
no_testing_population = no_testing.iloc[:, :-2].sum(axis=1)

post_ra_states = post_ra.groupby(level=2,axis=1).sum()
# Big than should be transfer to M_H day 6.5. Also M_I drops on same day. Maybe moving people to wrong index.
transfers = pd.read_csv(post_match_ra_dir+'Event que transfers '+ str(looking_at)+'.csv')

#%%
# just check nothing similar is going on with RT PCRs
post_rtpcr = pd.read_csv(post_match_rtpcr_dir +'Solution '+ str(looking_at)+'.csv',header=[0, 1,2])
post_rtpcr.index = days
post_rtpcr_population = post_rtpcr.iloc[:, :-2].sum(axis=1)
post_rtpcr_states = post_rtpcr.groupby(level=2,axis=1).sum()
# Good the same rapid increase in M_H is not happening

#%% simulation and debug
from LH_sampling.load_variables_and_parameters import load_parameters
from simulations.sports_match_sim import SportMatchMGESimulation
parameters_df, fixed_parameters = load_parameters()
fixed_parameters.update({'test type': 'RA',
                         'Pre-travel test': False,
                         'Pre-match test': False,
                         'Post-match test': True})
sample = LH_sample.iloc[looking_at,:].to_dict()
model_or_simulation_obj = SportMatchMGESimulation(fixed_parameters=fixed_parameters)
ouptuts = model_or_simulation_obj.run_simulation(sample)