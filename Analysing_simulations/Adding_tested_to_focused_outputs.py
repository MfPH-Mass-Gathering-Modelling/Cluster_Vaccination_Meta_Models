"""
Creation:
    Author: Martin Grunnill
    Date: 2022-11-14
Description: Initial simulation of through LHS forgot to include capturing tested in focused outputs.
    
"""
#%%
import pandas as pd
import os
from LH_sampling.load_variables_and_parameters import load_parameters, load_repeated_sample
from LH_sampling.LHS_and_PRCC_parallel import run_samples_in_parrallell, PRCC_parallel
from LH_sampling.gen_LHS_and_simulate_serrialy import format_sample, calucate_PRCC

#%%
parameters_df, fixed_parameters = load_parameters()
other_samples_to_repeat = load_repeated_sample()
sample_size = 275 * len(other_samples_to_repeat)
save_dir = 'C:/Users/mdgru/OneDrive - York University/Data/World_cup_modelling'
save_dir = save_dir + '/Assesing testing regimes with LH sample Size' + str(sample_size)

#%%
regime_names
