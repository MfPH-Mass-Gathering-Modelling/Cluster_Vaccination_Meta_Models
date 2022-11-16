"""
Creation:
    Author: Martin Grunnill
    Date: 2022-11-09
Description: Running testing scenarios along with Latin Hypercube Sampling.
    
"""
import pandas as pd
from LH_sampling.load_variables_and_parameters import load_parameters, load_repeated_sample
from LH_sampling.LHS_and_PRCC_parallel import run_samples_in_parrallell
from LH_sampling.gen_LHS_and_simulate_serrialy import format_sample
from scipy.stats import qmc
import os
from simulations.sports_match_sim import SportMatchMGESimulation

if __name__ == '__main__':
    parameters_df, fixed_parameters = load_parameters()
    other_samples_to_repeat = load_repeated_sample()
    sample_size = 275 * len(other_samples_to_repeat)
    save_dir = 'C:/Users/mdgru/OneDrive - York University/Data/World_cup_modelling'
    save_dir = save_dir + '/Assesing testing regimes with LH sample Size ' + str(sample_size) + '2nd run'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    LH_sample_file = save_dir+'/LH sample.csv'
    if os.path.isfile(LH_sample_file):
        sample_df = pd.read_csv(LH_sample_file)
        parameters_sampled = sample_df.columns.to_list()
        parameters_sampled.remove('Sample Number')
    else:
        LHS_obj = qmc.LatinHypercube(len(parameters_df))
        LH_sample = LHS_obj.random(sample_size)
        sample_df, parameters_sampled = format_sample(parameters_df, LH_sample, other_samples_to_repeat)
        sample_df.index.name = 'Sample Number'
        sample_df.reset_index(level=0,inplace=True)
        sample_df.to_csv(LH_sample_file, index=False)


    testing_regimes = {'No Testing': {'Pre-travel test': False,
                                      'Pre-match test': False,
                                      'Post-match test': False},
                       'Pre-travel RTPCR':{'test type': 'RTPCR',
                                           'Pre-travel test':True,
                                           'Pre-match test': False,
                                           'Post-match test': False},
                       'Pre-match RTPCR': {'test type': 'RTPCR',
                                           'Pre-travel test': False,
                                           'Pre-match test': True,
                                           'Post-match test': False},
                       'Post-match RTPCR': {'test type': 'RTPCR',
                                            'Pre-travel test': False,
                                            'Pre-match test': False,
                                            'Post-match test': True},
                       'Pre-travel RA': {'test type': 'RA',
                                         'Pre-travel test': True,
                                         'Pre-match test': False,
                                         'Post-match test': False},
                       'Pre-match RA': {'test type': 'RA',
                                        'Pre-travel test': False,
                                        'Pre-match test': True,
                                        'Post-match test': False},
                       'Post-match RA': {'test type': 'RA',
                                         'Pre-travel test': False,
                                         'Pre-match test': False,
                                         'Post-match test': True}
                       }
    for testing_regime, test_parmeters in testing_regimes.items():
        regime_save_dir = save_dir + '/' + testing_regime
        if not os.path.exists(regime_save_dir):
            os.makedirs(regime_save_dir)
        fixed_parameters.update(test_parmeters)
        model_or_simulation_obj = SportMatchMGESimulation(fixed_parameters=fixed_parameters)
        model_run_method = model_or_simulation_obj.run_simulation
        samples_already_run = []
        for sample_num in range(len(sample_df)):
            focused_output_file = regime_save_dir +'/Focused Outputs and Sample ' + str(sample_num)+ '.csv'
            if os.path.isfile(focused_output_file):
                samples_already_run.append(sample_num)
                focused_output_df = pd.read_csv(LH_sample_file)



        run_samples_in_parrallell(sample_df, model_run_method, save_dir=regime_save_dir, return_full_results=True)





