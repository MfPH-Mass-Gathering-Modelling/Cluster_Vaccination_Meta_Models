"""
Creation:
    Author: Martin Grunnill
    Date: 2022-10-05
Description:
    
"""
import pandas as pd
import os
# import Quasi-Monte Carlo submodule
from scipy.stats import qmc
from simulations.sports_match_sim import SportMatchMGESimulation
from tqdm import tqdm
from LH_sampling.LHS_and_PRCC_parallel import LHS_and_PRCC_parallel
from LH_sampling.load_variables_and_parameters import load_parameters, load_repeated_sample

def determine_LH_sample_size(parameters_df,
                             model_run_method,
                             start_n,
                             repeats_per_n,
                             std_aim,
                             LHS_PRCC_method,
                             save_dir_for_prcc_decriptive_stats,
                             attempts_to_make=float('inf'),
                             n_increase_addition = None,
                             n_increase_multiple = None,
                             y0=None,
                             other_samples_to_repeat=None):
    if n_increase_addition is not None:
        if not isinstance(n_increase_addition,int):
            if isinstance(n_increase_addition, float):
                if not n_increase_addition.is_integer():
                    raise TypeError('n_increase_addition must be an interger > 0.')
        if n_increase_addition <0:
            raise ValueError('n_increase_addition must be > 0.')
    elif n_increase_multiple is not None:
        if not isinstance(n_increase_multiple, (int, float)):
            raise TypeError('n_increase_multiple must be an int or float > 1.')
        if n_increase_multiple <1:
            raise ValueError('n_increase_multiple must be > 1.')
    else:
        raise AssertionError('Either n_increase_addition or n_increase_multiple must be given.')

    if save_dir_for_prcc_decriptive_stats is not None:
        if not os.path.exists(save_dir_for_prcc_decriptive_stats):
            os.makedirs(save_dir_for_prcc_decriptive_stats)

    LHS_obj = qmc.LatinHypercube(len(parameters_df))
    sample_size = start_n
    std_to_mean_aim_reached = False
    attempts_made = 0

    while attempts_made < attempts_to_make and not std_to_mean_aim_reached:
        prcc_measures = []
        save_prcc_stats_file = (save_dir_for_prcc_decriptive_stats +
                                '/PRCC descriptive stats at sample size ' +
                                str(sample_size) + '.csv')
        if not os.path.isfile(save_prcc_stats_file):
            range_of_repeats = range(repeats_per_n)
            for repeat_num in tqdm(range_of_repeats, leave=False, position=0, colour='blue',
                                   desc='LHS resample sample size of '+str(sample_size)):
                if other_samples_to_repeat is not None:
                    prcc_measure_enty = LHS_PRCC_method(parameters_df, sample_size, model_run_method,
                                                        LHS_obj=LHS_obj, y0=y0,
                                                        other_samples_to_repeat=other_samples_to_repeat
                                                        )
                else:
                    prcc_measure_enty = LHS_PRCC_method(parameters_df, sample_size, model_run_method,
                                                        LHS_obj=LHS_obj, y0=y0)
                prcc_measures.append(prcc_measure_enty['r'])
            prcc_measures_df = pd.concat(prcc_measures, axis=1)
            prcc_measures_df.columns = range_of_repeats
            prcc_measures_df = prcc_measures_df.transpose(copy=True)
            prcc_decriptive_stats = prcc_measures_df.describe()
            prcc_decriptive_stats.to_csv(save_prcc_stats_file)
        else:
            prcc_decriptive_stats = pd.read_csv(save_prcc_stats_file, index_col=0)

        max_std = prcc_decriptive_stats.loc['std', :].max()
        if max_std > std_aim:
            if n_increase_addition is not None:
                sample_size += n_increase_addition
            elif n_increase_multiple is not None:
                sample_size *= n_increase_multiple
        else:
            std_to_mean_aim_reached = True
        attempts_made += 1

    return sample_size

if __name__ == '__main__':
    parameters_df, fixed_parameters = load_parameters()
    other_samples_to_repeat = load_repeated_sample()


    sport_match_sim = SportMatchMGESimulation(fixed_parameters=fixed_parameters)


    repeats_per_n = 30
    start_n = 1000
    std_aim = 0.001
    model_run_method = sport_match_sim.run_simulation
    determine_LH_sample_size(parameters_df=parameters_df,
                             model_run_method=model_run_method,
                             start_n=start_n,
                             repeats_per_n=repeats_per_n,
                             std_aim=std_aim,
                             LHS_PRCC_method=LHS_and_PRCC_parallel,
                             n_increase_multiple=2,
                             save_dir_for_prcc_decriptive_stats='test determining LH sample size',
                             other_samples_to_repeat =other_samples_to_repeat)