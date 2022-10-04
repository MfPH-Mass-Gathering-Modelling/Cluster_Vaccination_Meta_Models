"""
Creation:
    Author: Martin Grunnill
    Date: 2022-10-02
Description: Generate Latin-Hypercube Sample (LHS) and run simulations.
    
"""

import pandas as pd
import os
from scipy.stats import qmc
from tqdm.auto import tqdm
from multiprocessing import Pool
from numbers import Number
from dask.diagnostics import ProgressBar
# import Quasi-Monte Carlo submodule
import pingouin as pg


def format_sample(parameters_df, LH_samples):
    if any(~parameters_df['Distribution'].isin(['boolean','uniform'])):
        raise ValueError('Only Boolean and Uniform distributions currently supported.')
    samples_df = pd.DataFrame(qmc.scale(LH_samples, parameters_df['Lower Bound'], parameters_df['Upper Bound']),
                              columns=parameters_df['Parameter'])
    convert_to_bool = parameters_df.Parameter[parameters_df['Distribution'] == 'boolean']
    for parameter in convert_to_bool:
        samples_df[parameter] = samples_df[parameter] >= 0.5
    return samples_df


def LHS_determine_sample_size(parameters_df,
                              model_run_method,
                              start_n,
                              repeats_per_n,
                              std_aim,
                              attempts_to_make=float('inf'),
                              n_increase_addition = None,
                              n_increase_multiple = None,
                              save_dir_for_prcc_decriptive_stats =None,
                              y0=None):
    if n_increase_addition is not None:
        if not isinstance(n_increase_addition,int):
            raise TypeError('n_increase_addition must be an interger > 0.')
        if n_increase_addition <0:
            raise ValueError('n_increase_addition must be > 0.')
    elif n_increase_multiple is not None:
        if not isinstance(n_increase_multiple, (int, float)):
            raise TypeError('n_increase_multiple must be an  or float > 1.')
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
        range_of_repeats = range(repeats_per_n)
        for repeat_num in tqdm(range_of_repeats, leave=False, position=0, colour='blue',
                               desc='LHS resample sample size of '+str(sample_size)):
            prcc_measure_enty = LHS_PRCC_serial(parameters_df, sample_size, model_run_method, LHS_obj=LHS_obj, y0=y0)
            prcc_measures.append(prcc_measure_enty['r'])
        prcc_measures_df = pd.concat(prcc_measures, axis=1)
        prcc_measures_df.columns = range_of_repeats
        prcc_measures_df = prcc_measures_df.transpose(copy=True)
        prcc_decriptive_stats = prcc_measures_df.describe()
        if save_dir_for_prcc_decriptive_stats is not None:
            prcc_decriptive_stats.to_csv(save_dir_for_prcc_decriptive_stats +
                                         '/PRCC descriptive stats at sample size ' +
                                         str(sample_size) + '.csv')
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

def LHS_PRCC_serial(parameters_df, sample_size, model_run_method, LHS_obj=None, y0=None):
    if LHS_obj is None:
        LHS_obj = qmc.LatinHypercube(len(parameters_df))
    LH_sample = LHS_obj.random(sample_size)
    sample_df = format_sample(parameters_df, LH_sample)
    focused_results_records = []
    samples = sample_df.to_dict('records')
    for sample in tqdm(samples, desc='Simulating LH Sample', position=1, leave=False, colour='green'):
        if y0 is None:
            focused_results_records.append(model_run_method(sample))
        else:
            focused_results_records.append(model_run_method(y0=y0, args=sample))
    focused_results_df = pd.DataFrame.from_records(focused_results_records)
    sample_df = pd.concat([sample_df, focused_results_df], axis=1)
    prccs = []
    for parameter in parameters_df['Parameter']:
        covariables = [item
                       for item in parameters_df['Parameter']
                       if item != parameter]
        for output in focused_results_df.columns:
            param_rank_pcor = pg.partial_corr(sample_df,
                                              x=parameter, y=output,
                                              covar=covariables,
                                              method='spearman')
            param_rank_pcor.rename(index={'spearman': parameter + ' on ' + output},inplace=True)
            prccs.append(param_rank_pcor)
    return pd.concat(prccs)

            
def LHS_PRCC_parallel(parameters_df, sample_size, model_run_method, cores_to_use, LHS_obj=None, y0=None):
    if LHS_obj is None:
        LHS_obj = qmc.LatinHypercube(len(parameters_df))
    pool = Pool(processes=cores_to_use)
    LH_sample = LHS_obj.random(sample_size)
    sample_df = format_sample(parameters_df, LH_sample)
    focused_results_records = []
    samples = sample_df.to_dict('records')
    for sample in tqdm(samples, desc='Simulating LH Sample', position=1, leave=False, colour='green'):
        if y0 is None:
            focused_results_records.append(model_run_method(sample))
        else:
            focused_results_records.append(model_run_method(y0=y0, args=sample))
    focused_results_df = pd.DataFrame.from_records(focused_results_records)
    sample_df = pd.concat([sample_df, focused_results_df], axis=1)
    prccs = []
    for parameter in parameters_df['Parameter']:
        covariables = [item
                       for item in parameters_df['Parameter']
                       if item != parameter]
        for output in focused_results_df.columns:
            param_rank_pcor = pg.partial_corr(sample_df,
                                              x=parameter, y=output,
                                              covar=covariables,
                                              method='spearman')
            param_rank_pcor.rename(index={'spearman': parameter + ' on ' + output},inplace=True)
            prccs.append(param_rank_pcor)
    return pd.concat(prccs)







