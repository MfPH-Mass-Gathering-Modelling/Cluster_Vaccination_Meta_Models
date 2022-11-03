"""
Creation:
    Author: Martin Grunnill
    Date: 2022-10-02
Description: Generate Latin-Hypercube Sample (LHS) and run simulations.
    
"""

import pandas as pd
import os
# import Quasi-Monte Carlo submodule
from scipy.stats import qmc
from tqdm.auto import tqdm
import pingouin as pg


def format_sample(parameters_df, LH_samples, other_samples_to_repeat=None):
    # if any(~parameters_df['Distribution'].isin(['boolean','uniform'])):
    #     raise ValueError('Only Boolean and Uniform distributions currently supported.')
    samples_df = pd.DataFrame(qmc.scale(LH_samples, parameters_df['Lower Bound'], parameters_df['Upper Bound']),
                              columns=parameters_df['Parameter'])
    if other_samples_to_repeat is not None:
        multiple = len(samples_df)/len(other_samples_to_repeat)
        if not multiple.is_integer():
            raise ValueError('LHS sample size devided by length of other_samples_to_repeat must be expressable as an interger value.')
        multiple = int(multiple)
        repeated_samples = pd.concat([other_samples_to_repeat] * multiple, ignore_index=True)
        samples_df = pd.concat([samples_df, repeated_samples], axis=1, ignore_index=True)


    # convert_to_bool = parameters_df.Parameter[parameters_df['Distribution'] == 'boolean']
    # for parameter in convert_to_bool:
    #     samples_df[parameter] = samples_df[parameter] >= 0.5
    return samples_df




def calucate_PRCC(sample_df, parameter, output, covariables):
    param_rank_pcor = pg.partial_corr(sample_df,
                                      x=parameter, y=output,
                                      covar=covariables,
                                      method='spearman')
    param_rank_pcor.rename(index={'spearman': parameter + ' on ' + output}, inplace=True)
    return param_rank_pcor
def LHS_PRCC_serial(parameters_df, sample_size, model_run_method, LHS_obj=None, y0=None, other_samples_to_repeat=None):
    if LHS_obj is None:
        LHS_obj = qmc.LatinHypercube(len(parameters_df))
    LH_sample = LHS_obj.random(sample_size)
    sample_df = format_sample(parameters_df, LH_sample, other_samples_to_repeat)
    focused_results_records = []
    samples = sample_df.to_dict('records')
    for sample in tqdm(samples, desc='Simulating LH Sample', position=1, leave=False, colour='green'):
        if y0 is None:
            focused_results_records.append(model_run_method(sample))
        else:
            focused_results_records.append(model_run_method(sample, y0=y0))
    focused_results_df = pd.DataFrame.from_records(focused_results_records)
    sample_df = pd.concat([sample_df, focused_results_df], axis=1)
    prccs = []
    for parameter in parameters_df['Parameter']:
        covariables = [item
                       for item in parameters_df['Parameter']
                       if item != parameter]
        for output in focused_results_df.columns:
            param_rank_pcor = calucate_PRCC(sample_df, parameter, output, covariables)
            prccs.append(param_rank_pcor)
    return pd.concat(prccs)

            








