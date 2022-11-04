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
from simulations.sports_match_sim import SportMatchMGESimulation
from LH_sampling.load_variables_and_parameters import load_parameters, load_repeated_sample


def format_sample(parameters_df, LH_samples, other_samples_to_repeat=None):
    # if any(~parameters_df['Distribution'].isin(['boolean','uniform'])):
    #     raise ValueError('Only Boolean and Uniform distributions currently supported.')
    samples_df = pd.DataFrame(qmc.scale(LH_samples, parameters_df['Lower Bound'], parameters_df['Upper Bound']),
                              columns=parameters_df.index)
    if other_samples_to_repeat is not None:
        multiple = len(samples_df)/len(other_samples_to_repeat)
        if not multiple.is_integer():
            raise ValueError('LHS sample size devided by length of other_samples_to_repeat must be expressable as an interger value.')
        multiple = int(multiple)
        repeated_samples = pd.concat([other_samples_to_repeat] * multiple, ignore_index=True)
        samples_df = pd.concat([samples_df, repeated_samples], axis=1, ignore_index=True)
        parameters_sampled = parameters_df.index.to_list() + other_samples_to_repeat.columns.to_list()
        samples_df.columns = parameters_sampled
    else:
        parameters_sampled = samples_df.columns.to_list()

    # convert_to_bool = parameters_df.Parameter[parameters_df['Distribution'] == 'boolean']
    # for parameter in convert_to_bool:
    #     samples_df[parameter] = samples_df[parameter] >= 0.5
    return samples_df, parameters_sampled

def calucate_PRCC(sample_df, parameter, output, covariables):
    param_rank_pcor = pg.partial_corr(sample_df,
                                      x=parameter, y=output,
                                      covar=covariables,
                                      method='spearman')
    param_rank_pcor.rename(index={'spearman': parameter + ' on ' + output}, inplace=True)
    return param_rank_pcor
def LHS_PRCC_serial(parameters_df, sample_size, model_run_method,
                    results_csv = None, LHS_obj=None, y0=None, other_samples_to_repeat=None):
    if LHS_obj is None:
        LHS_obj = qmc.LatinHypercube(len(parameters_df))
    LH_sample = LHS_obj.random(sample_size)
    sample_df, parameters_sampled = format_sample(parameters_df, LH_sample, other_samples_to_repeat)
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
    for parameter in parameters_sampled:
        covariables = [item
                       for item in parameters_sampled
                       if item != parameter]
        for output in focused_results_df.columns:
            param_rank_pcor = calucate_PRCC(sample_df, parameter, output, covariables)
            prccs.append(param_rank_pcor)

    prccs = pd.concat(prccs)
    prccs.sort_index(inplace=True)
    if results_csv is not None:
        prccs.to_csv(results_csv)
    else:
        return prccs

if __name__ == '__main__':
    parameters_df, fixed_parameters = load_parameters()
    other_samples_to_repeat = load_repeated_sample()

    sport_match_sim = SportMatchMGESimulation(fixed_parameters=fixed_parameters)
    sample_size = 2*len(other_samples_to_repeat)
    LHS_PRCC_serial(parameters_df=parameters_df,
                    sample_size=sample_size,
                    model_run_method=sport_match_sim.run_simulation,
                    results_csv='PRCCs LH sample size '+str(sample_size)+'.csv',
                    other_samples_to_repeat=other_samples_to_repeat)

            








