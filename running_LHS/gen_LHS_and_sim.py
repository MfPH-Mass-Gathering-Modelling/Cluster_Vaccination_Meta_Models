"""
Creation:
    Author: Martin Grunnill
    Date: 2022-10-02
Description: Generate Latin-Hypercube Sample (LHS) and run simulations.
    
"""

import pandas as pd
from dask import delayed
from scipy.stats import qmc
from tqdm import tqdm
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


@delayed
def return_list_without_item(list, to_be_removed):
    return [item for item in list
            if item != to_be_removed]


@delayed
def df_to_records(df):
    return df.to_dict('records')


@delayed
def df_columns_to_list(df):
    return list(df.columns)


@delayed
def df_horizontal_concat(df1, df2):
    return pd.concat([df1, df2], axis=1)


@delayed
def calulate_PRCC(df, parameter, output, covariables):
    param_rank_pcor = pg.partial_corr(df,
                                      x=parameter, y=output,
                                      covar=covariables,
                                      method='spearman')
    return {parameter + ' on ' + output: param_rank_pcor.loc['spearman', 'r']}

def run_many_LHS_determine_prcc(parameters_df,
                                LHS_obj,
                                model_run_method,
                                sample_size,
                                repeats_per_n,
                                y0=None):
    gen_rand_LH_sample = delayed(LHS_obj.random)
    model_run_method = delayed(model_run_method)
    records_to_df = delayed(pd.DataFrame.from_records)
    prcc_measures = []
    for repeat_num in range(repeats_per_n):
        LH_sample = gen_rand_LH_sample(sample_size)
        sample_df = format_sample(parameters_df, LH_sample)
        focused_results_records = []
        samples = df_to_records(sample_df)
        for sample in samples:
            if y0 is None:
                focused_results_records.append(model_run_method(sample))
            else:
                focused_results_records.append(model_run_method(y0=y0, args=sample))
        focused_results_df = records_to_df(focused_results_records)
        outputs = df_columns_to_list(focused_results_df)
        sample_df = df_horizontal_concat(sample_df, focused_results_df)
        prcc_measure_enty = {}
        for parameter in parameters_df['Parameter']:
            covariables = return_list_without_item(parameters_df['Parameter'])
            for output in outputs:
                prcc = calulate_PRCC(sample_df, parameter, output, covariables)
                prcc_measure_enty.update(prcc)
        prcc_measures.append(prcc_measure_enty)
    prcc_measures_df = records_to_df(prcc_measures)
    prcc_measures_df = prcc_measures_df.compute()
    return prcc_measures_df.describe()


def LHS_determine_sample_size(parameters_df,
                              model_run_method,
                              start_n,
                              n_increase,
                              repeats_per_n,
                              std_to_mean_aim,
                              y0=None):
    parameters = list(parameters_df.Parameter)
    LHS_obj = qmc.LatinHypercube(len(parameters_df))
    sample_size = start_n
    std_to_mean_aim_reached = False
    def df_std_to_abs_mean(df):
        std = df.std()
        mean = df.mean()
        abs_mean = mean.abs()
        return std.divide(abs_mean)

    while not std_to_mean_aim_reached:
        prcc_measures = []
        for repeat_num in range(repeats_per_n):
            LH_sample = gen_rand_LH_sample(sample_size, LHS_obj)
            sample_df = format_sample(LH_sample)
            focused_results_records = []
            samples = df_to_records(sample_df)
            for sample in samples:
                if y0 is None:
                    focused_results_records.append(model_run_method(sample))
                else:
                    focused_results_records.append(model_run_method(y0=y0, args=sample))
            focused_results_df = records_to_df(focused_results_records)
            outputs = df_columns_to_list(focused_results_df)
            sample_df = df_horizontal_concat(sample_df, focused_results_df)
            prcc_measure_enty = {}
            for parameter in parameters:
                covariables = return_list_without_item(parameters)
                for output in outputs:
                    prcc = calulate_PRCC(sample_df, parameter, output, covariables)
                    prcc_measure_enty.update(prcc)
            prcc_measures.append(prcc_measure_enty)
        prcc_measures_df = records_to_df(prcc_measures)
        prcc_measures_df = prcc_measures_df.compute()
        prcc_std_to_abs_mean = df_std_to_abs_mean(prcc_measures_df)
        if any(prcc_std_to_abs_mean.gt(std_to_mean_aim)):
            sample_size += n_increase
        else:
            std_to_mean_aim_reached = True

    return sample_size

def LHS_PRCC_serial(parameters_df, sample_size, model_run_method, y0=None):
    LHS_obj = qmc.LatinHypercube(len(parameters_df))
    LH_sample = LHS_obj.random(sample_size)
    sample_df = format_sample(parameters_df, LH_sample)
    focused_results_records = []
    samples = sample_df.to_dict('records')
    for sample in tqdm(samples, desc='Sample'):
        if y0 is None:
            focused_results_records.append(model_run_method(sample))
        else:
            focused_results_records.append(model_run_method(y0=y0, args=sample))
    focused_results_df = pd.DataFrame.from_records(focused_results_records)
    sample_df = pd.concat([sample_df, focused_results_df], axis=1)
    prccs = {}
    for parameter in parameters_df['Parameter']:
        covariables = [item
                       for item in parameters_df['Parameter']
                       if item != parameter]
        for output in focused_results_df.columns:
            param_rank_pcor = pg.partial_corr(sample_df,
                                              x=parameter, y=output,
                                              covar=covariables,
                                              method='spearman')
            prccs[parameter + ' on ' + output] = param_rank_pcor.loc['spearman', 'r']
    return prccs

            








