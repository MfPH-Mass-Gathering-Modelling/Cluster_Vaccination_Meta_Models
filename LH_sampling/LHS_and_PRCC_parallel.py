"""
Creation:
    Author: Martin Grunnill
    Date: 2022-10-05
Description:
    
"""
import pandas as pd
import sys
# import Quasi-Monte Carlo submodule
from scipy.stats import qmc
from tqdm.auto import tqdm
import concurrent
from simulations.sports_match_sim import SportMatchMGESimulation
from LH_sampling.gen_LHS_and_simulate_serrialy import format_sample, calucate_PRCC
from LH_sampling.load_variables_and_parameters import load_parameters, load_repeated_sample


def LHS_and_PRCC_parallel(parameters_df,
                          sample_size,
                          model_run_method,
                          results_csv = None,
                          LHS_obj=None,
                          y0=None,
                          other_samples_to_repeat=None):
    if LHS_obj is None:
        LHS_obj = qmc.LatinHypercube(len(parameters_df))
    LH_sample = LHS_obj.random(sample_size)
    sample_df, parameters_sampled = format_sample(parameters_df, LH_sample, other_samples_to_repeat)
    samples = sample_df.to_dict('records')
    with tqdm(total=sample_size,
              desc='Simulating LH Sample',
              position=1,
              leave=False,
              colour='green') as pbar: # add a progress bar.
        with concurrent.futures.ProcessPoolExecutor() as executor: # set up paralisation for simulations
            if y0 is None:
                simlations = [executor.submit(model_run_method, sample) for sample in samples]
            else:
                simlations = [executor.submit(model_run_method, sample, y0=y0) for sample in samples]
            focused_results_records = []
            for simlation in concurrent.futures.as_completed(simlations):
                focused_results_records.append(simlation.result())
                pbar.update(1)
    focused_results_df = pd.DataFrame.from_records(focused_results_records)
    sample_df = pd.concat([sample_df, focused_results_df], axis=1)
    prcc_args = []
    for parameter in parameters_sampled:
        covariables = [item
                       for item in parameters_sampled
                       if item != parameter]
        for output in focused_results_df.columns:
            prcc_args.append((parameter, output, covariables))
    with concurrent.futures.ProcessPoolExecutor() as executor:  # set up paralisation for PRCC calculations
        calculations = [executor.submit(calucate_PRCC, sample_df, parameter, output, covariables)
                        for parameter, output, covariables in prcc_args]
        prccs = []
        for calculation in calculations:
            prccs.append(calculation.result())

    prccs = pd.concat(prccs)
    prccs.sort_index(inplace=True)
    if results_csv is not None:
        prccs.to_csv(results_csv)
    else:
        return prccs

if __name__ == '__main__':
    parameters_df, fixed_parameters = load_parameters()
    other_samples_to_repeat = load_repeated_sample()

    model_or_simulation_obj = SportMatchMGESimulation(fixed_parameters=fixed_parameters)
    model_run_method = model_or_simulation_obj.run_simulation
    sample_size = 2*len(other_samples_to_repeat)
    LHS_and_PRCC_parallel(parameters_df=parameters_df,
                          sample_size=sample_size,
                          model_run_method=model_run_method,
                          results_csv='PRCCs LH sample size '+str(sample_size)+'.csv',
                          other_samples_to_repeat=other_samples_to_repeat)