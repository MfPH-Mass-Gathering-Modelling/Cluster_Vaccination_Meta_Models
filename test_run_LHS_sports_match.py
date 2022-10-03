"""
Creation:
    Author: Martin Grunnill
    Date: 14/09/2022
Description: Test run of world cup match simulation with latin hypercube sampling (LHS) and alternate travel screenings.
    
"""
import dask
#%%
# import packages

import numpy as np
import json
import os.path
import pandas as pd
from tqdm import trange

from simulations.sports_match_sim import SportMatchMGESimulation



#%%
# Setup model
# for now assume England vs Wales Ahmed Bin Ali stadium Tuesday 29th Nov FIFA World Cup 2022
# Team a: England, Team B: Wales


stadium_capacity = 40000 # https://hukoomi.gov.qa/en/article/world-cup-stadiums

# Qatar's population 2021
# https://data.worldbank.org/indicator/SP.POP.TOTL?locations=QA
host_population = 2930524
# From Qatars vaccination data enter populations
total_hosts_vaccinated = 2751485 # https://coronavirus.jhu.edu/region/qatar
hosts_effectively_vaccinated = 1870034 # assume all booster doses are effectively vaccinated https://covid19.moph.gov.qa/EN/Pages/Vaccination-Program-Data.aspx
qatar_cases_past_day = 733.429 # 7 day smoothed 15/09/2022 https://github.com/owid/covid-19-data/tree/master/public/data
host_cases_per_million = (qatar_cases_past_day/host_population)*1e6
UK_total_vaccinated = 50745901 # 2022-09-04 https://github.com/owid/covid-19-data/tree/master/public/data
UK_booster_doses = 40373987 # 2022-09-04 https://github.com/owid/covid-19-data/tree/master/public/data
team_A_proportion_effective = UK_booster_doses/UK_total_vaccinated
team_B_proportion_effective = team_A_proportion_effective
team_A_cases_per_million = 65.368  # UK as whole 7 day smoothed 15/09/2022 https://github.com/owid/covid-19-data/tree/master/public/data
team_B_cases_per_million = team_A_cases_per_million
nu = 0 # change between vaccination groups.

fixed_parameters = {    # Test parameter values - Vaccination parameters
    'p_d': 0.1,
    'p_h': 0.1,
    # These are available at https://docs.google.com/spreadsheets/d/1_XQn8bfPXKA8r1D_rz6Y1LeMTR_Y0RK5Rh4vnxDcQSo/edit?usp=sharing
    # These are placed in a list corresponding to the order of vaccination groups (enter MassGatheringModel.vaccination_groups)
    'alpha': 0, # considering the timeframe of this model alpha is not needed.
    'epsilon_1': 0.5988023952,
    'epsilon_2': 0.4291845494,
    'epsilon_3': 1,
    'epsilon_H': 0.1627*2,
    'gamma_A_1': 0.139*2,
    'gamma_A_2': 0.139*2,
    'gamma_I_1': 0.1627*2,
    'gamma_I_2': 0.1627*2,
    'gamma_H': 0.0826446281,
    'h_effective': 0.957,
    'h_unvaccinated': 0,
    'h_waned':0.815,
    'iota': 0,  # model runtime is to short to consider end of isolation.
    'l_effective':0.7443333333,
    'l_unvaccinated': 0,
    'l_waned': 0.2446666667,
    'nu_1':nu,
    'nu_b':nu,
    'nu_w':nu,
    'omega_G':1, # PCR result delay
    'p_s': 0.724,
    'p_h': 0.10,
    's_effective': 0.7486666667,
    's_unvaccinated': 0,
    's_waned': 0.274,
    'tau_A': 0, # Testing is performed by an event.
    'tau_G': 0, # Testing is performed by an event.
    'theta': 0.342,
    'proportion host tickets': 0.1,
    'staff per ticket': 0.1,
    'Pre-travel test': True,
    'Pre-match test': False,
    'Post-match test': False,
    'test type': 'RTPCR',
    'test sensitivity': 0.96, # https://pubmed.ncbi.nlm.nih.gov/34856308/,
    'team A prop effectively vaccinated': team_A_proportion_effective,
    'team B prop effectively vaccinated': team_B_proportion_effective,
    'team A cases per million': team_A_cases_per_million,
    'team B cases per million': team_B_cases_per_million
}


sport_match_sim = SportMatchMGESimulation(host_population, host_cases_per_million,
                                          total_hosts_vaccinated, hosts_effectively_vaccinated,
                                          stadium_capacity,
                                          team_A_cases_per_million, team_B_cases_per_million,
                                          fixed_parameters=fixed_parameters)

#%%
import pandas as pd
from dask import delayed
import os
from scipy.stats import qmc
import psutil
from dask.diagnostics import ProgressBar
from dask.distributed import Client
# import Quasi-Monte Carlo submodule
import pingouin as pg

number_of_workers = os.cpu_count()  # alter this if you do not want all your computing resources used.

client = Client(processes=False, n_workers=2,threads_per_worker=1)
client.dashboard_link

#%%

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
def df_to_records(df):
    return df.to_dict('records')

@delayed
def df_horizontal_concat(df1, df2):
    return pd.concat([df1, df2], axis=1)

@delayed
def calulate_PRCC(df, parameter, output, covariables):
    param_rank_pcor = pg.partial_corr(df,
                                      x=parameter, y=output,
                                      covar=covariables,
                                      method='spearman')
    return param_rank_pcor.loc['spearman', 'r']

def _run_to_get_out_keys(model_run_method, parameters_df):
    sample = dict(zip(parameters_df['Parameter'], parameters_df['Lower Bound']))
    run_to_get_out_put_keys = model_run_method(sample)
    return list(run_to_get_out_put_keys.keys())

parameters_df = pd.read_csv('Test Parameter distributions.csv')
LHS_obj = qmc.LatinHypercube(len(parameters_df))
model_run_method = sport_match_sim.run_simulation
sample_size = 10
repeats_per_sample_size = 20
output_keys = ['peak infected', 'total infections', 'peak hospitalised', 'total hospitalisations']
y0 = None





#%%
# def run_many_LHS_determine_prcc(parameters_df,
#                                 LHS_obj,
#                                 model_run_method,
#                                 sample_size,
#                                 repeats_per_sample_size,
#  output_keys=None,
#                                 y0=None):
gen_rand_LH_sample = delayed(LHS_obj.random)
model_run_method = delayed(model_run_method)
records_to_df = delayed(pd.DataFrame.from_records)
# format_sample = delayed(format_sample)
prcc_measures = []
# if output_keys is None:
#     output_keys = _run_to_get_out_keys(model_run_method, parameters_df)
# for repeat_num in range(repeats_per_sample_size):
LH_sample = LHS_obj.random(sample_size)
sample_df = format_sample(parameters_df, LH_sample)
focused_results_records = []
samples = sample_df.to_dict('records')
for sample_index in range(sample_size):
    sample = samples[sample_index]
    if y0 is None:
        focused_results_records.append(model_run_method(sample))
    else:
        focused_results_records.append(model_run_method(y0=y0, args=sample))
# records_to_df = dask.persist()
focused_results_records = dask.persist(*focused_results_records)
focused_results_df = records_to_df(focused_results_records)
# focused_results_df.visualize(filename='task graph.pdf')
results = focused_results_df.compute()



#%%



sample_df = df_horizontal_concat(sample_df, focused_results_df)
prcc_measure_enty = {}
    for parameter in parameters_df['Parameter']:
        covariables = [item
                       for item in parameters_df['Parameter']
                       if item != parameter]
        for output in output_keys:
            prcc = calulate_PRCC(sample_df, parameter, output, covariables)
            prcc_measure_enty[parameter + ' on ' + output] = prcc
    prcc_measures.append(prcc_measure_enty)
prcc_measures_df = records_to_df(prcc_measures)
# prcc_measures_df.visualize(filename='task graph.pdf')

prcc_measures_df = prcc_measures_df.compute()

prccs_descriptive_stats = prcc_measures_df.describe()