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
        save_prcc_stats_file = (save_dir_for_prcc_decriptive_stats +
                                '/PRCC descriptive stats at sample size ' +
                                str(sample_size) + '.csv')
        if not os.path.isfile(save_prcc_stats_file):
            range_of_repeats = range(repeats_per_n)
            for repeat_num in tqdm(range_of_repeats, leave=False, position=0, colour='blue',
                                   desc='LHS resample sample size of '+str(sample_size)):
                prcc_measure_enty = LHS_PRCC_method(parameters_df, sample_size, model_run_method, LHS_obj=LHS_obj, y0=y0)
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
    parameters_csv_file = 'C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/Mass Gathering work/Compartment based models/Cluster_Vaccination_Meta_Models/Test Parameter distributions.csv'
    parameters_df = pd.read_csv(parameters_csv_file)
    # Setup model
    # for now assume England vs Wales Ahmed Bin Ali stadium Tuesday 29th Nov FIFA World Cup 2022
    # Team a: England, Team B: Wales

    stadium_capacity = 40000  # https://hukoomi.gov.qa/en/article/world-cup-stadiums

    # Qatar's population 2021
    # https://data.worldbank.org/indicator/SP.POP.TOTL?locations=QA
    host_population = 2930524
    # From Qatars vaccination data enter populations
    total_hosts_vaccinated = 2751485  # https://coronavirus.jhu.edu/region/qatar
    hosts_effectively_vaccinated = 1870034  # assume all booster doses are effectively vaccinated https://covid19.moph.gov.qa/EN/Pages/Vaccination-Program-Data.aspx
    qatar_cases_past_day = 733.429  # 7 day smoothed 15/09/2022 https://github.com/owid/covid-19-data/tree/master/public/data
    host_cases_per_million = (qatar_cases_past_day / host_population) * 1e6
    UK_total_vaccinated = 50745901  # 2022-09-04 https://github.com/owid/covid-19-data/tree/master/public/data
    UK_booster_doses = 40373987  # 2022-09-04 https://github.com/owid/covid-19-data/tree/master/public/data
    team_A_proportion_effective = UK_booster_doses / UK_total_vaccinated
    team_B_proportion_effective = team_A_proportion_effective
    team_A_cases_per_million = 65.368  # UK as whole 7 day smoothed 15/09/2022 https://github.com/owid/covid-19-data/tree/master/public/data
    team_B_cases_per_million = team_A_cases_per_million
    nu = 0  # change between vaccination groups.

    fixed_parameters = {  # Test parameter values - Vaccination parameters
        'p_d': 0.1,
        'p_h': 0.1,
        # These are available at https://docs.google.com/spreadsheets/d/1_XQn8bfPXKA8r1D_rz6Y1LeMTR_Y0RK5Rh4vnxDcQSo/edit?usp=sharing
        # These are placed in a list corresponding to the order of vaccination groups (enter MassGatheringModel.vaccination_groups)
        'alpha': 0,  # considering the timeframe of this model alpha is not needed.
        'epsilon_1': 0.5988023952,
        'epsilon_2': 0.4291845494,
        'epsilon_3': 1,
        'epsilon_H': 0.1627 * 2,
        'gamma_A_1': 0.139 * 2,
        'gamma_A_2': 0.139 * 2,
        'gamma_I_1': 0.1627 * 2,
        'gamma_I_2': 0.1627 * 2,
        'gamma_H': 0.0826446281,
        'h_effective': 0.957,
        'h_unvaccinated': 0,
        'h_waned': 0.815,
        'iota': 0,  # model runtime is to short to consider end of isolation.
        'l_effective': 0.7443333333,
        'l_unvaccinated': 0,
        'l_waned': 0.2446666667,
        'nu_1': nu,
        'nu_b': nu,
        'nu_w': nu,
        'omega_G': 1,  # PCR result delay
        'p_s': 0.724,
        'p_h': 0.10,
        's_effective': 0.7486666667,
        's_unvaccinated': 0,
        's_waned': 0.274,
        'tau_A': 0,  # Testing is performed by an event.
        'tau_G': 0,  # Testing is performed by an event.
        'theta': 0.342,
        'proportion host tickets': 0.1,
        'staff per ticket': 0.1,
        'Pre-travel test': True,
        'Pre-match test': False,
        'Post-match test': False,
        'test type': 'RTPCR',
        'test sensitivity': 0.96,  # https://pubmed.ncbi.nlm.nih.gov/34856308/,
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
                             save_dir_for_prcc_decriptive_stats='test determining LH sample size')