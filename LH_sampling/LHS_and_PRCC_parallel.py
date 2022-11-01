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


def LHS_and_PRCC_parallel(parameters_df,
                          sample_size,
                          model_run_method,
                          results_csv = None,
                          LHS_obj=None,
                          y0=None):
    if LHS_obj is None:
        LHS_obj = qmc.LatinHypercube(len(parameters_df))
    LH_sample = LHS_obj.random(sample_size)
    sample_df = format_sample(parameters_df, LH_sample)
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
    for parameter in parameters_df['Parameter']:
        covariables = [item
                       for item in parameters_df['Parameter']
                       if item != parameter]
        for output in focused_results_df.columns:
            prcc_args.append((parameter, output, covariables))
    with concurrent.futures.ProcessPoolExecutor() as executor:  # set up paralisation for PRCC simulations
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
    parameters_csv_file = 'C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/Mass Gathering work/Compartment based models/Cluster_Vaccination_Meta_Models/Test Parameter distributions.csv'
    parameters_df = pd.read_csv(parameters_csv_file)
    # %%
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
    sample_size = 100
    LHS_and_PRCC_parallel(parameters_df=parameters_df,
                          sample_size=sample_size,
                          model_run_method=sport_match_sim.run_simulation,
                          results_csv='PRCCs LH sample size '+str(sample_size)+'.csv')