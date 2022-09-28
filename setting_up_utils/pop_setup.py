"""
Creation:
    Author: Marting Grunnill
    Date: 2022-09-20
Description: Code for setting up population for running mass gathering models
    
"""
import math

def gen_host_sub_popultion(host_population,
                           hosts_effectively_vaccinated, booster_pop, host_tickets, infections_to_seed):
    """

    :param host_population:
    :param full_dose_pop:
    :param booster_pop:
    :param host_tickets:
    :param infections_to_seed:
    :return:
    """

def gen_host_sub_popultion(host_population, host_tickets, host_staff,
                           hosts_unvaccinated, hosts_effectively_vaccinated, hosts_waned_vaccinated,
                           host_cases_per_million, host_infections_per_case):
    """

    :param host_population:
    :param host_tickets:
    :param host_staff:
    :param hosts_unvaccinated:
    :param hosts_effectively_vaccinated:
    :param hosts_waned_vaccinated:
    :param host_cases_per_million:
    :param host_infections_per_case:
    :return:
    """
    host_sub_population = {'hosts': {'unvaccinated': hosts_unvaccinated,
                                     'effective': hosts_effectively_vaccinated,
                                     'waned': hosts_waned_vaccinated},
                           'host_spectators': {},
                           'host_staff': {}}

    for sub_pop in ['unvaccinated', 'effective', 'waned']:
        spectator_sub_pop = round((host_sub_population['hosts'][sub_pop] / host_population) * host_tickets)
        staff_sub_pop = round((host_sub_population['hosts'][sub_pop] / host_population) * host_staff)
        host_sub_population['hosts'][sub_pop] -= (spectator_sub_pop+staff_sub_pop)
        host_sub_population['host_spectators'][sub_pop] = spectator_sub_pop
        host_sub_population['host_staff'][sub_pop] = staff_sub_pop

    host_estimate_infections = (host_cases_per_million *
                                (host_population / 1e6) *
                                host_infections_per_case)
    for cluster, vac_group_dict in host_sub_population.items():
        for vac_group, population in vac_group_dict.items():
            sub_pop_infections = round(host_estimate_infections * (population / host_population))
            host_sub_population[cluster][vac_group] = {'infections': sub_pop_infections,
                                                       'S': population - sub_pop_infections}

    return host_sub_population

def gen_visitor_sub_population(supportes_pop,
                               supportes_vaccine_prop,
                               cases_per_million,
                               infections_per_case):
    """

    :param supportes_pop:
    :param supportes_vaccine_prop:
    :param cases_per_million:
    :param infections_per_case:
    :return:
    """
    infection_prevelance = infections_per_case*(cases_per_million/1e6)
    proportions_total = sum(supportes_vaccine_prop.values())
    if not math.isclose(1, proportions_total, abs_tol=0.000001):
        raise ValueError('The sum of dictionary values in supportes_vaccine_prop should equal 1, it is equal to ' +
                         str(proportions_total) + '.')

    sub_populations = {}
    for vaccine_group, proportion in supportes_vaccine_prop.items():
        sub_population_total = round(supportes_pop*proportion)
        sub_pop_infections = round(sub_population_total*infection_prevelance)
        sub_populations[vaccine_group] = {'S':sub_population_total-sub_pop_infections,
                                          'infections': sub_pop_infections}

    return sub_populations