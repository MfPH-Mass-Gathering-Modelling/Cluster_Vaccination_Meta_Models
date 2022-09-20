"""
Creation:
    Author: Marting Grunnill
    Date: 2022-09-20
Description: Code for setting up population for running mass gathering models
    
"""
import math

def gen_host_sub_popultion(total_pop, full_dose_pop, booster_pop, host_tickets, infections_to_seed):
    """

    :param total_pop:
    :param full_dose_pop:
    :param booster_pop:
    :param host_tickets:
    :param infections_to_seed:
    :return:
    """
    host_sub_population = {'hosts': {'unvaccinated': total_pop - full_dose_pop,
                                     'effective': booster_pop,
                                     'waned': full_dose_pop - booster_pop},
                           'host_spectators':{}}

    for sub_pop in ['unvaccinated','effective','waned']:
        spectator_sub_pop = round((host_sub_population['hosts'][sub_pop]/total_pop) * host_tickets)
        host_sub_population['hosts'][sub_pop] -= spectator_sub_pop
        host_sub_population['host_spectators'][sub_pop] = spectator_sub_pop

    for cluster, vac_group_dict in host_sub_population.items():
        for vac_group, population in vac_group_dict.items():
            infections = round(infections_to_seed*(population/total_pop))
            host_sub_population[cluster][vac_group] ={'infections':infections,'S': population-infections}

    return host_sub_population

def gen_visitor_sub_population(supportes_pop,
                               supportes_vaccine_prop,
                               detections_per_something,
                               something_denominator,
                               infections_per_detection):
    """

    :param supportes_pop:
    :param supportes_vaccine_prop:
    :param detections_per_something:
    :param something_denominator:
    :param infections_per_detection:
    :return:
    """
    infections_per_something =  infections_per_detection*detections_per_something
    proportions_total = sum(supportes_vaccine_prop.values())
    if not math.isclose(1, proportions_total, abs_tol=0.000001):
        raise ValueError('The sum of dictionary values in supportes_vaccine_prop should equal 1, it is equal to ' +
                         str(proportions_total) + '.')

    sub_populations = {}
    for vaccine_group, proportion in supportes_vaccine_prop.items():
        sub_population_total = round(supportes_pop*proportion)
        sub_pop_infections = round(sub_population_total*(infections_per_something/something_denominator))
        sub_populations[vaccine_group] = {'S':sub_population_total-sub_pop_infections,
                                          'infections': sub_pop_infections}

    return sub_populations