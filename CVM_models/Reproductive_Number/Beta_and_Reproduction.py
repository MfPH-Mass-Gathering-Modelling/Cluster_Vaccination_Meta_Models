"""
Creation:
    Author: Martin Grunnill
    Date: 14/09/2022
Description:Functions for calculating Reproductive numbers or Beta for Cluster Vaccination models.
    
"""


def MGE_R0_no_vaccine_1_cluster(gamma_I_2, beta, p_d, gamma_I_1, epsilon_3,
                                p_s, kappa_D, gamma_A_2, p_h, theta, gamma_A_1):
    """ Calculates R0 for Mass gathering event assuming 1 cluster
        (or homogenous mixing between clusters) and no vaccination.
        See Dervinng_R0_and_beta_for_1_cluster_no_vaccine.py for dervation.
    """
    numrator = beta*(
            epsilon_3*gamma_A_1*gamma_A_2*gamma_I_1*p_s*(-kappa_D*p_d*p_h + kappa_D*p_d - p_d + 1) +
            epsilon_3*gamma_A_1*gamma_A_2*gamma_I_2*p_s*(-kappa_D*p_d*p_h + kappa_D*p_d - p_d + 1) +
            epsilon_3*gamma_A_1*gamma_I_1*gamma_I_2*theta*(1 - p_s) +
            epsilon_3*gamma_A_2*gamma_I_1*gamma_I_2*theta*(1 - p_s) +
            gamma_A_1*gamma_A_2*gamma_I_1*gamma_I_2*theta)
    denominator = epsilon_3*gamma_A_1*gamma_A_2*gamma_I_1*gamma_I_2
    return numrator/denominator

def MGE_beta_no_vaccine_1_cluster(gamma_I_2, R_0, p_d, gamma_I_1, epsilon_3,
                                  p_s, kappa_D, gamma_A_2, p_h, theta, gamma_A_1):
    """ Calculates beta given R0 for Mass gathering event assuming 1 cluster
        (or homogenous mixing between clusters) and no vaccination.
        See Dervinng_R0_and_beta_for_1_cluster_no_vaccine.py for dervation.
    """
    numrator = R_0*epsilon_3*gamma_A_1*gamma_A_2*gamma_I_1*gamma_I_2
    denominator = (-epsilon_3*gamma_A_1*gamma_A_2*gamma_I_1*kappa_D*p_d*p_h*p_s +
                   epsilon_3*gamma_A_1*gamma_A_2*gamma_I_1*kappa_D*p_d*p_s -
                   epsilon_3*gamma_A_1*gamma_A_2*gamma_I_1*p_d*p_s +
                   epsilon_3*gamma_A_1*gamma_A_2*gamma_I_1*p_s -
                   epsilon_3*gamma_A_1*gamma_A_2*gamma_I_2*kappa_D*p_d*p_h*p_s +
                   epsilon_3*gamma_A_1*gamma_A_2*gamma_I_2*kappa_D*p_d*p_s -
                   epsilon_3*gamma_A_1*gamma_A_2*gamma_I_2*p_d*p_s +
                   epsilon_3*gamma_A_1*gamma_A_2*gamma_I_2*p_s -
                   epsilon_3*gamma_A_1*gamma_I_1*gamma_I_2*p_s*theta +
                   epsilon_3*gamma_A_1*gamma_I_1*gamma_I_2*theta -
                   epsilon_3*gamma_A_2*gamma_I_1*gamma_I_2*p_s*theta +
                   epsilon_3*gamma_A_2*gamma_I_1*gamma_I_2*theta +
                   gamma_A_1*gamma_A_2*gamma_I_1*gamma_I_2*theta)
    return numrator/denominator