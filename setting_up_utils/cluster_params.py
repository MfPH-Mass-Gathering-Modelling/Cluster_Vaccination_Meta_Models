"""
Creation:
    Author: Martin Grunnill
    Date: 2022-09-26
Description: Functions for handling cluster specific params.
    
"""
def list_to_and_from_cluster_param(param, to_clusters, from_clusters):
    return [param + '_' + cluster_i + '_' + cluster_j for cluster_i in to_clusters for cluster_j in from_clusters]

def list_cluster_param(param, clusters):
    return [param + '_' + cluster for cluster in clusters]

def update_params_with_to_from_cluster_param(params_dict, to_clusters, from_clusters, param, value, return_list=False):
    pararms_to_add = list_to_and_from_cluster_param(param, to_clusters, from_clusters)
    update_dict = {term: value
                   for term in pararms_to_add}
    params_dict.update(update_dict)
    if return_list:
        return pararms_to_add

def update_params_with_cluster_param(params_dict, clusters, param, value, return_list=False):
    pararms_to_add = list_cluster_param(param, clusters)
    update_dict = {term: value
                   for term in pararms_to_add}
    params_dict.update(update_dict)
    if return_list:
        return list(update_dict.keys())