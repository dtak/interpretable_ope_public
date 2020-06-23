import numpy as np
import matplotlib.pyplot as plt
from time import time
from copy import deepcopy
from scipy import sparse

from scipy.spatial.distance import pdist, squareform

from temp_utils import SparseMatrix


def compute_nearest_neighbors_matrices(
        transitions_dataset,
        policy,
        neighborhood_radius,
        action_weight=1):
    ''' transitions - rl_basics.classes.TransitionsDataSet '''

    transitions = transitions_dataset.transitions
    num_trans = transitions_dataset.return_num_trans()
    states_dimensionality = len(transitions[0].state)
    actions_dimensionality = len(transitions[0].action)
    states_array = np.zeros([2 * num_trans, states_dimensionality])
    actions_array = np.zeros([2 * num_trans, actions_dimensionality])
    reward_vector = np.zeros([num_trans, 1])
    for trans_idx in range(num_trans):
        trans = transitions[trans_idx]
        states_array[trans_idx, :] = trans.state
        actions_array[trans_idx, :] = trans.action
        states_array[num_trans + trans_idx, :] = trans.next_state
        actions_array[num_trans + trans_idx, :] = (
            policy(trans.next_state, trans.time_step + 1))
        reward_vector[trans_idx] = trans.reward
    entire_dist_matrix = (
        (squareform(pdist(states_array))**2
         + action_weight * squareform(pdist(actions_array))**2)**0.5)
    M = entire_dist_matrix[:num_trans, :num_trans]
    M = (M <= neighborhood_radius).astype(float)
    M /= np.sum(M, axis=1).reshape(num_trans, 1)

    M_prime = entire_dist_matrix[num_trans:, :num_trans]
    M_prime = (M_prime <= neighborhood_radius).astype(float)
    M_prime_norm = np.sum(M_prime, axis=1).reshape(num_trans, 1)
    M_prime_norm[M_prime_norm == 0] = 1
    M_prime /= M_prime_norm

    return M, M_prime, reward_vector


def compute_non_zero_elements_in_neighbors_matrices(matrix):
    non_zero_elements = []
    for idx_col in range(matrix.shape[1]):
        non_zero_elements.append(
            {idx_non_zero: matrix[idx_non_zero, idx_col]
             for idx_non_zero in np.where(matrix[:, idx_col] != 0)[0]})
    return non_zero_elements


def compute_non_zero_elements_in_neighbors_matrices_by_rows(matrix):
    non_zero_elements_by_rows = []
    for idx_row in range(matrix.shape[0]):
        non_zero_elements_by_rows.append(
            {idx_non_zero: matrix[idx_row, idx_non_zero]
             for idx_non_zero in np.where(matrix[idx_row, :] != 0)[0]})
    return non_zero_elements_by_rows


def matrix_based_kernel_fqe(
        M, M_prime, reward_vector, time_horizon, gamma,
        return_multiple_values=True):
    phi_prime = np.zeros_like(M)
    M_prime = sparse.csr_matrix(M_prime)
    M_prime_to_the_power_of_T = np.eye(M_prime.shape[0])
    M_prime_to_the_power_of_T = sparse.csr_matrix(M_prime_to_the_power_of_T)
    for t_idx in range(1, time_horizon + 1):
        M_prime_to_the_power_of_T = np.dot(M_prime, M_prime_to_the_power_of_T)
        phi_prime += gamma ** (t_idx - 1) * M_prime_to_the_power_of_T
    q_prime_vector = np.dot(phi_prime, reward_vector)
    q_prime_vector = q_prime_vector.squeeze()

    phi = deepcopy(phi_prime)
    phi -= gamma ** (time_horizon - 1) * M_prime_to_the_power_of_T
    phi *= gamma
    phi += np.eye(M_prime.shape[0])
    phi = np.dot(M, phi)
    q_vector = np.dot(phi, reward_vector)
    q_vector = q_vector.squeeze()

    q_vector = np.array(q_vector)[0]
    q_prime_vector = np.array(q_prime_vector)[0]
    phi = np.array(phi)
    phi_prime = np.array(phi_prime)

    if return_multiple_values:
        return q_vector, q_prime_vector, phi, phi_prime
    else:
        return q_vector


def remove_transitions(transitions_dataset, states_range, transitions_to_leave):
    idx_trans = 0
    transitions_left = 0
    indices_of_transitions_left = []
    done = False
    while not done:
        trans = transitions_dataset.transitions[idx_trans]
        if states_range[0] < trans.state[0] < states_range[1]:
            if transitions_left < transitions_to_leave:
                transitions_left += 1
                indices_of_transitions_left.append(idx_trans)
                idx_trans += 1
            else:
                del transitions_dataset.transitions[idx_trans]
        else:
            idx_trans += 1
        done = idx_trans == len(transitions_dataset.transitions)
    return transitions_dataset, indices_of_transitions_left

def find_transitions_in_dataset(transitions_dataset, states_range):
    indices_of_found_transitions = []
    state_dimensionality = transitions_dataset.return_state_dimensionality()
    for idx_trans in range(transitions_dataset.return_num_trans()):
        trans = transitions_dataset.transitions[idx_trans]
        trans_is_in_range = True
        for dim in range(state_dimensionality):
            trans_is_in_range = (
                trans_is_in_range
                and (states_range[dim][0]
                     < trans.state[dim]
                     < states_range[dim][1]))
        if trans_is_in_range:
            indices_of_found_transitions.append(idx_trans)
    return indices_of_found_transitions


def effect_of_removing_a_transition__old(
        M,
        M_prime,
        reward_vector,
        idx_to_remove,
        time_horizon,
        gamma,
        return_multiple_values=False):
    ''' This is the old version of computing the influence of removing a
    transition. It essentially recomputes the OPE after removal of a transition,
    but performs the removal by computing the direct effect on M and M_prime
    rather than recomputing these distance matrices from scratch.'''
    delta_M = np.copy(M)
    array_of_neighbors = M[idx_to_remove, :] > 0
    for idx in range(M.shape[0]):
        if array_of_neighbors[idx] and idx != idx_to_remove:
            delta_M[idx, :] = (
                delta_M[idx, :] / (delta_M[idx, :]**(-1) - 1))
        else:
            delta_M[idx, :] = 0
    # TODO: Deal better with transition having only one neighbor

    delta_M_prime = np.copy(M_prime)
    array_of_neighbors = M_prime[:, idx_to_remove] > 0
    for idx in range(M_prime.shape[0]):
        if array_of_neighbors[idx] and idx != idx_to_remove:
            delta_M_prime[idx, :] = (
                delta_M_prime[idx, :] / (delta_M_prime[idx, :]**(-1) - 1))
        else:
            delta_M_prime[idx, :] = 0
    # TODO: Deal better with transition having only one neighbor
    delta_M_prime[np.isinf(delta_M_prime)] = 0

    shrunk_reward_vector = np.delete(reward_vector, idx_to_remove)
    shrunk_M = np.delete(M, idx_to_remove, axis=0)
    shrunk_delta_M = np.delete(delta_M, idx_to_remove, axis=0)
    shrunk_M_prime = np.delete(M_prime, idx_to_remove, axis=0)
    shrunk_delta_M_prime = np.delete(delta_M_prime, idx_to_remove, axis=0)

    shrunk_M = np.delete(shrunk_M, idx_to_remove, axis=1)
    shrunk_delta_M = np.delete(shrunk_delta_M, idx_to_remove, axis=1)
    shrunk_M_prime = np.delete(shrunk_M_prime, idx_to_remove, axis=1)
    shrunk_delta_M_prime = np.delete(
        shrunk_delta_M_prime, idx_to_remove, axis=1)

    phi = np.zeros_like(M)
    M_prime_to_the_power_of_T = np.eye(M_prime.shape[0])
    for t_idx in range(1, time_horizon + 1):
        phi += gamma ** (t_idx - 1) * M_prime_to_the_power_of_T
        M_prime_to_the_power_of_T = np.dot(M_prime, M_prime_to_the_power_of_T)
    phi = np.dot(M, phi)

    shrunk_phi = np.delete(phi, idx_to_remove, axis=0)
    shrunk_phi = np.delete(shrunk_phi, idx_to_remove, axis=1)

    new_phi = np.zeros_like(shrunk_phi)
    M_prime_to_the_power_of_T = np.eye(shrunk_M_prime.shape[0])
    for t_idx in range(1, time_horizon + 1):
        new_phi += gamma**(t_idx - 1) * M_prime_to_the_power_of_T
        M_prime_to_the_power_of_T = np.dot(
            (shrunk_M_prime + shrunk_delta_M_prime), M_prime_to_the_power_of_T)
    new_phi = np.dot((shrunk_M + shrunk_delta_M), new_phi)

    delta_q = (
        np.dot(new_phi - shrunk_phi, shrunk_reward_vector)
        - reward_vector[idx_to_remove]
        * np.delete(phi[:, idx_to_remove], idx_to_remove))

    if return_multiple_values:
        return delta_q, phi
    else:
        return delta_q


def effect_of_removing_a_transition__brute_force(
        transitions_dataset,
        idx_to_remove,
        policy_eval,
        neighborhood_radius,
        actions_weight_in_metric,
        time_horizon,
        gamma,
        original_q_vector):
    '''  Computes the influence of removing a transition by recomputing from
    scratch the OPE after removal of a transition from the original data.'''
    transitions_dataset_after_removal = deepcopy(transitions_dataset)
    transitions_dataset_after_removal.remove_transition(idx_to_remove)

    M_after_removal, M_prime_after_removal, reward_vector_after_removal = (
        compute_nearest_neighbors_matrices(
            transitions_dataset_after_removal,
            policy_eval,
            neighborhood_radius,
            actions_weight_in_metric))
    q_vector_after_removal = matrix_based_kernel_fqe(
        M_after_removal,
        M_prime_after_removal,
        reward_vector_after_removal,
        time_horizon,
        gamma)

    shrunk_q_vector = np.delete(original_q_vector, idx_to_remove)
    delta_q = - (shrunk_q_vector - q_vector_after_removal)

    return delta_q

def effect_of_removing_a_transition(
        M,
        M_prime_non_zero,
        phi,
        reward_vector,
        q_vector,
        q_prime_vector,
        idx_to_remove,
        idx_influenced_trans,
        gamma):
    influence = 0
    for idx_neighbors in M_prime_non_zero[idx_to_remove].keys():
        num_neighbors = 1 / M_prime_non_zero[idx_to_remove][idx_neighbors]
        if idx_neighbors != idx_to_remove:
            if num_neighbors > 1:
                influence += (
                    phi[idx_influenced_trans, idx_neighbors]
                    * (num_neighbors - 1)**(-1)
                    * gamma
                    * (q_prime_vector[idx_neighbors]
                       - (reward_vector[idx_to_remove]
                          + gamma * q_prime_vector[idx_to_remove])))
            elif num_neighbors == 1:
                influence -= (
                    phi[idx_influenced_trans, idx_neighbors]
                    * gamma * q_vector[idx_to_remove])
    if M[idx_influenced_trans, idx_to_remove] != 0:
        num_neighbors = 1 / M[idx_influenced_trans, idx_to_remove]
        influence += (
            (num_neighbors - 1) ** (-1)
            * (q_vector[idx_influenced_trans]
               - (reward_vector[idx_to_remove][0]
                  + gamma * q_prime_vector[idx_to_remove])))

    return influence


def diagnose_evaluation(
        M,
        M_prime,
        phi,
        reward_vector,
        q_vector,
        q_prime_vector,
        gamma,
        indices_of_initial_transitions,
        influence_threshold,
        max_neighbors=np.inf):

    M_prime_non_zero = compute_non_zero_elements_in_neighbors_matrices(M_prime)
    M_prime_non_zero_by_rows = (
        compute_non_zero_elements_in_neighbors_matrices_by_rows(M_prime))
    num_transitions = len(q_vector)
    num_init_trans = len(indices_of_initial_transitions)
    total_influence_of_each_transition = np.zeros(num_transitions)
    policy_value = np.mean(q_vector[indices_of_initial_transitions])
    for idx_influencer in range(num_transitions):
        if len(M_prime_non_zero[idx_influencer]) <= max_neighbors:
            for idx_init_trans in range(num_init_trans):
                if (idx_influencer
                        != indices_of_initial_transitions[idx_init_trans]):
                    temp_influence = effect_of_removing_a_transition(
                        M,
                        M_prime_non_zero,
                        phi,
                        reward_vector,
                        q_vector,
                        q_prime_vector,
                        idx_influencer,
                        indices_of_initial_transitions[idx_init_trans],
                        gamma)
                    total_influence_of_each_transition[idx_influencer] += (
                        temp_influence / num_init_trans)
        else:
            # Technically the next assignment shouldn't be zero but something
            # we can bound, but we just assume the bound will be lower than
            # influence_threshold. If that is not the case, it means we chose
            # too small of a value for max_neighbors
            total_influence_of_each_transition[idx_influencer] = 0
    total_influence_of_each_transition = (
        np.abs(total_influence_of_each_transition / policy_value))
    print('Number of transitions with influence over {}: {}'.format(
        influence_threshold,
        np.sum(total_influence_of_each_transition > influence_threshold)))
    transitions_with_high_total_influence = {
        idx_trans: {'total_influence':
                        total_influence_of_each_transition[idx_trans],
                    'sequence': []}
        for idx_trans in list(np.where(
            total_influence_of_each_transition > influence_threshold)[0])}

    for key in transitions_with_high_total_influence.keys():
        if len(M_prime_non_zero_by_rows[key]) == 1:
            next_trans = list(M_prime_non_zero_by_rows[key].keys())[0]
            if next_trans in list(transitions_with_high_total_influence.keys()):
                transitions_with_high_total_influence[key]['sequence'].append(
                    next_trans)

    high_influence_trans_idxs = list(transitions_with_high_total_influence.keys())

    for key in transitions_with_high_total_influence.keys():
        print(key, transitions_with_high_total_influence[key])
    print()

    # Removing sequences
    trans_to_delete = []
    for key in transitions_with_high_total_influence.keys():
        if transitions_with_high_total_influence[key]['sequence'] != []:
            next_in_sequence = transitions_with_high_total_influence[key]['sequence'][0]
            if next_in_sequence not in trans_to_delete:
                trans_to_delete.append(next_in_sequence)
            found_loop = False
            while transitions_with_high_total_influence[next_in_sequence]['sequence'] != [] and not found_loop:
                next_next_in_sequence = transitions_with_high_total_influence[next_in_sequence]['sequence'][0]
                if next_next_in_sequence not in transitions_with_high_total_influence[key]['sequence']:
                    transitions_with_high_total_influence[key]['sequence'].append(
                        next_next_in_sequence)
                    next_in_sequence = next_next_in_sequence
                else:
                    found_loop = True

    for key_to_delete in trans_to_delete:
        del transitions_with_high_total_influence[key_to_delete]

    for key in transitions_with_high_total_influence.keys():
        print(key, transitions_with_high_total_influence[key])
    print()
    print(trans_to_delete)

    return high_influence_trans_idxs