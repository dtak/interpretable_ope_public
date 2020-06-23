import rl_basics_local
import rl_basics_local.domains.continuous_navigation

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.font_manager import FontProperties

from matrix_based_kernel_fqe import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--plot_trajectory", type=int, default=1)
parser.add_argument("--save_figures", type=int, default=0)
parser_args = parser.parse_args()

''' Define experiment parameters '''
time_horizon = 17
gamma = 0.9
num_traj = 10

reward_func = rl_basics_local.domains.continuous_navigation.RewardFuncGaussians(
    [[np.array([7.5, 7.5]), 10, 1.25]])
env = rl_basics_local.domains.continuous_navigation.EnvContinuousNavigation(
    dimensionality=2,
    reward_func=reward_func,
    step_size=1,
    step_noise=0.0075,
    max_steps=time_horizon)

policy_eval = (
    rl_basics_local.domains.continuous_navigation.PolicyContinuousNavigation(
        dimensionality=2,
        direction_vector=np.array([1, 1])))

policy_behavior = (
    rl_basics_local.domains.continuous_navigation.PolicyContinuousNavigation(
        dimensionality=2,
        direction_vector=[
            np.array([1, 1]),
            np.array([1, 0.3]),
        ],
        direction_vector_probability=np.array([1.0, 0.0]),
        randomness_magnitude=0.0))

''' Estimate true value of evaluation policy using MCMC sampling '''
true_transitions_dataset = rl_basics_local.classes.TransitionsDataSet()
true_returns_list = []
for traj_idx in range(num_traj):
    trajectory = rl_basics_local.generate_trajectory(env, policy_eval)
    true_transitions_dataset.add_trajectory(trajectory)
    true_returns_list.append(trajectory.return_trajectory_return(gamma))
true_eval_policy_value = np.mean(true_returns_list)

num_exp = 50
influence_of_important_trans_list = []
influence_of_unimportant_trans_list = []
influence_of_second_unimportant_trans_list = []
for i in range(num_exp):
    print(i)
    transitions_dataset = rl_basics_local.classes.TransitionsDataSet()
    returns_list = []
    for traj_idx in range(num_traj):
        trajectory = rl_basics_local.generate_trajectory(env, policy_behavior)
        transitions_dataset.add_trajectory(trajectory)
        returns_list.append(trajectory.return_trajectory_return(gamma))
    true_behavior_policy_value = np.mean(returns_list)
    num_trans_in_dataset = transitions_dataset.return_num_trans()

    transitions_dataset, indices_of_transitions_left = remove_transitions(
        transitions_dataset,
        [4, 5],
        3)

    transitions_dataset, indices_of_second_transitions_left = remove_transitions(
        transitions_dataset,
        [9, 10],
        3)

    M, M_prime, reward_vector = compute_nearest_neighbors_matrices(
        transitions_dataset, policy_eval, 0.2, 1)
    q_vector, q_prime_vector, phi, phi_prime = matrix_based_kernel_fqe(
        M, M_prime, reward_vector, time_horizon, gamma)

    M_prime_non_zero = compute_non_zero_elements_in_neighbors_matrices(M_prime)

    idx_of_important_trans_inspected = 6
    idx_influenced_trans = 0
    influence_of_important_transition = effect_of_removing_a_transition(
        M,
        M_prime_non_zero,
        phi,
        reward_vector,
        q_vector,
        q_prime_vector,
        idx_of_important_trans_inspected,
        idx_influenced_trans,
        gamma)

    idx_of_unimportant_trans_inspected = 2
    idx_influenced_trans = 0
    influence_of_unimportant_transition = effect_of_removing_a_transition(
        M,
        M_prime_non_zero,
        phi,
        reward_vector,
        q_vector,
        q_prime_vector,
        idx_of_unimportant_trans_inspected,
        idx_influenced_trans,
        gamma)



    idx_of_second_unimportant_trans_inspected = 14
    idx_influenced_trans = 0
    influence_of_second_unimportant_transition = (
        effect_of_removing_a_transition(
            M,
            M_prime_non_zero,
            phi,
            reward_vector,
            q_vector,
            q_prime_vector,
            idx_of_second_unimportant_trans_inspected,
            idx_influenced_trans,
            gamma))

    influence_of_important_trans_list.append(
        abs(influence_of_important_transition)/true_eval_policy_value)
    influence_of_unimportant_trans_list.append(
        abs(influence_of_unimportant_transition)/true_eval_policy_value)
    influence_of_second_unimportant_trans_list.append(
        abs(influence_of_second_unimportant_transition)/true_eval_policy_value)

    if parser_args.plot_trajectory:
        if i == num_exp - 1:
            plt.figure()
            env.plot_transitions_dataset(transitions_dataset)
            trans_inspected = (
                transitions_dataset.transitions[
                    idx_of_important_trans_inspected])
            plt.plot([trans_inspected.state[0], trans_inspected.next_state[0]],
                     [trans_inspected.state[1], trans_inspected.next_state[1]],
                     'ro-')
            trans_inspected = (
                transitions_dataset.transitions[
                    idx_of_unimportant_trans_inspected])
            plt.plot([trans_inspected.state[0], trans_inspected.next_state[0]],
                     [trans_inspected.state[1], trans_inspected.next_state[1]],
                     'ro-')
            trans_inspected = (
                transitions_dataset.transitions[
                    idx_of_second_unimportant_trans_inspected])
            plt.plot([trans_inspected.state[0], trans_inspected.next_state[0]],
                     [trans_inspected.state[1], trans_inspected.next_state[1]],
                     'ro-')
            plt.xticks(np.array(range(time_horizon + 1))/2**0.5,
                       range(time_horizon + 1))
            plt.yticks(np.array(range(time_horizon + 1))/2**0.5,
                       range(time_horizon + 1))
            font = FontProperties()
            font.set_weight('semibold')
            font.set_family('serif')
            font.set_size('xx-large')
            plt.text(2/2**0.5, 3.5/2**0.5, 'I', fontproperties=font)
            plt.text(6/2**0.5, 8/2**0.5, 'II', fontproperties=font)
            plt.text(12.5/2**0.5, 15.5/2**0.5, 'III', fontproperties=font)
            # plt.savefig('intuition_for_influence.pdf',
            #             bbox_inches='tight')
            break

plt.figure()
plt.boxplot([
    influence_of_unimportant_trans_list,
    influence_of_important_trans_list,
    influence_of_second_unimportant_trans_list])
plt.xticks([1, 2, 3], [
    'I', 'II', 'III'], fontproperties=font)
plt.ylabel(r'$\tilde{I}_j$', fontsize=20)
# plt.savefig('intuition_for_influence_boxplots.pdf',
#             bbox_inches='tight')

plt.show()