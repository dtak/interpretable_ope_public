import rl_basics_local
import numpy as np
import matplotlib.pyplot as plt

from matrix_based_kernel_fqe import *

def plot_conceptual_demonstration_results(
        trans_influence_array,
        transitions_dataset,
        env,
        true_trajectory,
        influence_threshold):
    idx_fig = 1

    plt.figure()
    color_list = [[0.7, 0.7, 0.7] for _
                  in range(transitions_dataset.return_num_trans())]
    env.plot_transitions_dataset(transitions_dataset, color_list)
    env.plot_trajectory(true_trajectory,
                        plot_color='k',
                        line_width=5)
    for idx_trans in range(transitions_dataset.return_num_trans()):
        if trans_influence_array[idx_trans] > influence_threshold:
            env.plot_transition(transitions_dataset.transitions[idx_trans],
                                'r')
    idx_fig += 1

if __name__ == '__main__':

    gamma = 0.95
    num_true_traj = 100
    num_traj = 5
    neighborhood_radius = 1
    actions_weight_in_metric = 100
    influence_threshold = 0.05
    policy_eval = rl_basics_local.domains.cancer.PolicyCancer()



    ''' Parameters for different scenarios - Results depend on the exact transitions
        sampled, so you don't always get the case you're expecting'''

    ''' No influentiai transitions  '''
    # env = rl_basics.domains.cancer.EnvCancer(transition_noise=0.0)
    # policy_behavior = rl_basics.domains.cancer.PolicyCancer(eps_behavior=0.3)
    # num_traj = 20

    ''' "Dead end"  '''
    env = rl_basics_local.domains.cancer.EnvCancer(transition_noise=0.0)
    policy_behavior = rl_basics_local.domains.cancer.PolicyCancer(eps_behavior=0.3)
    num_traj = 5

    ''' Reasonable influential transitions '''
    # env = rl_basics.domains.cancer.EnvCancer(transition_noise=0.05)
    # policy_behavior = rl_basics.domains.cancer.PolicyCancer(eps_behavior=0.3)
    # num_traj = 20

    ''' Bad influential transitions '''
    # env = rl_basics.domains.cancer.EnvCancer(transition_noise=0.2)
    # policy_behavior = rl_basics.domains.cancer.PolicyCancer(eps_behavior=0.3)
    # num_traj = 20




    time_horizon = env.max_steps

    ''' Estimate true value of evaluation policy using MCMC sampling '''
    true_transitions_dataset = rl_basics_local.classes.TransitionsDataSet()
    true_returns_list = []
    for traj_idx in range(num_true_traj):
        trajectory = rl_basics_local.generate_trajectory(env, policy_eval)
        true_transitions_dataset.add_trajectory(trajectory)
        true_returns_list.append(trajectory.return_trajectory_return(gamma))
    true_eval_policy_value = np.mean(true_returns_list)

    ''' Generate sample true trajectory with no noise'''
    clean_env = rl_basics_local.domains.cancer.EnvCancer(transition_noise=0.00)
    true_trajectory = rl_basics_local.generate_trajectory(clean_env, policy_eval)

    ''' Generate data using behavior policy '''
    transitions_dataset = rl_basics_local.classes.TransitionsDataSet()
    returns_list = []
    for traj_idx in range(num_traj):
        trajectory = rl_basics_local.generate_trajectory(env, policy_behavior)
        transitions_dataset.add_trajectory(trajectory)
        returns_list.append(trajectory.return_trajectory_return(gamma))
    true_behavior_policy_value = np.mean(returns_list)

    ''' Preform matrix-kernel OPE '''
    M, M_prime, reward_vector = compute_nearest_neighbors_matrices(
        transitions_dataset,
        policy_eval,
        neighborhood_radius,
        actions_weight_in_metric)
    q_vector, q_prime_vector, phi, phi_prime = matrix_based_kernel_fqe(
        M, M_prime, reward_vector, time_horizon, gamma)

    num_trans = transitions_dataset.return_num_trans()
    trans_influence_array = np.zeros(num_trans)
    idx_influenced_trans = 0
    M_prime_non_zero = compute_non_zero_elements_in_neighbors_matrices(M_prime)
    for idx_to_remove in range(num_trans):
        print(idx_to_remove)
        trans_influence = effect_of_removing_a_transition(
            M,
            M_prime_non_zero,
            phi,
            reward_vector,
            q_vector,
            q_prime_vector,
            idx_to_remove,
            idx_influenced_trans,
            gamma)
        trans_influence_array[idx_to_remove] = trans_influence

    trans_influence_array = (
            np.abs(trans_influence_array)
            / np.abs(q_vector[0]))

    plot_conceptual_demonstration_results(
        trans_influence_array,
        transitions_dataset,
        env,
        true_trajectory,
        influence_threshold)

    print('Value estimation error: {0:.3f}'.format(
        (q_vector[0] - true_eval_policy_value)/true_eval_policy_value))

    plt.show()