import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

import rl_basics_local
from rl_basics_local.ope.ope import *

import sys
sys.path.append("../..")
from importance_sampling_influence import *


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--plot_trajectory", type=int, default=1)
parser.add_argument("--save_figures", type=int, default=0)
parser.add_argument("--save_results", type=int, default=1)
parser.add_argument('--results_fname', default='results.pkl')
parser_args = parser.parse_args()

def plot_demonstration_results(
        traj_influence_array,
        traj_dataset,
        env,
        true_trajectory,
        influence_threshold,
        save_figures=True,
        figure_fname=None):
    idx_fig = 1

    plt.figure()
    color_list_scaled = [
        [abs(traj_influence_array[idx_traj]) / np.max(abs(traj_influence_array)),
         0,
         1 - (abs(traj_influence_array[idx_traj])
              / np.max(abs(traj_influence_array)))]
        for idx_traj
        in range(traj_dataset.return_num_traj())]
    color_list = [[0.7, 0.7, 0.7] for _
                  in range(traj_dataset.return_num_traj())]
    for idx_traj, traj in enumerate(traj_dataset.trajectories):
        env.plot_trajectory(traj, plot_color=color_list[idx_traj])
    env.plot_trajectory(true_trajectory, plot_color='k', line_width=5)
    for idx_traj in range(traj_dataset.return_num_traj()):
        # if abs(traj_influence_array[idx_traj]) > influence_threshold:
        if abs(traj_influence_array[idx_traj]) > np.sort(abs(traj_influence_array))[-6]:
            env.plot_trajectory(traj_dataset.trajectories[idx_traj],
                                # color_list_scaled[idx_traj])
                                'r')

    plt.subplot(1,2,1)
    ratio = 0.4
    xvals,yvals = plt.gca().axes.get_xlim(),plt.gca().axes.get_ylim()
    xrange = xvals[1]-xvals[0]
    yrange = yvals[1]-yvals[0]
    plt.gca().set_aspect(ratio*(xrange/yrange), adjustable='box')
    plt.subplot(1,2,2)
    xvals,yvals = plt.gca().axes.get_xlim(),plt.gca().axes.get_ylim()
    xrange = xvals[1]-xvals[0]
    yrange = yvals[1]-yvals[0]
    plt.gca().set_aspect(ratio*(xrange/yrange), adjustable='box')

gamma = 0.95
num_true_traj = 100
num_traj = 50
influence_threshold = 0.5
env = rl_basics_local.domains.cancer.EnvCancer(transition_noise=0.2)
policy_eval = rl_basics_local.domains.cancer.PolicyCancer()
policy_behavior = rl_basics_local.domains.cancer.PolicyCancer(eps_behavior=0.15)
time_horizon = env.max_steps

''' Estimate true value of evaluation policy using MCMC sampling '''
true_eval_policy_value = rl_basics_local.mcmc_value_estimation(
    env, policy_eval, gamma, num_traj=1000)

''' Generate sample true trajectory with no noise'''
clean_env = rl_basics_local.domains.cancer.EnvCancer(transition_noise=0.00)
true_trajectory = rl_basics_local.generate_trajectory(clean_env, policy_eval)

''' Generate data using behavior policy '''
traj_dataset = rl_basics_local.classes.TrajectoriesDataSet()
returns_list = []
for traj_idx in range(num_traj):
    trajectory = rl_basics_local.generate_trajectory(env, policy_behavior)
    traj_dataset.add_trajectory(trajectory)
    returns_list.append(trajectory.return_trajectory_return(gamma))
true_behavior_policy_value = np.mean(returns_list)

print()
print('Evaluation policy true value : {0:.2f}'.format(true_eval_policy_value))
print('Behavior policy true value : {0:.2f}'.format(true_behavior_policy_value))
print()


ope_func = ope_is
influence_analysis_func = influence_analysis__is
value_estimate, traj_influence_arr = influence_analysis_func(
    traj_dataset, policy_eval, policy_behavior, gamma)
value_estimate_is = value_estimate
traj_influence_arr_is = traj_influence_arr
print('IS estimate : {0:.2f}'.format(value_estimate))
test_influence_analysis(
        traj_dataset,
        policy_eval,
        policy_behavior,
        gamma,
        ope_func,
        influence_analysis_func)
plot_demonstration_results(
        traj_influence_arr,
        traj_dataset,
        env,
        true_trajectory,
        influence_threshold=influence_threshold,
        save_figures=False)

ope_func = ope_wis
influence_analysis_func = influence_analysis__wis
value_estimate, traj_influence_arr = influence_analysis_func(
    traj_dataset, policy_eval, policy_behavior, gamma)
value_estimate_wis = value_estimate
traj_influence_arr_wis = traj_influence_arr
print('WIS estimate : {0:.2f}'.format(value_estimate))
test_influence_analysis(
        traj_dataset,
        policy_eval,
        policy_behavior,
        gamma,
        ope_func,
        influence_analysis_func)
plot_demonstration_results(
        traj_influence_arr,
        traj_dataset,
        env,
        true_trajectory,
        influence_threshold=influence_threshold,
        save_figures=False)

ope_func = ope_pdis
influence_analysis_func = influence_analysis__pdis
value_estimate, traj_influence_arr = influence_analysis_func(
    traj_dataset, policy_eval, policy_behavior, gamma)
value_estimate_pdis = value_estimate
traj_influence_arr_pdis = traj_influence_arr
print('PDIS estimate : {0:.2f}'.format(value_estimate))
test_influence_analysis(
        traj_dataset,
        policy_eval,
        policy_behavior,
        gamma,
        ope_func,
        influence_analysis_func)
plot_demonstration_results(
        traj_influence_arr,
        traj_dataset,
        env,
        true_trajectory,
        influence_threshold=influence_threshold,
        save_figures=False)



influence_hist_y_is, influence_hist_x_is = np.histogram(
    traj_influence_arr_is, 20)
influence_hist_x_is = (influence_hist_x_is[1:] + influence_hist_x_is[:-1])/2

influence_hist_y_wis, influence_hist_x_wis = np.histogram(
    traj_influence_arr_wis, 20)
influence_hist_x_wis = (influence_hist_x_wis[1:] + influence_hist_x_wis[:-1])/2

influence_hist_y_pdis, influence_hist_x_pdis = np.histogram(
    traj_influence_arr_pdis, 20)
influence_hist_x_pdis = (influence_hist_x_pdis[1:] + influence_hist_x_pdis[:-1])/2
traj_influence_arr_is /= true_eval_policy_value
traj_influence_arr_wis /= true_eval_policy_value
traj_influence_arr_pdis /= true_eval_policy_value
plt.figure()
bin_width = 0.01
kwargs = dict(alpha=0.5, density=True, stacked=True)
bin_num = int((max(traj_influence_arr_is) - min(traj_influence_arr_is)) // bin_width)
plt.hist(traj_influence_arr_is, **kwargs, label='IS', bins=bin_num)
bin_num = int((max(traj_influence_arr_wis) - min(traj_influence_arr_wis)) // bin_width)
plt.hist(traj_influence_arr_wis, **kwargs, label='WIS', bins=bin_num)
bin_num = int((max(traj_influence_arr_pdis) - min(traj_influence_arr_pdis)) // bin_width)
plt.hist(traj_influence_arr_pdis, **kwargs, label='PDIS', bins=bin_num)
plt.legend()
plt.title('Influence Distribution')
plt.xlabel('Normalized Influence')


plt.show()