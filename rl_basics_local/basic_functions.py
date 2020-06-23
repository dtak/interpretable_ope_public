import numpy as np

from rl_basics_local.classes import Transition, Trajectory

def generate_trajectory(env, policy, initial_state=None, initial_action=None):
    env.reset()
    if initial_state:
        env.set_state(initial_state)
    trajectory = Trajectory()
    state = env.observe()
    time_step = 0
    if initial_action:
        next_state, reward = env.perform_action(initial_action)
        new_transition = Transition(
            state, initial_action, reward, next_state, time_step=time_step)
        if hasattr(env, 'latent_state'):
            new_transition.set_latent_state(env.latent_state)
        trajectory.add_transition(new_transition)
        state = next_state
        time_step += 1
    while not env.is_done():
        action = policy(state, time_step)
        next_state, reward = env.perform_action(action)
        new_transition = Transition(
            state, action, reward, next_state, time_step=time_step)
        if hasattr(env, 'latent_state'):
            new_transition.set_latent_state(env.latent_state)
        trajectory.add_transition(new_transition)
        state = next_state
        time_step += 1
    return trajectory

def generate_transition(env, state=None, action=None, policy=None):
    if (action is not None) and (policy is not None):
        raise Exception("action and policy cannot both be specified.")
    env.reset(state)
    state = env.observe()
    if action is not None:
        pass
    elif policy is not None:
        action = policy(state)
    else:
        action = env.sample_action()
    next_state, reward = env.perform_action(action)
    transition = Transition(state, action, reward, next_state)
    if hasattr(env, 'latent_state'):
        transition.set_latent_state(env.latent_state)
    return transition

def mcmc_value_estimation(env, policy, gamma, num_traj = 10000):
    returns_arr = np.zeros(num_traj)
    for idx_traj in range(num_traj):
        trajectory = generate_trajectory(env, policy)
        returns_arr[idx_traj] = trajectory.return_trajectory_return(gamma)
    return np.mean(returns_arr)
