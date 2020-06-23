import numpy as np
import matplotlib.pyplot as plt


def reward_func_flat(state, action, base_value=0, reward_std=1):
    return np.random.normal(base_value, reward_std)


class RewardFuncGaussians:
    ''' gauss_params given as a tuple with three entries :
        [Center, Amplitude, Width] '''
    def __init__(self, gauss_params):
        self.gauss_params = gauss_params

    def __call__(self, state, action):
        reward = 0
        for params in self.gauss_params:
            distance_from_center = np.sum((state - params[0])**2)**0.5
            reward += (params[1] * np.exp(-(distance_from_center/params[2])**2))
        return reward

class EnvContinuousNavigation:
    def __init__(self,
                 reward_func=reward_func_flat,
                 step_size=1,
                 step_noise=0.1,
                 max_steps=30,
                 dimensionality=1):
        self.reward_func = reward_func
        self.step_size = step_size
        self.step_noise = step_noise
        self.max_steps = max_steps
        self.dimensionality = dimensionality
        self.state = None
        self.time_step = None
        self.reward_func = reward_func

    def reset(self):
        self.state = np.zeros(self.dimensionality)
        self.time_step = 0

    def is_done(self):
        return self.time_step >= self.max_steps

    def observe(self):
        return self.state

    def perform_action(self, action):
        next_state = (self.state
                      + self.step_size * action
                      + np.random.multivariate_normal(
                          np.zeros(self.dimensionality),
                          self.step_noise * np.eye(self.dimensionality)))
        reward = self.reward_func(next_state, action)
        self.state = next_state
        self.time_step += 1
        return next_state, reward

    def plot_trajectory(self, trajectory):
        if self.dimensionality == 1:
            for trans_idx in range(len(trajectory.transitions)-1):
                trans = trajectory.transitions[trans_idx]
                plt.plot(
                    [trans_idx, trans_idx+1],
                    [trans.state[0], trans.next_state[0]], 'r')
            plt.gca().xaxis.set_ticks_position('bottom')
            plt.xlabel("Time step")
            plt.ylabel("State")
        elif self.dimensionality == 2:
            # Plot the reward as a contour plot
            # TODO : Make more efficient by changing reward_func to accept arrays
            min_states_values, max_states_values = (
                trajectory.return_states_extremes())
            x = np.linspace(min_states_values[0], max_states_values[0], 50)
            y = np.linspace(min_states_values[1], max_states_values[1], 50)
            xx, yy = np.meshgrid(x, y)
            z = np.zeros_like(xx)
            for idx_x in range(xx.shape[0]):
                for idx_y in range(xx.shape[1]):
                    z[idx_x, idx_y] = self.reward_func(
                        np.array([xx[idx_x, idx_y], yy[idx_x, idx_y]]),
                        None)
            plt.contourf(x, y, z)
            # End of plotting the reward as a contour plot
            for trans_idx in range(len(trajectory.transitions)-1):
                trans = trajectory.transitions[trans_idx]
                plt.plot(
                    [trans.state[0], trans.next_state[0]],
                    [trans.state[1], trans.next_state[1]], 'r')
                plt.gca().set_aspect(1)

    def plot_transitions_dataset(self, transitions_dataset, plot_color='k'):
        if self.dimensionality == 1:
            pass
        elif self.dimensionality == 2:
            # Plot the reward as a contour plot
            # TODO : Make more efficient by changing reward_func to accept arrays
            min_states_values, max_states_values = (
                transitions_dataset.return_states_extremes())
            x = np.linspace(min_states_values[0], max_states_values[0], 50)
            y = np.linspace(min_states_values[1], max_states_values[1], 50)
            xx, yy = np.meshgrid(x, y)
            z = np.zeros_like(xx)
            for idx_x in range(xx.shape[0]):
                for idx_y in range(xx.shape[1]):
                    z[idx_x, idx_y] = self.reward_func(
                        np.array([xx[idx_x, idx_y], yy[idx_x, idx_y]]),
                        None)
            plt.contourf(x, y, z, cmap='Blues')
            # End of plotting the reward as a contour plot
            for trans in transitions_dataset.transitions:
                plt.plot(
                    [trans.state[0], trans.next_state[0]],
                    [trans.state[1], trans.next_state[1]], color=plot_color)
                plt.gca().set_aspect(1)


class EnvContinuousNavigationConfounded(EnvContinuousNavigation):

    def perform_action(self, action):
        next_state = (self.state
                      + self.step_size * action
                      + np.random.multivariate_normal(
                          np.zeros(self.dimensionality),
                          self.step_noise * np.eye(self.dimensionality)))
        next_state[0] -= 0.8 / 2**0.5
        reward = self.reward_func(next_state, action)
        self.state = next_state
        self.time_step += 1
        return next_state, reward


