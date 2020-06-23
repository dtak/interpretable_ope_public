import numpy as np
import matplotlib.pyplot as plt


class EnvCancer:
    def __init__(self, dose_penalty=0, max_steps=30, transition_noise=0.0):
        ''' Intitilize patient parameters '''
        self.kde = 0.24
        self.lambda_p = 0.121
        self.k_qpp = 0.0031
        self.k_pq = 0.0295
        self.gamma = 0.729
        self.delta_qp = 0.00867
        self.k = 100
        self.dose_penalty = dose_penalty
        self.max_steps = max_steps
        self.state = None
        self.time_step = None
        self.transition_noise = transition_noise

    def reset(self):
        C = 0
        P = 7.13
        Q = 41.2
        Q_p = 0

        self.state = np.array([C, P, Q, Q_p])
        self.time_step = 0

    def is_done(self):
        return self.time_step >= self.max_steps

    def observe(self):
        return self.state

    def perform_action(self, action):
        C, P, Q, Q_p = self.state
        P_star = P + Q + Q_p
        if action == 1:
            C += 1
        C = C - self.kde * C
        P = (P + self.lambda_p * P * (1-P_star/self.k) + self.k_qpp * Q_p
             - self.k_pq * P - self.gamma * C * self.kde * P)
        Q = Q + self.k_pq * P - self.gamma * C * self.kde * Q
        Q_p = (Q_p + self.gamma * C * self.kde * Q - self.k_qpp * Q_p
               - self.delta_qp * Q_p)



        # if np.random.rand() < 1/1500:
        #     P += 5



        next_state = np.array([C, P, Q, Q_p])
        noise = 1 + self.transition_noise * np.random.randn(4)
        next_state *= noise
        self.state = next_state
        P_star_new = P + Q + Q_p
        reward = (P_star - P_star_new) - self.dose_penalty * C

        # reward *= (1 + np.random.randn())


        self.time_step += 1
        return next_state, reward

    # def plot_trajectory(self, trajectory, plot_color='b', line_width=1):
    #     for k in range(4):
    #         plt.subplot(2, 2, k + 1)
    #         plt.plot(np.arange(self.max_steps),
    #                  [trajectory.transitions[t].state[k]
    #                   for t in range(self.max_steps)],
    #                  color=plot_color,
    #                  linewidth=line_width)
    #
    #     plt.subplot(2, 2, 1)
    #     plt.ylabel(r'$C$', fontsize=14)
    #     plt.subplot(2, 2, 2)
    #     plt.ylabel(r'$P$', fontsize=14)
    #     plt.subplot(2, 2, 3)
    #     plt.ylabel(r'$Q$', fontsize=14)
    #     plt.xlabel('Time', fontsize=14)
    #     plt.subplot(2, 2, 4)
    #     plt.ylabel(r'$Q_p$', fontsize=14)
    #     plt.xlabel('Time', fontsize=14)
    #     plt.tight_layout()
    #
    #     # for k in range(4):
    #     #     plt.subplot(3, 2, k + 1)
    #     #     plt.plot(np.arange(self.max_steps),
    #     #              [trajectory.transitions[t].state[k]
    #     #               for t in range(self.max_steps)])
    #     # plt.subplot(3, 2, 5)
    #     # plt.plot(np.arange(self.max_steps),
    #     #          [trajectory.transitions[t].reward
    #     #           for t in range(self.max_steps)])


    def plot_trajectory(self, trajectory, plot_color='b', line_width=1):
        # This function is for only plotting two states for the OPE influence analysis paper

        plt.subplot(1, 2, 1)
        plt.plot(np.arange(self.max_steps),
                 [trajectory.transitions[t].state[0]
                  for t in range(self.max_steps)],
                 color=plot_color,
                 linewidth=line_width)
        plt.ylabel(r'$C$', fontsize=14)
        plt.xlabel('Time', fontsize=14)
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(self.max_steps),
                 [trajectory.transitions[t].state[3]
                  for t in range(self.max_steps)],
                 color=plot_color,
                 linewidth=line_width)
        plt.ylabel(r'$Q_p$', fontsize=14)
        plt.xlabel('Time', fontsize=14)
        plt.tight_layout()

        # for k in range(4):
        #     plt.subplot(2, 2, k + 1)
        #     plt.plot(np.arange(self.max_steps),
        #              [trajectory.transitions[t].state[k]
        #               for t in range(self.max_steps)],
        #              color=plot_color)

        # for k in range(4):
        #     plt.subplot(3, 2, k + 1)
        #     plt.plot(np.arange(self.max_steps),
        #              [trajectory.transitions[t].state[k]
        #               for t in range(self.max_steps)])
        # plt.subplot(3, 2, 5)
        # plt.plot(np.arange(self.max_steps),
        #          [trajectory.transitions[t].reward
        #           for t in range(self.max_steps)])

    # def plot_transition(self, trans, plot_color='r'):
    #     for k in range(4):
    #         plt.subplot(2, 2, k + 1)
    #         plt.plot([trans.time_step, trans.time_step + 1],
    #                  [trans.state[k], trans.next_state[k]],
    #                  '.-', color=plot_color)

    def plot_transition(self, trans, plot_color='r'):
    # This function is for only plotting two states for the OPE influence analysis paper
        plt.subplot(1, 2, 1)
        plt.plot([trans.time_step, trans.time_step + 1],
                 [trans.state[0], trans.next_state[0]],
                 '.-', color=plot_color)
        plt.subplot(1, 2, 2)
        plt.plot([trans.time_step, trans.time_step + 1],
                 [trans.state[3], trans.next_state[3]],
                 '.-', color=plot_color)

    # def plot_transitions_dataset(self, transitions_dataset, plot_color='r'):
    #     for k in range(4):
    #         plt.subplot(2, 2, k + 1)
    #         for trans_idx in range(transitions_dataset.return_num_trans()):
    #             trans = transitions_dataset.transitions[trans_idx]
    #             if isinstance(plot_color, str):
    #                 current_color = plot_color
    #             else:
    #                 current_color = plot_color[trans_idx]
    #             plt.plot([trans.time_step, trans.time_step + 1],
    #                      [trans.state[k], trans.next_state[k]],
    #                      '.-', color=current_color)

        # for k in range(4):
        #     plt.subplot(3, 2, k + 1)
        #     for trans_idx in range(transitions_dataset.return_num_trans()):
        #         trans = transitions_dataset.transitions[trans_idx]
        #         if isinstance(plot_color, str):
        #             current_color = plot_color
        #         else:
        #             current_color = plot_color[trans_idx]
        #         plt.plot([trans.time_step, trans.time_step + 1],
        #                  [trans.state[k], trans.next_state[k]],
        #                  '.-', color=current_color)
        # plt.subplot(3, 2, 5)
        # for trans_idx in range(transitions_dataset.return_num_trans()):
        #     trans = transitions_dataset.transitions[trans_idx]
        #     if isinstance(plot_color, str):
        #         current_color = plot_color
        #     else:
        #         current_color = plot_color[trans_idx]
        #     plt.plot([trans.time_step, trans.time_step + 1],
        #              [trans.reward, trans.reward],
        #              '.-', color=current_color)


    def plot_transitions_dataset(self, transitions_dataset, plot_color='r'):
        # This function is for only plotting two states for the OPE influence analysis paper
        plt.subplot(1, 2, 1)
        for trans_idx in range(transitions_dataset.return_num_trans()):
            trans = transitions_dataset.transitions[trans_idx]
            if isinstance(plot_color, str):
                current_color = plot_color
            else:
                current_color = plot_color[trans_idx]
            plt.plot([trans.time_step, trans.time_step + 1],
                     [trans.state[0], trans.next_state[0]],
                     '.-', color=current_color)
        plt.subplot(1, 2, 2)
        for trans_idx in range(transitions_dataset.return_num_trans()):
            trans = transitions_dataset.transitions[trans_idx]
            if isinstance(plot_color, str):
                current_color = plot_color
            else:
                current_color = plot_color[trans_idx]
            plt.plot([trans.time_step, trans.time_step + 1],
                     [trans.state[3], trans.next_state[3]],
                     '.-', color=current_color)
