import numpy as np


class PolicyCancer:
    def __init__(self, months_for_treatment=15, eps_behavior=0):
        self.num_actions = 2
        self.months_for_treatment = months_for_treatment
        self.eps_behavior = eps_behavior

    def __call__(self, state, time_step):
        if np.random.rand() < self.eps_behavior and time_step > 0:
            return np.array([np.random.choice(2)])
        if time_step <= self.months_for_treatment:
            return np.array([1])
        else:
            return np.array([0])

    def return_proba(self, state, action, time_step):
        if time_step <= self.months_for_treatment:
            if action == 1:
                return 1.0 - self.eps_behavior/2
            else:
                return self.eps_behavior/2
        else:
            if action == 1:
                return self.eps_behavior/2
            else:
                return 1 - self.eps_behavior/2