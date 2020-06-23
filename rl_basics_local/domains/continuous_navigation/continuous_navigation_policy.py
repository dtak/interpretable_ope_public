import numpy as np
import random


class PolicyContinuousNavigation:
    def __init__(self,
                 dimensionality=1,
                 direction_vector=None,
                 direction_vector_probability=None,
                 randomness_magnitude=0):
        self.dimensionality = dimensionality
        self.randomness_magnitude = randomness_magnitude
        self.direction_vector = direction_vector
        self.direction_vector_probability = direction_vector_probability

    def __call__(self, state, time_step):
        if self.dimensionality == 1:
            pass
        else:
            if self.direction_vector is None:
                action = np.zeros(self.dimensionality)
            elif isinstance(self.direction_vector, list):
                if self.direction_vector_probability is None:
                    action = random.choice(self.direction_vector) + 0.0
                else:
                    action = (
                        self.direction_vector[np.random.choice(
                            len(self.direction_vector),
                            p=self.direction_vector_probability)]
                        + 0.0)
            else:
                action = self.direction_vector + 0.0
            action /= np.sum(action**2)**0.5
            if self.randomness_magnitude > 0:
                perturbation = np.random.randn(self.dimensionality)
                perturbation /= np.sum(perturbation**2)**0.5
                action += perturbation * self.randomness_magnitude
                action /= np.sum(action ** 2) ** 0.5
            return action

    def return_proba(self, state, action, time_step):
        pass
