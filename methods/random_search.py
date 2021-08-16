import random
import numpy as np

class RandomSearch:

    def __init__(self):

        print("Using random search method...")

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None):

        if X_pen is not None:
            size_pending_eval = len(X_pen)
            idx = random.randint(0, size_pending_eval-1)
            return idx

        else:
            dim = len(X_obs[0])
            bounds = tuple([(0,1) for i in range(dim)])
            x_new = np.array([random.uniform(lower, upper) for upper, lower in bounds]).reshape(-1, dim)

            return x_new



