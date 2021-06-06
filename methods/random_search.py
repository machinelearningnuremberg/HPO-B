import random

class RandomSearch:

    def __init__(self):

        print("Using random search method...")

    def observe_and_suggest(self, X_obs, y_obs, X_pen):

        size_pending_eval = len(X_pen)
        idx = random.randint(0, size_pending_eval-1)
        return idx
