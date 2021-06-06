import numpy as np
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates import RandomForest as RF

class RandomForest:

    def __init__(self, acq_name="Entropy"):

        print("Using Random Forest as method...")

        self.surrogate = RF.ExtraForest()
        self.acquistion = Acquisition(mode=acq_name)

    
    def observe_and_suggest(self, X_obs, y_obs, X_pen):
        
        self.surrogate.fit(X_obs, y_obs.reshape(-1))
        tau = np.max(y_obs).reshape(-1)
        mean, std = self.surrogate.predict(X_pen, return_std=True)
        eval_acq = self.acquistion.eval(tau, mean, std)

        return np.argmax(eval_acq) 
