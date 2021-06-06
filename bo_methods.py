
import random
from botorch.acquisition.analytic import ConstrainedExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_step_lookahead import _construct_sample_weights
from botorch.sampling.samplers import SobolQMCNormalSampler
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, ProbabilityOfImprovement, PosteriorMean
import torch
import numpy as np

class RandomSearch:

    def __init__(self):

        print("Using random search method...")

    def observe_and_suggest(self, X_obs, y_obs, X_pen):

        size_pending_eval = len(X_pen)
        idx = random.randint(0, size_pending_eval-1)
        return idx


class GaussianProcess:

    def __init__(self, acq_name="UCB"):
        print("Using Gaussian Process as method...")

        self.acq_name = acq_name        
        #initialize acquisition functions

    def get_acquisition(self, gp = None, best_f =0.0):

        assert gp != None, "The model was not correctly specified"
        

        if self.acq_name == "UCB":
            return UpperConfidenceBound(gp, beta=0.1)
        
        elif self.acq_name == "EI":
            return ExpectedImprovement(gp, best_f=best_f)

        elif self.acq_name == "PM":
            return PosteriorMean(gp)

        elif self.acq_name == "PI":
            return ProbabilityOfImprovement(gp, best_f=best_f)

        elif self.acq_name == "qEI":
            sampler = SobolQMCNormalSampler(1000)
            return qExpectedImprovement(gp, best_f=best_f)
            
    def observe_and_suggest(self, X_obs, y_obs, X_pen):



        #fit the gaussian process
        dim = X_obs.shape[1]
        X_obs = torch.FloatTensor(X_obs)
        y_obs = torch.FloatTensor(y_obs)
        X_pen = torch.FloatTensor(X_pen).reshape(-1,1,dim)
        gp = SingleTaskGP(X_obs, y_obs)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        best_f = torch.max(y_obs)
        acq = self.get_acquisition( gp=gp, best_f=best_f)
        eval_acq = acq( X_pen).detach().numpy()
        
        return np.argmax(eval_acq)
        #sample the acquisition functions




        
