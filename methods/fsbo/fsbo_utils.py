
import numpy as np
from scipy.stats import norm
import torch



def EI(incumbent, mu, stddev):
    
    mu             = mu.reshape(-1,)
    stddev         = stddev.reshape(-1,)

    with np.errstate(divide='warn'):
        imp = mu - incumbent
        Z = imp / stddev
        score = imp * norm.cdf(Z) + stddev * norm.pdf(Z)

    return score
    
    
class Metric(object):
    def __init__(self,prefix='train: '):
        self.reset()
        self.message=prefix + "loss: {loss:.2f} - noise: {log_var:.2f} - mse: {mse:.2f}"
        
    def update(self,loss,noise,mse):
        self.loss.append(np.asscalar(loss))
        self.noise.append(np.asscalar(noise))
        self.mse.append(np.asscalar(mse))
    
    def reset(self,):
        self.loss = []
        self.noise = []
        self.mse = []
    
    def report(self):
        return self.message.format(loss=np.mean(self.loss),
                            log_var=np.mean(self.noise),
                            mse=np.mean(self.mse))
    
    def get(self):
        return {"loss":np.mean(self.loss),
                "noise":np.mean(self.noise),
                "mse":np.mean(self.mse)}
    
def totorch(x,device):

    return torch.Tensor(x).to(device)    
