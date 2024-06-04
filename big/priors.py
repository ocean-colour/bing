""" Classes and methods to deal with priors """

import numpy as np

class Priors:

    nparam:int = None
    """
    The number of parameters for the model
    """
    
    approach:str = None
    """
    Approach to priors
    """

    def __init__(self, approach:str, nparam:int):
        self.approach = approach
        self.nparam = nparam

        self.set_priors()

    def set_priors(self):
        """
        Set the priors for the model

        """
        if self.approach == 'log':
            self.priors = self.log_priors()
        else:
            raise ValueError(f"Unknown prior approach: {self.prior_approach}")

    def log_priors(self):
        """
        Calculate the log-priors for the model

        Returns:
            float: The log-priors
        """
        priors = np.zeros((self.nparam, 2))
        priors[:,0] = -6
        priors[:,1] = 5

        return priors

    def calc(self, params:np.ndarray):
        if self.approach == 'log':
            return np.any(params < self.priors[:,0]) or \
                np.any(params > self.priors[:,1])
        else:
            raise ValueError(f"Unknown prior approach: {self.prior_approach}")
        