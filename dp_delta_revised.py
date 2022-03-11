import diffprivlib.models as dp

import warnings
import numpy as np
from diffprivlib.mechanisms import Laplace, LaplaceBoundedDomain
from diffprivlib.models.utils import _check_bounds
from diffprivlib.utils import PrivacyLeakWarning, warn_unused_args

class GaussianNBEpsDelta(dp.GaussianNB):
    def __init__(self, epsilon=1, delta = 0, bounds=None, priors=None, var_smoothing=1e-9):
        super().__init__(priors, var_smoothing)

        self.epsilon = epsilon
        self.delta = delta
        self.bounds = bounds

    def _randomise(self, mean, var, n_samples):
        """Randomises the learned means and variances subject to differential privacy."""
        features = var.shape[0]

#         delta MUST be scaled as well!! TODO Corollary B.2 p.268 Privacy Book
        # divide by 2n because running n queries twice (two mechs)
#         local_epsilon = self.epsilon / 2
#         local_epsilon /= features

#         local_delta = self.delta / 2
#         local_delta /= features

#         local_epsilon = self.epsilon / (2*np.sqrt(2*2*features*np.log(1/self.delta)))
#         local_delta = 0
        
        #DWORK Book p.52
        if self.epsilon < 1:
            local_epsilon = self.epsilon / (2*np.sqrt(2*2*features*np.log(1/self.delta)))
            local_delta = 0
        else:
            import sympy as sym
            # let e'=x and e=y
            x,y,k,d = sym.symbols(('x','y','k','d'))
            eq1=sym.Eq(sym.sqrt(2*k*sym.log(1/d))*y+k*y*(sym.exp(y)-1)-x,0)
            local_epsilon = sym.nsolve(eq1.subs({x:self.epsilon, d:self.delta, k:2*features}), y, 1)
               
#             local_epsilon = (np.sqrt(2)*
#                 (np.sqrt(features*np.log(1/self.delta)+2*features*self.epsilon) -
#                 np.sqrt(features*np.log(1/self.delta))))/(2*features)
            local_delta = 0
        
        if len(self.bounds) != features:
            raise ValueError("Bounds must be specified for each feature dimension")

        new_mu = np.zeros_like(mean)
        new_var = np.zeros_like(var)

        for feature in range(features):
            local_diameter = self.bounds[feature][1] - self.bounds[feature][0]
            mech_mu = Laplace().set_sensitivity(local_diameter / n_samples)\
                .set_epsilon_delta(local_epsilon, local_delta)
            mech_var = LaplaceBoundedDomain()\
                .set_sensitivity((n_samples - 1) * local_diameter ** 2 / n_samples ** 2)\
                .set_epsilon_delta(local_epsilon, local_delta)\
                .set_bounds(0, float("inf"))

            new_mu[feature] = mech_mu.randomise(mean[feature])
            new_var[feature] = mech_var.randomise(var[feature])

        return new_mu, new_var
