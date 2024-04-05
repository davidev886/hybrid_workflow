import numpy as np
from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.greens_function import greens_function


class S2Mixed(EstimatorBase):
    def __init__(self, ham):
        super().__init__()
        # We define a dictionary to contain whatever we want to compute.
        # Note we typically want to separate the numerator and denominator of
        # the estimator
        # We require complex valued buffers for accumulation
        self._data = {
            "S2Numer": np.zeros((1), dtype=np.complex128),
            "S2Denom": np.zeros((1), dtype=np.complex128),
        }
        # We also need to specify the shape of the desired estimator
        self._shape = (1,)
        # Optional but good to know (we can redirect to custom filepath (ascii)
        # and / or print to stdout but we shouldnt do this for non scalar
        # quantities
        self.print_to_stdout = True
        self.ascii_filename = None
        # Must specify that we're dealing with array valued estimator
        self.scalar_estimator = False

    def compute_estimator(self, system, walkers, hamiltonian, trial):
        greens_function(walkers, trial, build_full=True)

        ndown = system.ndown
        nup = system.nup
        Ms = (nup - ndown) / 2.0
        two_body = -np.einsum("wij,wji->w", walkers.Ga, walkers.Gb)
        two_body = two_body * walkers.weight

        denom = np.sum(walkers.weight)
        numer = np.sum(two_body) + denom * (Ms * (Ms + 1) + ndown)

        self["S2Numer"] = numer
        self["S2Denom"] = denom
