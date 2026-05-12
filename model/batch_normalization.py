import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)

        if training:
            ub = np.mean(x, axis=0)
            varb = np.var(x, axis=0)
            x_hat = (x-ub) / np.sqrt(varb + 1e-5)
            running_mean = (1-momentum) * running_mean + momentum * ub
            running_var = (1-momentum) * running_var + momentum * varb
        else:
            x_hat = (x-running_mean) / np.sqrt(running_var + 1e-5)
        out = gamma * x_hat + beta
        return (np.round(out, 4), np.round(running_mean, 4), np.round(running_var, 4))
