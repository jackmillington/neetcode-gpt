import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        pre_act = np.dot(x,w)+b
        if activation == 'sigmoid':
            sig = 1 / (1+np.exp(-pre_act))
            return float(np.round(sig, 5))
        else:
            relu = max(0, pre_act)
            return float(np.round(relu, 5))
