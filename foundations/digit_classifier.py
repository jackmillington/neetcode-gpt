import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.l = nn.Linear(784, 512)
        self.r = nn.ReLU()
        self.d = nn.Dropout(0.2)
        self.l2 = nn.Linear(512, 10)
        self.s = nn.Sigmoid()
        pass

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        return torch.round(self.s(self.l2(self.d(self.r(self.l(images))))), decimals=4)
