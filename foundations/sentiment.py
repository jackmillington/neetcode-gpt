import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.embed = nn.Embedding(vocabulary_size, 16)
        self.l = nn.Linear(16, 1)
        self.s = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        # Return a B, 1 tensor and round to 4 decimal places
        x = self.embed(x)
        x = torch.mean(x, dim=1)
        x = self.l(x)
        x = self.s(x)
        return torch.round(x, decimals=4)
