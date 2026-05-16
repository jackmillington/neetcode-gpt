import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        torch.manual_seed(0)
        tokenised = raw_dataset.split()
        indicies = torch.randint(low=0, high=len(tokenised) - context_length, size=(batch_size,)).tolist()
        X = []
        Y = []
        for idx in indicies:
            X.append(tokenised[idx:idx+context_length])
            Y.append(tokenised[idx+1:idx+1+context_length])
        return X, Y
