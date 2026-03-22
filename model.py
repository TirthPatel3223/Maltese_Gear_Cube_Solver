"""
CategoricalResNet — HL-Gauss Value Network for Maltese Gear Cube

Architecture (following DeepCubeA, Agostinelli et al., 2019):
  Input (816) → FC+BN+ReLU (7000) → FC+BN+ReLU (2000)
  → 6× ResBlocks (2000) → Linear (max_dist)

Output: logits over max_dist bins representing cost-to-go distribution.
Trained with KL-divergence against HL-Gauss soft targets
(Farebrother et al., 2024, "Stop Regressing").

Expected value (CTG) is computed as E[softmax(logits) · indices].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalResNet(nn.Module):
    def __init__(
        self,
        state_dim=816,
        hidden_dim=5000,
        res_dim=1000,
        num_blocks=6,
        max_dist=505,
    ):
        super().__init__()
        self.max_dist = max_dist

        # Input projection
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Dimensionality reduction to residual stream
        self.fc2 = nn.Linear(hidden_dim, res_dim)
        self.bn2 = nn.BatchNorm1d(res_dim)

        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList([
                    nn.Linear(res_dim, res_dim),
                    nn.BatchNorm1d(res_dim),
                    nn.Linear(res_dim, res_dim),
                    nn.BatchNorm1d(res_dim),
                ])
            )

        # Output head: logits over categorical bins
        self.fc_out = nn.Linear(res_dim, max_dist)

        # Pre-computed indices for expected value calculation
        self.register_buffer("indices", torch.arange(0, max_dist).float())

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))

        for block in self.blocks:
            res = x
            x = F.relu(block[1](block[0](x)))
            x = F.relu(block[3](block[2](x)) + res)

        return self.fc_out(x)

    def get_ctg(self, logits):
        """Compute expected cost-to-go from logits: E[softmax(logits) · indices]."""
        probs = F.softmax(logits, dim=-1)
        return torch.sum(probs * self.indices, dim=-1)
