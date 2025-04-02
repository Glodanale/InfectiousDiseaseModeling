# Title: models.py
# Author: Alex Glover
# Created: March 24, 2025
# Last Updated: March 24, 2025

import torch
import torch.nn as nn
from torchdyn.models import NeuralODE


class LSTMHybrid(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, t, y, args=None):
        if y.dim() == 2:
            y = y.unsqueeze(1)
        out, _ = self.lstm(y)
        return self.fc(out[:, -1, :])

class HybridODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, solver="tsit5", sensitivity="adjoint"):
        super().__init__()
        self.ode_func = LSTMHybrid(input_dim, hidden_dim, num_layers, output_dim)
        self.ode_solver = NeuralODE(self.ode_func, solver=solver, sensitivity=sensitivity)

    def forward(self, y0, t, args=None):
        t_eval, sol = self.ode_solver(y0, t)
        return sol

