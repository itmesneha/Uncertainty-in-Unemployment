import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
class VariationalLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-3))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features).fill_(-3))

    def forward(self, x):
        weight_sigma = torch.exp(self.weight_log_sigma)
        bias_sigma = torch.exp(self.bias_log_sigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        return F.linear(x, weight, bias)

    def kl_loss(self):
        prior = Normal(0, 1)
        q_weight = Normal(self.weight_mu, torch.exp(self.weight_log_sigma))
        q_bias = Normal(self.bias_mu, torch.exp(self.bias_log_sigma))
        kl = torch.distributions.kl_divergence(q_weight, prior).sum() + torch.distributions.kl_divergence(q_bias, prior).sum()
        return kl


class BayesianHazardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = VariationalLinear(input_dim, hidden_dim)
        self.fc2 = VariationalLinear(hidden_dim, hidden_dim)
        self.out_layer = VariationalLinear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        log_lambda = self.out_layer(x)
        lambda_ = torch.exp(log_lambda)  # Ensure positive hazard
        return lambda_

    def kl_loss(self):
        return self.fc1.kl_loss() + self.fc2.kl_loss() + self.out_layer.kl_loss()
