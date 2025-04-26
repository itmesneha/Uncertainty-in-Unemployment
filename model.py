import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def sample_half_normal(scale=1.0, shape=(1,)):
    """Sample sigma from Half-Normal distribution."""
    return torch.abs(torch.randn(shape) * scale)  # Half-Normal sampling

def compute_embedding_output_size(category_sizes):
    return sum([min(50, (size // 2) + 1) for size in category_sizes])

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


class EmbeddingBlock(nn.Module):
    def __init__(self, category_sizes):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, min(50, (num_categories // 2) + 1))
            for num_categories in category_sizes
        ])

    def forward(self, x_cat):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(embedded, dim=1)
    

class ContinuousBlock(nn.Module):
    def __init__(self, n_cont_features):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(n_cont_features)

    def forward(self, x_cont):
        return self.batch_norm(x_cont)



class BayesianRiskNetwork(nn.Module):
    def __init__(self, category_sizes, n_cont_features):
        super().__init__()
        self.embed = EmbeddingBlock(category_sizes)
        self.cont_block = ContinuousBlock(n_cont_features)

        embedding_output_size = compute_embedding_output_size(category_sizes)
        self.linear1 = VariationalLinear(embedding_output_size + n_cont_features, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.dropout1 = nn.Dropout(0.6)

        self.linear2 = VariationalLinear(200, 70)
        self.bn2 = nn.BatchNorm1d(70)
        self.dropout2 = nn.Dropout(0.4)

        self.linear_out = VariationalLinear(70, 2)

    def forward(self, x_cat, x_cont):
        x = torch.cat([self.embed(x_cat), self.cont_block(x_cont)], dim=1)
        x = self.dropout1(F.relu(self.bn1(self.linear1(x))))
        x = self.dropout2(F.relu(self.bn2(self.linear2(x))))
        out = self.linear_out(x)
        mu = out[:, 0:1]
        sigma = torch.exp(out[:, 1:2])  # Ensure positivity
        return mu, sigma
    
    def kl_loss(self):
        return self.linear1.kl_loss() + self.linear2.kl_loss() + self.linear_out.kl_loss()