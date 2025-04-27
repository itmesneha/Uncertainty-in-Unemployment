import torch
from torch.distributions import Normal, HalfNormal

def elbo_loss_log_normal(mu, sigma, duration, event, kl_term):
    eps = 1e-8
    log_t = torch.log(duration + eps)
    normal_dist = Normal(mu, sigma)
    log_pdf = normal_dist.log_prob(log_t) - torch.log(duration + eps)
    survival_prob = 1 - normal_dist.cdf(log_t)
    log_survival = torch.log(survival_prob + eps)
    log_likelihood = event * log_pdf + (1 - event) * log_survival
    log_likelihood = log_likelihood.sum()
    loss = -log_likelihood + kl_term
    return log_likelihood, loss
