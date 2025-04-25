import torch

def elbo_loss(lambda_pred, duration, event, kl_term, weights=None):
    likelihood = (event * torch.log(lambda_pred + 1e-8) - lambda_pred * duration)
    if weights is not None:
        likelihood = likelihood * weights  # Apply weights here
    log_likelihood = likelihood.sum()
    return -log_likelihood + kl_term  # Negative ELBO