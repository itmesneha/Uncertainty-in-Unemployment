import torch
import numpy as np
from torch.distributions import Normal

def load_model_from_checkpoint(model_class, 
                               category_sizes, 
                               n_cont_features, 
                               checkpoint_path, 
                               device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")):
    """
    Load a trained Bayesian survival model from a checkpoint.

    Args:
        model_class (nn.Module): The model class (e.g., BayesianRiskNetwork).
        category_sizes (list): List of categorical feature sizes (for embeddings).
        n_cont_features (int): Number of continuous features.
        checkpoint_path (str): Path to the saved model checkpoint.
        device (torch.device): Torch device.

    Returns:
        model (nn.Module): Loaded model ready for prediction.
    """
    model = model_class(category_sizes, n_cont_features).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    print(f"Loaded model from checkpoint: {checkpoint_path}")
    return model



def predict_survival_probability(model, 
                                 x_cat, 
                                 x_cont, 
                                 time_points, 
                                 n_samples=1000, 
                                 device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")):
    """
    Predict survival probabilities S(t | x) with uncertainty via Monte Carlo sampling.

    Args:
        model (nn.Module): Trained Bayesian survival model.
        x_cat (torch.Tensor): Categorical covariates (single sample).
        x_cont (torch.Tensor): Continuous covariates (single sample).
        time_points (list or np.array): Time points to compute S(t | x).
        n_samples (int): Number of Monte Carlo samples.
        device (torch.device): Device to run computation on.

    Returns:
        mean_probs (np.array): Mean survival probabilities across samples.
        lower (np.array): Lower bound of 95% credible interval.
        upper (np.array): Upper bound of 95% credible interval.
    """
    model.eval()
    x_cat = x_cat.to(device).unsqueeze(0)  # Add batch dimension
    x_cont = x_cont.to(device).unsqueeze(0)

    survival_samples = []

    with torch.no_grad():
        for _ in range(n_samples):
            mu, sigma = model(x_cat, x_cont)
            sigma = sigma.clamp(min=1e-6)  # Stability safeguard

            normal_dist = Normal(mu, sigma)
            log_t = torch.log(torch.tensor(time_points, dtype=torch.float32).to(device).unsqueeze(1))
            survival_prob = 1 - normal_dist.cdf(log_t)
            survival_samples.append(survival_prob.squeeze().cpu().numpy())

    survival_samples = np.stack(survival_samples)
    mean_probs = survival_samples.mean(axis=0)
    lower = np.percentile(survival_samples, 2.5, axis=0)
    upper = np.percentile(survival_samples, 97.5, axis=0)

    return mean_probs, lower, upper

