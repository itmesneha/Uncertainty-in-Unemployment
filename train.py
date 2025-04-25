import torch
import torch.nn as nn
import torch.optim as optim
from loss import elbo_loss

def train_bayesian_survival_model(model, 
                                  x_train, 
                                  duration_train, 
                                  event_train, 
                                  weight_train,
                                  epochs=1000, 
                                  learning_rate=1e-3, 
                                  print_every=100):
    """
    Train a Bayesian Hazard Neural Network using Variational Inference.

    Args:
        model (nn.Module): Your Bayesian Hazard Model (e.g., BayesianHazardNN).
        x_train (torch.Tensor): Input covariates (shape: [batch_size, input_dim]).
        duration_train (torch.Tensor): Duration/time-to-event data (shape: [batch_size, 1]).
        event_train (torch.Tensor): Event indicator (1 = observed, 0 = censored) (shape: [batch_size, 1]).
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.
        print_every (int): Frequency of printing loss during training.

    Returns:
        model: Trained model.
    """

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        lambda_pred = model(x_train)  # Predict hazard rates
        kl_term = model.kl_loss()     # Compute KL divergence

        # ELBO loss
        loss = elbo_loss(lambda_pred, duration_train, event_train, kl_term, weight_train)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Logging
        if epoch % print_every == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.item()}, KL: {kl_term.item()}")

    return model