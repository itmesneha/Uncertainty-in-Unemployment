import torch
import torch.nn as nn
import torch.optim as optim
from loss import elbo_loss

def train_bayesian_survival_model(model, 
                                  dataloader,
                                  epochs=1, 
                                  learning_rate=1e-3, 
                                  device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
                                  print_every=100):
    """
    Train a Bayesian Hazard Neural Network using Variational Inference.

    Args:
        model (nn.Module): Your Bayesian Hazard Model (e.g., BayesianHazardNN).
        dataloader (DataLoader): Dataloader containing the training data.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.
        print_every (int): Frequency of printing loss during training.

    Returns:
        model: Trained model.
    """

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_kl = 0
        for batch in dataloader:
            x_cat, x_cont, duration_train, event_train = batch
            x_cat = x_cat.to(device)
            x_cont = x_cont.to(device)
            duration_train = duration_train.to(device)
            event_train = event_train.to(device)

            optimizer.zero_grad()

            # Forward pass
            lambda_pred = model(x_cat, x_cont)  # Predict hazard rates
            kl_term = model.kl_loss()     # Compute KL divergence

            # ELBO loss
            loss = elbo_loss(lambda_pred, duration_train, event_train, kl_term)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Accumulate loss and KL
            total_loss += loss.item()
            total_kl += kl_term.item()

        # Logging
        if epoch % print_every == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}, KL: {total_kl / len(dataloader)}")

    return model