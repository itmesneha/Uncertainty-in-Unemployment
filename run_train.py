import torch
from torch.utils.data import DataLoader

from data import UnemploymentSurvivalDataset, create_train_val_loaders
from model import BayesianRiskNetwork
from train import train_bayesian_survival_model

def run_train(dataframe,
              batch_size=128,
              epochs=1000,
              learning_rate=1e-3,
              censoring_rate=1,
              print_every=100):
    """
    Full training pipeline for Bayesian Hazard Neural Network.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        input_dim (int): Number of input features.
        hidden_dim (int): Hidden layer size.
        batch_size (int): Batch size.
        epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
        censoring_rate (float): Probability of censoring for '52 and over' group.
        print_every (int): Logging frequency.

    Returns:
        Trained model.
    """
    # Set the MPS device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Dataset and DataLoader initialization
    dataset = UnemploymentSurvivalDataset(dataframe, censoring_rate=censoring_rate)
    train_loader, val_loader = create_train_val_loaders(dataset, val_ratio=0.2, batch_size=batch_size)

    # Initialize the model
    category_sizes = dataset.get_category_sizes()
    n_cont_features = dataset.get_continuous_feature_count()
    model = BayesianRiskNetwork(category_sizes=category_sizes, n_cont_features=n_cont_features)
    model.to(device)
    # Training loop using your existing function
    
    model = train_bayesian_survival_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        print_every=print_every,
        eval_every=10
    )

    print("✅ Training finished successfully.")
    return model

if __name__ == "__main__":
    dataframe = "datasets/unemployment_survival_data.csv"
    run_train(dataframe=dataframe, epochs=5000, print_every=10)