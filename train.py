import torch
import torch.nn as nn
import torch.optim as optim
import os
from loss import elbo_loss_log_normal

def train_bayesian_survival_model(model, 
                                  train_loader,
                                  val_loader,
                                  epochs=1000, 
                                  learning_rate=1e-4, 
                                  device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
                                  print_every=50,
                                  eval_every=50,
                                  patience=5,
                                  checkpoint_path="best_model.pt"):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_kl = 0

        for batch in train_loader:
            x_cat, x_cont, duration, event = [b.to(device) for b in batch]

            optimizer.zero_grad()
            mu, sigma = model(x_cat, x_cont)
            kl = model.kl_loss()
            loss = elbo_loss_log_normal(mu, sigma, duration, event, kl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_kl += kl.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_kl = total_kl / len(train_loader)

        # Evaluate on validation set
        if epoch % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_cat, x_cont, duration, event = [b.to(device) for b in batch]
                    mu, sigma = model(x_cat, x_cont)
                    kl = model.kl_loss()
                    loss = elbo_loss_log_normal(mu, sigma, duration, event, kl)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, KL: {avg_kl:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Checkpointing and early stopping
            if avg_val_loss < best_val_loss - 1e-4:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
                print("Saved new best model.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("🛑 Early stopping triggered.")
                    break
        else:
            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, KL: {avg_kl:.4f}")

    # Load best model
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print("🔁 Loaded best model from checkpoint.")

    return model
