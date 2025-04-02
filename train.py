import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import pandas as pd


def train_model(model, train_loader, val_loader, test_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training with: {device}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    training_log = []

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            y0 = sequences[:, 0, :]
            t = torch.linspace(0, 1, sequences.shape[1], device=device)

            optimizer.zero_grad()
            outputs = model(y0, t)
            pred = outputs[-1]
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.6f}")

        val_loss = None
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_loss_total = 0
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences, targets = sequences.to(device), targets.to(device)
                    y0 = sequences[:, 0, :]
                    t = torch.linspace(0, 1, sequences.shape[1], device=device)
                    outputs = model(y0, t)
                    pred = outputs[-1]
                    val_loss_total += loss_fn(pred, targets).item()

            val_loss = val_loss_total / len(val_loader)
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()

        training_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss if val_loss is not None else "NA"
        })

    os.makedirs(config['save_dir'], exist_ok=True)
    model_path = os.path.join(config['save_dir'], 'best_model.pth')
    if best_model_state:
        torch.save(best_model_state, model_path)
        print(f"✅ Best model saved to {model_path}")
    else:
        torch.save(model.state_dict(), model_path)
        print(f"⚠️ No validation used. Last model saved to {model_path}")

    log_df = pd.DataFrame(training_log)
    log_path = os.path.join(config['save_dir'], 'training_log.csv')
    log_df.to_csv(log_path, index=False)
    print(f"📊 Training log saved to {log_path}")

    return model, test_loader

