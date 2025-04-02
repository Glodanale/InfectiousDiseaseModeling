import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import wasserstein_distance
from statsmodels.tsa.stattools import acf
import torch


def flatten_and_align_sequences(actual, predicted):
    actual_aligned = []
    predicted_aligned = []

    for a, p in zip(actual, predicted):
        if a.shape == p.shape:
            actual_aligned.append(a)
            predicted_aligned.append(p)
        else:
            # Match both to the same number of time steps
            min_len = min(a.shape[0], p.shape[0])
            print(f"⚠️ Mismatch detected: actual={a.shape}, pred={p.shape}. Truncating to {min_len} steps.")
            actual_aligned.append(a[:min_len])
            predicted_aligned.append(p[:min_len])

    # Flatten: (batch, time, features) → (batch*time, features)
    actual_flat = np.concatenate(actual_aligned, axis=0).reshape(-1, actual_aligned[0].shape[-1])
    predicted_flat = np.concatenate(predicted_aligned, axis=0).reshape(-1, predicted_aligned[0].shape[-1])

    # ✅ Print to debug
    print(f"✅ Final actual_flat shape: {actual_flat.shape}")
    print(f"✅ Final predicted_flat shape: {predicted_flat.shape}")

    return actual_flat, predicted_flat


def visualize_supervised(model, test_loader, scaler_path='scaler.pkl', output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    model.eval()
    all_actual, all_predicted = [], []
    with torch.no_grad():
        for sequence, _ in test_loader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sequence = sequence.to(device)
            y0 = sequence[:, 0, :]
            t = torch.linspace(0, 1, sequence.shape[1], device=device)
            
            prediction = model(y0, t).cpu().numpy()  # (T, B, F)
            prediction = np.transpose(prediction, (1, 0, 2))  # → (B, T, F)
            
            actual = sequence.cpu().numpy()  # already (B, T, F)
            all_actual.append(actual)
            all_predicted.append(prediction)

    actual_flat, predicted_flat = flatten_and_align_sequences(all_actual, all_predicted)
    
    print("✅ supervised actual_flat shape before scaling:", actual_flat.shape)
    print("✅ supervised predicted_flat shape before scaling:", predicted_flat.shape)


    actual_flat = scaler.inverse_transform(actual_flat)
    predicted_flat = scaler.inverse_transform(predicted_flat)

    mse = mean_squared_error(actual_flat, predicted_flat, multioutput='raw_values')
    mae = mean_absolute_error(actual_flat, predicted_flat, multioutput='raw_values')
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_flat, predicted_flat, multioutput='raw_values')

    metrics = pd.DataFrame({
        "Feature": scaler.feature_names_in_,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2
    })
    metrics.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)

    for i, feature in enumerate(scaler.feature_names_in_):
        plt.figure(figsize=(10, 4))
        plt.plot(actual_flat[:, i], label="Actual", linestyle="dashed")
        plt.plot(predicted_flat[:, i], label="Predicted")
        plt.title(f"Actual vs Predicted ({feature}) - Full Trajectory")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{feature}_comparison.png"))
        plt.close()

    print(metrics)


def visualize_trajectory(model, test_loader, scaler_path='scaler.pkl', output_dir="visualizations_trajectory"):
    os.makedirs(output_dir, exist_ok=True)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    model.eval()
    all_actual, all_predicted = [], []
    with torch.no_grad():
        for sequence, _ in test_loader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sequence = sequence.to(device)
            y0 = sequence[:, 0, :]
            t = torch.linspace(0, 1, sequence.shape[1], device=device)
            
            prediction = model(y0, t).cpu().numpy()  # (T, B, F)
            prediction = np.transpose(prediction, (1, 0, 2))  # → (B, T, F)

            actual = sequence.cpu().numpy()  # already (B, T, F)
            all_actual.append(actual)
            all_predicted.append(prediction)

    actual_flat, predicted_flat = flatten_and_align_sequences(all_actual, all_predicted)
    
    print("✅ trajectory actual_flat shape before scaling:", actual_flat.shape)
    print("✅ trajectory predicted_flat shape before scaling:", predicted_flat.shape)


    actual_flat = scaler.inverse_transform(actual_flat)
    predicted_flat = scaler.inverse_transform(predicted_flat)

    # Plot trajectories
    for i, feature in enumerate(scaler.feature_names_in_):
        plt.figure(figsize=(10, 4))
        plt.plot(actual_flat[:, i], label="Actual", linestyle="dashed")
        plt.plot(predicted_flat[:, i], label="Predicted")
        plt.title(f"Trajectory: Actual vs Predicted ({feature})")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"trajectory_{feature}.png"))
        plt.close()

    # Calculate unsupervised metrics
    unsupervised_df = compute_unsupervised_metrics(actual_flat, predicted_flat, scaler.feature_names_in_)
    unsupervised_df.to_csv(os.path.join(output_dir, "unsupervised_metrics.csv"), index=False)

    # Generate additional unsupervised plots
    plot_distribution_alignment(actual_flat, predicted_flat, scaler.feature_names_in_, output_dir)
    plot_autocorrelation_comparison(actual_flat, predicted_flat, scaler.feature_names_in_, output_dir)

    print(unsupervised_df)


def compute_unsupervised_metrics(actual, predicted, feature_names):
    results = []
    for i, feature in enumerate(feature_names):
        wasserstein = wasserstein_distance(actual[:, i], predicted[:, i])
        results.append({
            "Feature": feature,
            "Wasserstein Distance": wasserstein
        })
    return pd.DataFrame(results)


def plot_distribution_alignment(actual, predicted, feature_names, output_dir="distribution_alignment"):
    os.makedirs(output_dir, exist_ok=True)
    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(8, 4))
        plt.hist(actual[:, i], bins=50, alpha=0.5, label="Actual", density=True)
        plt.hist(predicted[:, i], bins=50, alpha=0.5, label="Predicted", density=True)
        plt.title(f"Distribution Alignment - {feature}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"dist_align_{feature}.png"))
        plt.close()


def plot_autocorrelation_comparison(actual, predicted, feature_names, output_dir="autocorrelation"):
    os.makedirs(output_dir, exist_ok=True)
    for i, feature in enumerate(feature_names):
        acf_actual = acf(actual[:, i], fft=True, nlags=40)
        acf_pred = acf(predicted[:, i], fft=True, nlags=40)

        plt.figure(figsize=(8, 4))
        plt.plot(acf_actual, label="Actual ACF", linestyle="dashed")
        plt.plot(acf_pred, label="Predicted ACF")
        plt.title(f"Autocorrelation - {feature}")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"acf_{feature}.png"))
        plt.close()
