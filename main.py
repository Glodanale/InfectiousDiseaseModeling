# project/
# ├── main.py                 → experiment loop entry point
# ├── dataloader.py            → data preprocessing, loader, scaling
# ├── models.py               → HybridODE, LSTMHybrid
# ├── train.py                → training + early stopping
# ├── visualize.py            → metrics, plots
# └── config.py               → param_grid + experiment combos

# main.py

import os
import json
import csv
import torch
import random
import numpy as np
from itertools import product

from config import param_grid
from dataloader import SEIRDDataLoader, get_split_strategy
from models import HybridODE
from train import train_model
from visualize import visualize_supervised, visualize_trajectory

EXPERIMENT_LOG_PATH = "experiment_log.csv"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_experiment(run_id, config, exp_folder):
    log_exists = os.path.exists(EXPERIMENT_LOG_PATH)
    with open(EXPERIMENT_LOG_PATH, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["run_id", "folder"] + list(config.keys()))
        if not log_exists:
            writer.writeheader()
        writer.writerow({"run_id": run_id, "folder": exp_folder, **config})


def run_experiment(config, run_id):
    set_seed(config.get("seed", 42))
    exp_folder = f"exp_{run_id:03d}"
    config["exp_folder"] = exp_folder  # ✅ Add exp_folder to config

    save_dir = os.path.join("results", exp_folder)
    os.makedirs(save_dir, exist_ok=True)

    print("\n==============================")
    print(f"Running {exp_folder}:")
    print(config)
    print("==============================")

    # Log experiment
    log_experiment(run_id, config, exp_folder)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load and scale data
    dataloader = SEIRDDataLoader(
        dataset_path='../../SEIR_CSV.csv',
        sequence_length=config['sequence_length'],
        scaler_path=os.path.join("results", exp_folder, "scaler.pkl")  # ✅ Use shared scaler path
    )
    data = dataloader.data

    # Split strategy
    split_result = get_split_strategy(
        data,
        strategy=config["split_strategy"],
        sequence_length=config["sequence_length"],
        batch_size=config["batch_size"],
        test_size=config.get("test_size", 0.2),
        val_size=config.get("val_size", 0.2),
        window_size=config.get("window_size", 200),
        horizon=config.get("horizon", 25),
        stride=config.get("stride", 25)
    )

    # Handle walkforward/rolling vs standard
    if isinstance(split_result, list):
        for fold_idx, (train_loader, val_loader, test_loader) in enumerate(split_result):
            print(f"→ Fold {fold_idx + 1}/{len(split_result)}")
            fold_folder = os.path.join(exp_folder, f"fold{fold_idx+1}")
            train_single_model(
                config,
                dataloader,
                train_loader,
                val_loader,
                folder_name=fold_folder,
                test_loader=test_loader
            )
    else:
        train_loader, val_loader, test_loader = split_result
        train_single_model(config, dataloader, train_loader, val_loader, exp_folder, test_loader)


def train_single_model(config, dataloader, train_loader, val_loader, folder_name, test_loader=None):
    save_dir = os.path.join("results", folder_name)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridODE(
        input_dim=dataloader.num_features,
        hidden_dim=64,
        num_layers=2,
        output_dim=dataloader.num_features,
        solver=config["solver"],
        sensitivity=config["sensitivity"]
    ).to(device)

    trained_model, test_loader = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config={
            "learning_rate": config["learning_rate"],
            "epochs": config["epochs"],
            "save_dir": save_dir
        }
    )

    # Visualization
    if test_loader:
        print("Trigger Visualization")
        visualize_supervised(
            trained_model,
            test_loader,
            scaler_path=os.path.join("results", config["exp_folder"], "scaler.pkl"),  # ✅ Shared scaler
            output_dir=save_dir
        )

        visualize_trajectory(
            trained_model,
            test_loader,
            scaler_path=os.path.join("results", config["exp_folder"], "scaler.pkl"),
            output_dir=os.path.join(save_dir, "trajectory_plots")
        )


if __name__ == "__main__":
    print(f"🚀 Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

    combinations = list(product(*param_grid.values()))
    for run_id, combo in enumerate(combinations):
        config = dict(zip(param_grid.keys(), combo))
        run_experiment(config, run_id)