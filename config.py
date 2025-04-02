# config.py

param_grid = {
    "split_strategy": ["walkforward"],
    "solver": ["tsit5", "adaptive_heun", "fehlberg2"],
    "sensitivity": ["interpolated_adjoint"],
    "learning_rate": [0.01, 0.025, 0.05],
    "batch_size": [50, 100],
    "window_size": [200],  # Used by walkforward/rolling
    "horizon": [50],       # Used by walkforward
    "stride": [25],        # Used by rolling
    "sequence_length": [200],
    "epochs": [150],
    "seed": [42]
}
