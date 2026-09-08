import numpy as np

def relative_l2_error(pred, true):
    return np.linalg.norm(pred - true) / np.linalg.norm(true) * 100

def compute_metrics(pred, true):
    l2_rel = relative_l2_error(pred, true)
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true)**2))
    return {
        'relative_l2_error': l2_rel,
        'mean_absolute_error': mae,
        'root_mean_square_error': rmse
    }