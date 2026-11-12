import numpy as np


def _as_matching_arrays(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError("pred and true must have identical shapes")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(true)):
        raise ValueError("pred and true must contain only finite values")
    return pred, true


def relative_l2_error(pred, true):
    """Return the relative L2 error as a percentage."""
    pred, true = _as_matching_arrays(pred, true)
    denominator = np.linalg.norm(true)
    if denominator == 0:
        return 0.0 if np.linalg.norm(pred) == 0 else float("inf")
    return float(np.linalg.norm(pred - true) / denominator * 100)


def compute_metrics(pred, true):
    """Compute relative L2 error, MAE, and RMSE."""
    pred, true = _as_matching_arrays(pred, true)
    l2_rel = relative_l2_error(pred, true)
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    return {
        "relative_l2_error": l2_rel,
        "mean_absolute_error": float(mae),
        "root_mean_square_error": float(rmse),
    }
