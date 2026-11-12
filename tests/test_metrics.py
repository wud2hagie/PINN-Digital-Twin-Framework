import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from evaluation.metrics import compute_metrics, relative_l2_error


def test_relative_l2_error_is_percentage():
    assert relative_l2_error([2.0, 3.0], [1.0, 3.0]) == pytest.approx(
        np.sqrt(1.0 / 10.0) * 100
    )


def test_zero_reference_is_handled_without_nan():
    assert relative_l2_error([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert np.isinf(relative_l2_error([1.0, 0.0], [0.0, 0.0]))


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        compute_metrics([1.0, 2.0], [1.0])


def test_metrics_reject_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        compute_metrics([np.nan], [1.0])


def test_metrics_reject_empty_arrays():
    with pytest.raises(ValueError, match="must not be empty"):
        compute_metrics([], [])
