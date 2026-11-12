import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from geometry_encoding.sdf_computation import SignedDistanceFunction


def test_sdf_is_negative_inside_and_positive_outside():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    field, gradient = SignedDistanceFunction(grid_resolution=3).compute_sdf(
        square, (-1, 2, -1, 2)
    )

    assert field.shape == (3, 3)
    assert field[1, 1] < 0
    assert field[1, 0] > 0
    assert len(gradient) == 2
    assert all(component.shape == field.shape for component in gradient)
    assert all(np.all(np.isfinite(component)) for component in gradient)


def test_sdf_rejects_invalid_polygon():
    with pytest.raises(ValueError, match="three 2D"):
        SignedDistanceFunction().compute_sdf([[0, 0], [1, 1]], (0, 1, 0, 1))
