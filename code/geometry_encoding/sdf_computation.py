import numpy as np
from scipy.spatial import KDTree


class SignedDistanceFunction:
    """Approximate signed distance field for an ordered 2D polygon boundary.

    Boundary vertices must be ordered clockwise or counter-clockwise.
    Magnitudes are measured to the nearest sampled boundary point, so accuracy
    depends on boundary sampling density.
    """

    def __init__(self, grid_resolution=100):
        if not isinstance(grid_resolution, int) or grid_resolution < 2:
            raise ValueError("grid_resolution must be an integer >= 2")
        self.grid_resolution = grid_resolution
        self.sdf = None
        self.gradient = None

    @staticmethod
    def _inside_polygon(points, vertices):
        x = points[:, 0]
        y = points[:, 1]
        inside = np.zeros(points.shape[0], dtype=bool)
        previous = len(vertices) - 1

        for current in range(len(vertices)):
            x_current, y_current = vertices[current]
            x_previous, y_previous = vertices[previous]
            crosses = (y_current > y) != (y_previous > y)
            denominator = y_previous - y_current
            if denominator != 0:
                intersection_x = (
                    (x_previous - x_current) * (y - y_current) / denominator
                    + x_current
                )
                inside ^= crosses & (x < intersection_x)
            previous = current

        return inside

    def compute_sdf(self, boundary_points, domain_bounds):
        """Compute a signed field on a regular grid.

        Args:
            boundary_points: Ordered polygon vertices with shape (n, 2).
            domain_bounds: (x_min, x_max, y_min, y_max).
        """
        boundary_points = np.asarray(boundary_points, dtype=np.float64)
        if boundary_points.ndim != 2 or boundary_points.shape[1] != 2 or len(boundary_points) < 3:
            raise ValueError("boundary_points must contain at least three 2D polygon vertices")
        if not np.all(np.isfinite(boundary_points)):
            raise ValueError("boundary_points must contain only finite values")
        if len(domain_bounds) != 4:
            raise ValueError("domain_bounds must be (x_min, x_max, y_min, y_max)")

        x_min, x_max, y_min, y_max = map(float, domain_bounds)
        if not (x_min < x_max and y_min < y_max):
            raise ValueError("domain bounds must have positive width and height")

        x = np.linspace(x_min, x_max, self.grid_resolution)
        y = np.linspace(y_min, y_max, self.grid_resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

        tree = KDTree(boundary_points)
        distances, _ = tree.query(grid_points)
        inside = self._inside_polygon(grid_points, boundary_points)
        signed_distances = np.where(inside, -distances, distances)

        self.sdf = signed_distances.reshape(self.grid_resolution, self.grid_resolution)
        self.gradient = np.gradient(self.sdf)
        return self.sdf, self.gradient
