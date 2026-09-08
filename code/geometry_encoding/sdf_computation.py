import numpy as np
from scipy.spatial import KDTree

class SignedDistanceFunction:
    def __init__(self, grid_resolution=100):
        self.grid_resolution = grid_resolution
        self.sdf = None
        self.gradient = None
    
    def compute_sdf(self, boundary_points, domain_bounds):
        x_min, x_max, y_min, y_max = domain_bounds
        x = np.linspace(x_min, x_max, self.grid_resolution)
        y = np.linspace(y_min, y_max, self.grid_resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
        tree = KDTree(boundary_points)
        distances, _ = tree.query(grid_points)
        self.sdf = distances.reshape(self.grid_resolution, self.grid_resolution)
        self.gradient = np.gradient(self.sdf)
        return self.sdf, self.gradient