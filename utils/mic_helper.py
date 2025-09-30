"""
Utility functions for molecular simulations, including Minimum Image Convention (MIC) calculations.
"""


import numpy as np
from numba import njit

# --- Minimum Image Convention (vector) ---
@njit(cache=True, fastmath=True)
def mic_vector(dx, box_dimensions):
    """Return minimum-image displacement for vector dx under PBC."""
    k = np.rint(dx / box_dimensions)
    return dx - k * box_dimensions

@njit(cache=True, fastmath=True)
def mic_distance(a, b, box_dimensions):
    """Return MIC distance between two 3D points a,b."""
    return np.linalg.norm(mic_vector(b - a, box_dimensions))

# Helper function to wrap points into the primary box
@njit(cache=True, fastmath=True)
def wrap_points(points, box_dimensions):
    """
    Wrap points into the primary simulation box using periodic boundary conditions.
    """
    return points % box_dimensions