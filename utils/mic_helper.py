"""
This module provides functions for handling the Minimum Image Convention (MIC)
of positions and vectors under Periodic Boundary Conditions (PBC).

Main Functions:
- mic_vector(): Calculate MIC displacement vector.
- mic_distance(): Calculate MIC scalar distance.
- wrap_points(): Wrap coordinates into the primary box.
"""
import numpy as np
from numba import njit

# --- Minimum Image Convention ---

@njit(cache=True, fastmath=True)
def mic_vector(dx, box_dimensions):
    """
    Return minimum-image displacement for vector dx under PBC.
    """
    k = np.rint(dx / box_dimensions)
    return dx - k * box_dimensions

@njit(cache=True, fastmath=True)
def mic_distance(a, b, box_dimensions):
    """
    Return MIC distance between two 3D points a,b.
    """
    return np.linalg.norm(mic_vector(b - a, box_dimensions))

@njit(cache=True, fastmath=True)
def wrap_points(points, box_dimensions):
    """
    Wrap points into the primary simulation box using periodic boundary conditions.
    """
    return points % box_dimensions

# -----------------------------

# Export wildcard
__all__ = ['mic_vector', 'mic_distance', 'wrap_points']

# Check from CLI
if __name__ == "__main__":
    print("Running mic_helper.py as a script")