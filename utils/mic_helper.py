"""
This module provides core utility functions essential for post-processing and
analysis within molecular dynamics simulations, specifically those involving
Periodic Boundary Conditions (PBC).

Main Functions:
- mic_vector(): Calculate MIC displacement vector.
- mic_distance(): Calculate MIC scalar distance.
- wrap_points(): Wrap coordinates into the primary box.

Dependencies:
- numpy
- numba (for @njit)
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