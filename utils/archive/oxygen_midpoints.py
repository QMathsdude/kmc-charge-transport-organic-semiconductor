"""
This module provides high-performance functions for calculating geometric midpoints
between oxygen atoms in molecular dynamics systems with periodic boundary conditions.
The functions are optimized with Numba JIT compilation for significant performance
gains on large datasets.

Core Functions:
- minimum_image_jit(): Minimum image convention helper
- midpoint_pbc_jit(): PBC-aware midpoint calculation between two points
- compute_midpoints_jit_core(): Parallel core for bulk midpoint calculations
- compute_midpoints_df_jit(): Main interface for DataFrame processing

Performance Notes:
- Numba JIT compilation provides 10-100x speedup over pure Python
- Parallel execution for multi-core systems
- Memory-efficient array operations

Dependencies:
- numpy >= 1.21.0
- pandas >= 1.3.0  
- numba >= 0.55.0

Note: Input DataFrames must contain columns: 'res_id', 'atom_name', 'x', 'y', 'z'
"""

import numpy as np 
import pandas as pd
from numba import njit, prange

__all__ = ['minimum_image_jit', 'midpoint_pbc_jit', 'compute_midpoints_jit_core', 'compute_midpoints_df_jit']

# -----------------------------

# MIC and PBC helper function
@njit('float64(float64, float64, float64)', cache=True)
def minimum_image_jit(dx, box_length, reciprocal_half_box):
    """
    Compute the minimum image of a displacement vector.
    Args:
        dx (np.float64)                  : The displacement vector component.
        box_length (np.float64)          : The length of the simulation box.
        reciprocal_half_box (np.float64) : The reciprocal of half the box length.
    Returns:
        np.float64: The minimum image of the displacement vector.
    """
    k = int(dx * reciprocal_half_box)
    return dx - k * box_length

@njit('float64(float64, float64, float64, float64)', cache=True)
def midpoint_pbc_jit(p1, p2, box_length, reciprocal_half_box):
    """
    Compute the periodic boundary condition (PBC) midpoint between two points.
    
    Args:
        p1 (np.float64)                  : The first point.
        p2 (np.float64)                  : The second point.
        box_length (np.float64)          : The length of the simulation box.
        reciprocal_half_box (np.float64) : The reciprocal of half the box length.
        
    Returns:
        np.float64: The periodic boundary condition (PBC) midpoint.
    """
    dx = p2 - p1
    dx_mic = minimum_image_jit(dx, box_length, reciprocal_half_box)
    midpoint = (p1 + 0.5 * dx_mic) % box_length
    return midpoint

# -----------------------------

# Core Numba-compiled function to help compute midpoints
@njit(parallel=True, cache=True) # nopython=True is the default
def compute_midpoints_jit_core(o1_x, o1_y, o1_z, o2_x, o2_y, o2_z, box_length):
    """
    Compute midpoints for all residues using fast parallel execution.
    Requires data extracted from Pandas DataFrame first.

    Args:
        o1_x (np.ndarray)       : The x-coordinates of O1 atoms.
        o1_y (np.ndarray)       : The y-coordinates of O1 atoms.
        o1_z (np.ndarray)       : The z-coordinates of O1 atoms.
        o2_x (np.ndarray)       : The x-coordinates of O2 atoms.
        o2_y (np.ndarray)       : The y-coordinates of O2 atoms.
        o2_z (np.ndarray)       : The z-coordinates of O2 atoms.
        box_length (np.ndarray) : The length of the box (unit cell).

    Returns:
        np.ndarray: The computed midpoints of the two oxygen atoms.
    """
    num_residues = o1_x.shape[0]
    midpoints = np.empty((num_residues, 3), dtype=np.float64) # Preallocate array for midpoints
    reciprocal_half_box = 1.0 / (0.5 * box_length)
    
    for i in prange(num_residues):
        midpoints[i, 0] = midpoint_pbc_jit(o1_x[i], o2_x[i], box_length, reciprocal_half_box)
        midpoints[i, 1] = midpoint_pbc_jit(o1_y[i], o2_y[i], box_length, reciprocal_half_box)
        midpoints[i, 2] = midpoint_pbc_jit(o1_z[i], o2_z[i], box_length, reciprocal_half_box)
        
    return midpoints

# -----------------------------

# Main function to handle Pandas data
def compute_midpoints_df_jit(data, box_length):
    """
    Computes the midpoints between O1 and O2 atoms for each residue in a multi-indexed DataFrame using fast Numba-accelerated routines.
    Is made from 3 helper functions: compute_midpoints_jit_core, midpoint_pbc_jit and minimum_image_jit.

    Args:
        data (pd.DataFrame)     : The input DataFrame containing atomic coordinates.
        box_length (np.float64) : The length of the simulation box.

    Returns:
        pd.DataFrame: A DataFrame containing the midpoints between O1 and O2 atoms.
    """
    # Perform all Pandas operations outside the Numba function
    index = data['res_id'].unique().tolist()

    filtered_o1 = data.loc[data['atom_name'] == 'O1']
    filtered_o2 = data.loc[data['atom_name'] == 'O2']

    # Extract coordinates for O1 and O2 into NumPy array 
    o1_x = filtered_o1.loc[:, 'x'].to_numpy(dtype=np.float64)
    o1_y = filtered_o1.loc[:, 'y'].to_numpy(dtype=np.float64)
    o1_z = filtered_o1.loc[:, 'z'].to_numpy(dtype=np.float64)
    
    o2_x = filtered_o2.loc[:, 'x'].to_numpy(dtype=np.float64)
    o2_y = filtered_o2.loc[:, 'y'].to_numpy(dtype=np.float64)
    o2_z = filtered_o2.loc[:, 'z'].to_numpy(dtype=np.float64)

    # Call the core Numba function with NumPy arrays
    midpoints_array = compute_midpoints_jit_core(o1_x, o1_y, o1_z, o2_x, o2_y, o2_z, box_length)
    
    # Create final DataFrame from the results
    df_midpoints = pd.DataFrame(
        data=midpoints_array,
        columns=['mid_x', 'mid_y', 'mid_z']
    )
    
    # Insert res_id as the first column
    df_midpoints.insert(0, 'res_id', index)
    
    # Reset the index to a default integer index (0, 1, 2, ...)
    df_midpoints.reset_index(drop=True, inplace=True)
    
    return df_midpoints

# -----------------------------

# Check from CLI
if __name__ == "__main__":
    print("Running oxygen_midpoints.py as a script")