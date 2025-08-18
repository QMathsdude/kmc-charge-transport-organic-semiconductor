import numpy as np 
import pandas as pd
from numba import njit, prange

# MIC and PBC helper function
@njit('float64(float64, float64, float64)', cache=True)
def minimum_image_jit(dx, box_length, reciprocal_half_box):
    k = int(dx * reciprocal_half_box)
    return dx - k * box_length

@njit('float64(float64, float64, float64, float64)', cache=True)
def midpoint_pbc_jit(p1, p2, box_length, reciprocal_half_box):
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
def compute_midpoints_df_jit(filtered_data, box_length):
    """
    Computes the midpoints between O1 and O2 atoms for each residue in a multi-indexed 
    DataFrame using fast Numba-accelerated routines. Consists for 3 helper Numba functions: 
    compute_midpoints_jit_core, midpoint_pbc_jit and minimum_image_jit.
    """
    # Perform all Pandas operations outside the Numba function
    filtered_index = filtered_data.index.get_level_values(0).unique()

    filtered_o1 = filtered_data.loc[(filtered_index, 'O1'), :]
    filtered_o2 = filtered_data.loc[(filtered_index, 'O2'), :]

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
    df_midpoints.insert(0, 'res_id', filtered_index)
    
    # Reset the index to a default integer index (0, 1, 2, ...)
    df_midpoints.reset_index(drop=True, inplace=True)
    
    return df_midpoints

# -----------------------------

# Check from CLI
if __name__ == "__main__":
    print("Running mod1.py as a script")