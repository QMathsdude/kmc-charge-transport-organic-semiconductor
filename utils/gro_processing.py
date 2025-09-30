"""
This module provides functions for reading and processing GROMACS .gro files,
which are commonly used in molecular dynamics simulations. The module handles
file parsing, coordinate extraction, and advanced data processing including
oxygen midpoint calculations for water molecules.

File Format Support:
- Standard GROMACS .gro format with fixed-width columns

Main Functions:
- read_gro(): Read .gro file and extract atomic data with metadata

Dependencies:
- pandas
- numpy 
"""


import pandas as pd
import numpy as np

all = ['read_gro']

# -----------------------------

# Function to extract data from any type of .gro file
def read_gro(file, multiply=1, positions=True, velocities=False):
    """
    Reads a GROMACS .gro file and extracts atomic coordinates.

    Args:
        file (str): The path to the .gro file.
        multiply (int, optional): Factor to change units. Defaults to 1.

    Returns:
        pd.DataFrame: A DataFrame containing all data from .gro file.
    """
    # Reading Title, number of atoms and box_dimensions
    with open(file, "rb") as f:
        title = f.readline().decode().strip()  # First line of .gro file
        num_atoms = f.readline().decode().strip()  # Second line of .gro file
        num_atoms = np.int64(num_atoms) 

        # Last line of file
        f.seek(-2, 2)
        while f.read(1) != b"\n":
            f.seek(-2, 1)
        box_dimensions = f.readline().decode().strip().split() # Last line (with text) of .gro file
        box_dimensions = np.array(list(map(float, box_dimensions)), dtype=np.float64)
        
    # Setting column specifications
    gro_colspecs = [
        (0, 5),     # Residue number    (5 positions, int)
        (5, 10),    # Residue name      (5 characters)
        (10, 15),   # Atom name         (5 characters)
        (15, 20),   # Atom ID           (5 positions, int)
        (20, 28),   # x coordinate      (8 positions with 3 decimal places, float)
        (28, 36),   # y coordinate      (nm)
        (36, 44),   # z coordinate
        (44, 52),   # Vx (float)        (8 positions with 4 decimal places, float)
        (52, 60),   # Vy (float)        (nm/ps or km/s)
        (60, 68),   # Vz (float)
    ]

    # Setting names of columns
    gro_names = ["res_id", "res_name", "atom_name", "atom_id", "x", "y", "z", "Vx", "Vy", "Vz"]

    # Reading data
    data = pd.read_fwf(
        file,
        colspecs=gro_colspecs,
        names=gro_names,
        skiprows=2,
        skipfooter=1,
    )
    
    # Multiply the units
    data[['x', 'y', 'z']] *= multiply
    data[['Vx', 'Vy', 'Vz']] *= multiply
    box_dimensions *= multiply

    # Problem: After atom id 99999, it reverts back to 0. The following solves this:
    overflow_count = num_atoms // 100_000
    # Case 1: < 100_000 molecules
    if overflow_count == 0: pass
    # Case 2: < 200_000 molecules
    else:
        offset_mask = (np.arange(num_atoms) + 1) // 100_000
        data.loc[:,'atom_id'] += offset_mask * 100_000

    # Create multi-index using residue ID and atom name
    data = data.set_index(['res_id', 'atom_name'])
    
    # Choose what data to display
    if positions == False:
        data = data.drop(columns=['x', 'y', 'z'])
    if velocities == False:
        data = data.drop(columns=['Vx', 'Vy', 'Vz'])

    return data, title, num_atoms, box_dimensions

# -----------------------------

# Check from CLI
if __name__ == "__main__":
    print("Running gro_processing.py as a script")