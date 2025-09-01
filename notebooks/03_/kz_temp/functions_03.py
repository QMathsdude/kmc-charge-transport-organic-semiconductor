# imports
import pandas as pd


def read_gro(file="./data/npt-HK4.gro"):
    """
    Reads the npt-HK4.gro and returns the following:

    data(pd.Dataframe): A dataframe that contains Residue ID(res_id) and Name(res_name), Atom Name (atom_name) and ID(atom_id), the atom coordinates(x, y, z) and velocity components(Vx, Vy, Vz)
    title(str): The title of the .gro file
    num_atoms(str): The number of atoms
    box_dimensions(list): The simulation box dimensions in a list of [x,y,z] 

    """
    # setting column specifications
    colspecs = [
        (0, 5),
        (5, 8),
        (8, 15),
        (15, 20),
        (20, 28),
        (28, 36),
        (36, 44),
        (44, 52),
        (52, 60),
        (60, 68),
    ]

    # setting names of columns
    names = ["res_id", "res_name", "atom_name", "atom_id", "x", "y", "z", "Vx", "Vy", "Vz"]

    # reading data
    data = pd.read_fwf(
        file,
        colspecs=colspecs,
        names=names,
        skiprows=2,
        skipfooter=1,
    )

    # Problem: after atom id 99999, it reverts back to 0. The following solves this:
    for i in data.index:
        if i >= 99999:
            data.at[i, "atom_id"] += 100000

    # reading Title, number of atoms and box_dimensions
    with open(file, "rb") as f:
        title = f.readline().decode().strip()  # First line of .gro file
        num_atoms = f.readline().decode().strip()  # Second line of .gro file

        # Last line of file
        f.seek(-2, 2)
        while f.read(1) != b"\n":
            f.seek(-2, 1)
        box_dimensions = f.readline().decode().strip().split()
        box_dimensions = list(map(float, box_dimensions))

    return (data, title, num_atoms, box_dimensions)
