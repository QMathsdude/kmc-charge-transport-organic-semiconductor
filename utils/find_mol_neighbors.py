"""
Hello
"""

import multiprocessing as mp
from functools import partial

import numpy as np
import trimesh
from scipy.spatial import KDTree
from tqdm import tqdm

from . import mic_helper as mh


# --------- USER MANUALLY DEFINE E-CLUMP CLUSTER ---------

# Helper function to extend user input
def extend_input(user_input):
    if '-' in user_input:
        start, end = user_input.split('-')
        return [*range(int(start), int(end) + 1)]
    else:
        return [int(user_input)]
    
# User selects atoms within a single molecule 
def user_indices_prompt(num_atoms_per_mol, total_num_mols):
    # user chooses atoms within molecule
    print("Insert atom id for one molecule.")
    print("Use the following format: '20-30; 30; 20; 40-60' (for range use '-', for multi-input split using ';' ")
    # user_input = input("Enter atom id:")
    user_input = "32-53"
    
    print('-' * 40)
    print(f"You entered: [{user_input}]")

    # extract atom indices that user selected
    try: 
        indices = [extend_input(i) for i in user_input.split(";")]
        indices = np.unique(list(chain.from_iterable(indices)))
        print(f"Selected atom_id(s): {indices}") # Verify selection
    except: 
        print("Invalid input format. Please use the specified format.")
        
    # filters all molecules based on user selected atoms
    indices_all = [indices]
    for i in range(1, total_num_mols):
        indices_all.append(indices + i * num_atoms_per_mol)
    
    return np.concatenate(indices_all) # flatten the list


# --------- E-CENTROIDS ---------

# parameters
# num_res = df_gro.index.levels[0][-1]

def calculate_e_centroids(df_e_clumps, num_res, box_dimensions):
    """
    Calculate centroids of electron clump clusters for each molecule under PBC.

    Parameters
    ----------
    df_e_clumps : pd.DataFrame
        DataFrame containing selected atoms for electron clump clusters, indexed by (res_id, atom_name).
    num_res : int
        Number of molecules (residues).
    box_dimensions : np.ndarray
        Simulation box dimensions (nm).

    Returns
    -------
    df_e_centroids : pd.DataFrame
        DataFrame of shape (num_res, 3) with centroid coordinates for each molecule.
    """
    
    e_centroids = np.zeros((num_res, 3)) # chunking

    for res_id in range(1, num_res + 1):
        # selects residue (molecule) one-by-one
        res_atoms = df_e_clumps.loc[(res_id,slice(None)),['x','y','z']].to_numpy()
        ref_atom = res_atoms[0]
        rel_displacement = np.zeros_like(res_atoms) # chunking
        
        # rel displacement of all atoms from ref_atom
        for i in range(res_atoms.shape[0]):
            rel_displacement[i] = mic_vector(res_atoms[i] - ref_atom, box_dimensions)
        
        # calculate centroid of the molecule from origin [0,0,0]
        mean_displacement = np.mean(rel_displacement, axis=0)
        centroids_unwrapped = ref_atom + mean_displacement
        e_centroids[res_id-1] = wrap_points(centroids_unwrapped, box_dimensions)

    # convert to dataframe
    df_e_centroids = pd.DataFrame(e_centroids, columns=['x', 'y', 'z'], index=range(1, num_res + 1))
    
    return df_e_centroids


# --------- BLOCKING FUNCTIONS ---------


# --------- MAIN FUNCTION ---------

def find_neighbors(mol_meshes, df_gro, box_dimensions):
    """
    Determine all unblocked neighbor pairs from a list of candidate molecule pairs.

    Parameters
    ----------
    centroids : dict[int, np.ndarray]
        Dictionary mapping molecule IDs to their centroid coordinates (in Å).
    radii : dict[int, float]
        Dictionary mapping molecule IDs to their effective radii (in Å).
    mol_meshes : dict[int, trimesh.Trimesh]
        Dictionary mapping molecule IDs to their trimesh mesh objects.
    neighbor_candidates : list[tuple[int, int]]
        List of candidate neighbor pairs (i, j) to test for blocking.

    Returns
    -------
    neighbor_pairs : list[tuple[int, int]]
        List of unblocked neighbor pairs (i, j).
    """
    # Parameters
    total_num_mols = df_gro.index.levels[0][-1]
    num_atoms_per_mol = df_gro.loc[(1,),:].shape[0]
    
    # Prompt user to select atoms to become e-clump clusters
    indices_all = user_indices_prompt(num_atoms_per_mol, total_num_mols)
    mask = df_gro["atom_id"].isin(indices_all)
    df_e_clumps = df_gro.loc[mask].copy()

    # Calculate e-centroids
    df_e_centroids = calculate_e_centroids(df_e_clumps, total_num_mols, box_dimensions)
    e_centroids = {res_id + 1: e_centroid for res_id, e_centroid in enumerate(df_e_centroids.values)}

    ids = list(centroids.keys())

    neighbor_pairs = []
    for pair in tqdm(neighbor_candidates, desc="Testing neighbor pairs"):
        i, j = pair
        if not blocked_by_any(i, j, e_centroids, centroids, radii, mol_meshes):
            neighbor_pairs.append((i, j))
            
    return neighbor_pairs
