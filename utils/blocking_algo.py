"""
This module provides algorithms for identifying neighboring molecules and blocked paths
in molecular simulations, with support for periodic boundary conditions (PBC).

Sections:
----------
1. Electron Clump Centroid Generation
    - Main Function:
      - calculate_e_centroids(): Calculates centroids of user-defined electron clump clusters for each molecule under PBC.
    - Helper Functions:
      - extend_input(): Expands user input ranges into lists of atom indices.
      - user_indices_prompt(): Prompts user to select atom indices for electron clump clusters.
      
2. Centroid and Radius Calculation
    - Main Function:
      - compute_centroids_and_radii_pbc(): Computes PBC-aware centroids and effective radii for molecule meshes.
      
3. Neighbor Candidate Identification
    - Main Function:
      - get_neighbor_candidates(): Finds k nearest neighbor candidates for each molecule using KDTree and PBC.
      
4. Blocking Algorithm
    - Main Function:
      - find_neighbors(): Main pipeline to compute neighbor_pairs, centroids, radii, neighbor candidates, and optionally export results.
    - Helper Functions:
      - blocked_by_any(): Checks if the direct path between two molecules is blocked by any other molecule using sphere rejection and ray-mesh intersection.
      - true_neighbors(): Determines unblocked neighbor pairs from candidate pairs using geometric and mesh intersection tests.

Dependencies Notes:
--------------------
- time (for immersive user input)
- chain (to flatten lists)
"""

import multiprocessing as mp
import os
import time
from itertools import chain

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import KDTree
from tqdm import tqdm

from . import mic_helper as mh

# ------------------------------
# GENERATE E-CENTROIDS
# ------------------------------

# --------- HELPER FUNCTIONS ---------

# extends user input into list of integers
def extend_input(user_input):
    '''
    Removes '-' from string and returns list of integers.
    e.g. '20-30' -> [20,21,22,23,24,25,26,27,28,29,30]
         '30'    -> [30]
    '''
    if '-' in user_input:
        start, end = user_input.split('-')
        return [*range(int(start), int(end) + 1)]
    else:
        return [int(user_input)]
    
# User selects atoms within a single molecule 
def user_indices_prompt(res_ids, num_atoms):
    '''
    Prompts user to select specific atoms for a single molecule, 
    then returns those atoms ids across ALL molecules.
    '''
    # user chooses atoms within molecule
    print("Insert atom id for one molecule."); time.sleep(1)
    print("Use the following format: '20-30; 30; 20; 40-60' (for range use '-', for multi-input split using ';' ")
    # user_input = input("Enter atom id:"); time.sleep(1)
    user_input = "32-53"; time.sleep(1) # for testing purposes
    print('-' * 40)
    print(f"You entered: [{user_input}]")
    print('-' * 40); time.sleep(1)

    try: # extract atom indices that user selected
        indices = [extend_input(i) for i in user_input.split(";")]
        indices = np.unique(list(chain.from_iterable(indices)))
        print(f"Selected atom_id(s): {indices}") # Verify selection
    except: 
        print("Invalid input format. Please use the specified format.")
        
    # filters all molecules ID based on user selected atoms
    e_indices = np.concatenate([indices + (res_id - 1) * num_atoms for res_id in res_ids]) # flattens [[],[]]
    
    return e_indices

# --------- MAIN FUNCTION ---------

# Calculates centroids of electron clump clusters (user defined) using df_gro
def calculate_e_centroids(mol_meshes, df_gro, box_dimensions):
    """
    Calculate centroids of electron clump clusters for each molecule under PBC.

    Parameters
    ----------
    mol_meshes : dict[int, trimesh.Trimesh]
        Dictionary of molecule meshes.
    df_gro : pd.DataFrame
        Multi-index DataFrame containing atom positions with indices (res_id, atom_id).
    box_dimensions : array-like, shape (3,) or (3,3)
        Simulation box dimensions (in Å).  

    Returns
    -------
    e_centroids : dict[int, np.ndarray]
        Dictionary of electron clump centroids for each molecule, with keys as molecule IDs
        and values as numpy arrays of shape (3,) representing the (x, y, z) coordinates.
    """
    # This assumes all molecules are similar (each have exactly identical atoms)
    res_ids = list(mol_meshes.keys()) # list of molecule ids
    num_atoms = df_gro.loc[(1,),:].shape[0] # checks number of atoms in first molecule
    
    # Create new df consisting only of electron clump cluster atoms
    e_indices = user_indices_prompt(res_ids, num_atoms) 
    mask = df_gro["atom_id"].isin(e_indices)
    df_e_clumps = df_gro.loc[mask].copy()
    
    # Calculating e-centroids
    e_centroids = {}
    for res_id in res_ids:
        # selects residue (molecule) one-by-one
        res_atoms = df_e_clumps.loc[(res_id,slice(None)),['x','y','z']].to_numpy()
        ref_atom = res_atoms[0]
        rel_displacement = np.zeros_like(res_atoms) # chunking
        
        # rel displacement of all atoms from ref_atom
        for i in range(res_atoms.shape[0]):
            rel_displacement[i] = mh.mic_vector(res_atoms[i] - ref_atom, box_dimensions)
        
        # calculate centroid of the molecule from origin [0,0,0]
        mean_displacement = np.mean(rel_displacement, axis=0)
        centroids_unwrapped = ref_atom + mean_displacement
        e_centroids[res_id] = mh.wrap_points(centroids_unwrapped, box_dimensions)
    
    return e_centroids

# ------------------------------
# GENERATE CENTROIDS & RADII
# ------------------------------

# Calculate the true centroids and effective radii using vertices of molecule meshes
def compute_centroids_and_radii_pbc(mol_meshes, box_dimensions):
    """
    Compute periodic-boundary-condition (PBC) aware centroids and radii for a set of molecules.

    For each molecule mesh, this function:
      1. Selects a reference vertex (atom) as the origin.
      2. Unwraps all other vertices relative to this reference using the minimum-image convention (MIC),
         so that all atoms are locally unwrapped and contiguous in space.
      3. Computes the centroid (geometric center) of the unwrapped coordinates.
      4. Calculates the maximum MIC distance from the centroid to any vertex, defining the molecule's effective radius.

    Parameters
    ----------
    mol_meshes : dict[int, trimesh.Trimesh]
        Dictionary mapping molecule IDs to their corresponding trimesh mesh objects.
    box_dimensions : array-like, shape (3,) or (3,3)
        Simulation box dimensions (in Å). Should be a 3-element array for orthorhombic boxes.

    Returns
    -------
    centroids : dict[int, np.ndarray]
        Dictionary mapping molecule IDs to their centroid coordinates (in Å), unwrapped in PBC.
    radii : dict[int, float]
        Dictionary mapping molecule IDs to their effective radii (in Å), defined as the maximum MIC distance
        from the centroid to any vertex in the molecule.
    """
    centroids, radii = {}, {}
    # box_dimensions = np.array(box_dimensions, dtype=float)

    for mol_id, mesh in mol_meshes.items():
        verts = mesh.vertices
        ref = verts[0]  # reference atom
        disp = mh.mic_vector(verts - ref, box_dimensions)
        unwrapped = ref + disp

        # centroid in unwrapped space
        center = unwrapped.mean(axis=0)

        # MIC distances from centroid to each vertex
        disp_center = mh.mic_vector(verts - center, box_dimensions)
        radius = np.linalg.norm(disp_center, axis=1).max()

        centroids[mol_id] = center
        radii[mol_id] = radius

    return centroids, radii

# ------------------------------
# NEIGHBOR CANDIDATES
# ------------------------------

def get_neighbor_candidates(box, centroids, k=10):
    """
    Find the k nearest neighbors of each molecule, based on centroid distance.
    
    Parameters
    ----------
    box : array-like, shape (3,) or (3,3)
        Simulation box dimensions (in Å). Should be a 3-element array for orthorhombic boxes.
    centroids : dict[int, np.ndarray]
        Dictionary mapping molecule IDs to their centroid coordinates (in Å).
    k : int, optional
        Number of nearest neighbors to return per molecule. Default is 10.

    Returns
    -------
    dict[int, list[tuple[int,float]]]
        Mapping mol_id -> list of (neighbor_id, distance).
    """

    ids = list(centroids.keys())
    coords = np.vstack([centroids[i] for i in ids])
    coords_wrapped = mh.wrap_points(coords, box)
    kd = KDTree(coords_wrapped, boxsize=box)

    neighbors = {}
    for idx, mol_id in enumerate(ids):
        dists, idxs = kd.query(coords[idx], k=k+1)
        dists, idxs = dists[1:], idxs[1:]
        neighbors[mol_id] = [(ids[j], float(d)) for j, d in zip(idxs, dists)]

    # Remove duplicate pairs (A,B) and (B,A), keep only (A,B) where A < B
    neighbor_candidates = [(min(key, t[0]), max(key, t[0]))
                           for key, value in neighbors.items() for t in value if key < t[0]]
    neighbor_candidates = list(set(neighbor_candidates))
    neighbor_candidates.sort()
        
    return neighbor_candidates


# ------------------------------
# BLOCKING ALGORITHM
# ------------------------------

# --------- HELPER FUNCTIONS ---------

def blocked_by_any(i, j, e_centroids, centroids, radii, mol_meshes, neighbor_candidates):
    """
    Determine if the direct path between two molecule e_centroids is obstructed by any other molecule.

    For a given pair of molecules (i, j), this function checks whether the straight line
    connecting their e_centroids is intersected ("blocked") by any other molecule in the system.
    The check is performed in two steps:
      1. Fast sphere rejection: For each candidate blocking molecule, if its centroid is not
         within its effective radius of the line segment, it is skipped.
      2. Ray-mesh intersection: If the sphere check passes, a ray-mesh intersection test is
         performed to determine if the mesh of the candidate molecule blocks the path.

    Periodic boundary conditions (PBC) are handled using the minimum-image convention.

    Parameters
    ----------
    i, j : int
        IDs of the two molecules to test for a direct connection.
    e_centroids : dict[int, np.ndarray]
        Dictionary mapping molecule IDs to their centroid coordinates (in Å).
    radii : dict[int, float]
        Dictionary mapping molecule IDs to their effective radii (in Å).
    mol_meshes : dict[int, trimesh.Trimesh]
        Dictionary mapping molecule IDs to their trimesh mesh objects.

    Returns
    -------
    blocked : bool
        True if the path between i and j is blocked by any other molecule, False otherwise.
    """
    # Map molecule IDs to their index in ids

    ci, cj = e_centroids[i], e_centroids[j]
    seg_vec = cj - ci
    seg_len = np.linalg.norm(seg_vec)
    if seg_len < 1e-6:
        return False
    direction = seg_vec / seg_len

    # Get candidate molecule IDs (not indices)
    cand_ids = [t[1] for t in neighbor_candidates if t[0] == i]

    for mol_k in cand_ids:
        if mol_k in (i, j):
            continue

        # Quick sphere reject
        ck = centroids[mol_k]
        v = cj - ci
        w = ck - ci
        proj = np.dot(w, v) / np.dot(v, v)
        proj = np.clip(proj, 0.0, 1.0)
        closest = ci + proj * v
        if np.linalg.norm(ck - closest) > radii[mol_k]:
            continue

        # Expensive ray test
        if mol_meshes[mol_k].ray.intersects_any(
            ray_origins=ci.reshape(1, 3),
            ray_directions=direction.reshape(1, 3)
        ):
            return True
        
    return False

def true_neighbors(e_centroids, centroids, radii, mol_meshes, neighbor_candidates):
    """
    Determine all unblocked neighbor pairs from a list of candidate molecule pairs.

    Parameters
    ----------
    e_centroids : dict[int, np.ndarray]
        Dictionary mapping molecule IDs to their electron clump centroid coordinates (in Å).
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

    ids = list(centroids.keys())

    neighbor_pairs = []
    for pair in tqdm(neighbor_candidates, desc="Testing neighbor pairs"):
        i, j = pair
        if not blocked_by_any(i, j, e_centroids, centroids, radii, mol_meshes, neighbor_candidates):
            neighbor_pairs.append((i, j))
            
    return neighbor_pairs

# --------- MAIN FUNCTION ---------

def find_neighbors(mol_meshes, df_gro, box_dimensions, path, k=10, export_csv=False):
    """
    Determine all unblocked neighbor pairs.

    Parameters
    ----------
    mol_meshes : dict[int, trimesh.Trimesh]
        Dictionary mapping molecule IDs to their trimesh mesh objects.
    df_gro : pd.DataFrame
        DataFrame containing molecular structure data.
    box_dimensions : array-like, shape (3,) or (3,3)
        Simulation box dimensions (in Å).
    k : int, optional
        Number of nearest neighbors to consider per molecule.
    export_csv : bool, optional
        Whether to export intermediate data (e_centroids, centroids, neighbor_candidates) as CSV files.

    Returns
    -------
    neighbor_pairs : list[tuple[int, int]]
        List of unblocked neighbor pairs (i, j).
    e_centroids : dict[int, np.ndarray]
        Dictionary of electron clump centroids for each molecule.
    centroids : dict[int, np.ndarray]
        Dictionary of centroids for each molecule.
    neighbor_candidates : list[tuple[int, int]]
        List of candidate neighbor pairs (i, j).
    """
    # 1. Get e-centroids
    e_centroids = calculate_e_centroids(mol_meshes, df_gro, box_dimensions)
    
    # 2. Get centroids and radii
    centroids, radii = compute_centroids_and_radii_pbc(mol_meshes, box_dimensions)
    
    # 3. Get neighbor candidates
    neighbor_candidates = get_neighbor_candidates(box_dimensions, centroids, k)
    
    # 4. Find true neighbors
    # neighbor_pairs = true_neighbors(e_centroids, centroids, radii, mol_meshes, neighbor_candidates)
    
    
    # Export as csv file
    if export_csv == True:
        print('-' * 40)
        name = os.path.basename(path).rsplit('.', 1)[0] # e.g., 'npt-HK4'
        
        df_e_centroids = pd.DataFrame(e_centroids, index=['x', 'y', 'z']).T
        df_e_centroids.to_csv(f'{name}_e_centroids.csv', index_label='res_id')
        print(f"Successfully exported {name}_e_centroids.csv")

        df_centroids = pd.DataFrame(centroids, index=['x', 'y', 'z']).T
        df_centroids.to_csv(f'{name}_centroids.csv', index_label='res_id')
        print(f"Successfully exported {name}_centroids.csv")

        df_neighbors_candidates = pd.DataFrame(neighbor_candidates, columns=['mol_id_1', 'mol_id_2'])
        df_neighbors_candidates.to_csv(f'{name}_neighbor_candidates.csv', index=False)
        print(f"Successfully exported {name}_neighbor_candidates.csv")

        # df_neighbor_pairs = pd.DataFrame(neighbor_pairs, columns=['mol_id_1', 'mol_id_2'])
        # df_neighbor_pairs.to_csv(f'{name}_neighbor_pairs.csv', index=False)
        # print(f"Exported {name}_neighbor_pairs.csv")

    return e_centroids, centroids, neighbor_candidates
    # return neighbor_pairs, e_centroids, centroids, neighbor_candidates