"""
This module provides functions to find centroids (even user defined centroids), 
algorithm which identifies unblocked neighbor pairs and visual verification of
blocking, all under periodic boundary conditions (PBC).

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
      - find_neighbor_pairs(): Compute neighbor_pairs, centroids, radii, neighbor candidates, and optionally export results.
    - Helper Functions:
      - blocked_by_any(): Checks if direct path between two molecules is blocked by using ray-mesh intersection.
      - true_neighbors(): Determines unblocked neighbor pairs from candidate pairs using geometric and mesh intersection tests.
      
5. Import Neighbor Pairs
    - Main Function:
        - import_csv(): Imports either neighbor_pairs, e_centroids, centroids and neighbor_candidates from CSV files.
        
6. Verify Blocking Molecules
    - Main Function:
        - view_molecule(): Visualizes molecular blocking.
    - Helper Functions:
        - map_symmetric_neighbors(): Maps neighbor pairs to a symmetric dictionary for easy lookup.
        - get_distinct_colors(): Generates distinct colors for visualization.
        - get_user_input_unpaired(): Handles user input for selecting unpaired neighbors to view.

Dependencies Notes:
--------------------
- ast (to safely evaluate user input)
- chain (to flatten lists)
- defaultdict (for easy dictionary of lists)
- get_cmap (for color maps)
- time (for immersive user input)
"""

import ast
import multiprocessing as mp
import os
import time
from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd
import trimesh
from matplotlib import colors as mcolors
from matplotlib.pyplot import get_cmap
from scipy.spatial import KDTree
from tqdm import tqdm

from . import mic_helper as mh

# ------------------------------
# 1. GENERATE E-CENTROIDS
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
        Dictionary where keys are molecule IDs (int) and values are
        the corresponding molecular meshes (trimesh.Trimesh objects).
    df_gro : pd.DataFrame
        DataFrame containing molecular data, multi-indexed by (res_id, atom_name).
    box_dimensions : np.ndarray
        Simulation box_dimensions (Å), shape (3,) for orthorhombic or (3,3) for triclinic

    Returns
    -------
    e_centroids : dict[int, np.ndarray]
        Dictionary where keys are molecule IDs (int) and values are
        the corresponding electron clump centroids (numpy arrays of shape (3,)).
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
# 2. GENERATE CENTROIDS & RADII
# ------------------------------

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
        Dictionary where keys are molecule IDs (int) and values are
        the corresponding molecular meshes (trimesh.Trimesh objects).
    box_dimensions : np.ndarray
        Simulation box_dimensions (Å), shape (3,) for orthorhombic or (3,3) for triclinic

    Returns
    -------
    centroids : dict[int, np.ndarray]
        Dictionary where keys are molecule IDs (int) and values are
        the corresponding centroids (numpy arrays of shape (3,)).
    radii : dict[int, float]
        Dictionary where keys are molecule IDs (int) and values are
        their maximum radii measured from centroid to furthest atom.
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
# 3. NEIGHBOR CANDIDATES
# ------------------------------

def get_neighbor_candidates(box_dimensions, centroids, k=10, threshold_distance=10):
    """
    Find the k nearest neighbor candidates for each molecule using KDTree and periodic boundary conditions (PBC),
    and filter pairs by a maximum centroid-to-centroid distance.

    Parameters
    ----------
    box_dimensions : np.ndarray
        Simulation box dimensions (Å), shape (3,) for orthorhombic or (3,3) for triclinic.
    centroids : dict[int, np.ndarray]
        Dictionary where keys are molecule IDs (int) and values are
        the corresponding centroids (numpy arrays of shape (3,)).
    k : int, optional
        Number of nearest neighbors to return per molecule. Default is 10.
    threshold_distance : float, optional
        Maximum allowed centroid-to-centroid distance (in Å) for a pair to be considered neighbors.
        Default is 10.

    Returns
    -------
    neighbor_candidates : list[tuple[int, int]]
        List of candidate neighbor pairs (i, j) where i < j and distance < threshold_distance.
    """

    ids = list(centroids.keys())
    coords = np.vstack([centroids[i] for i in ids])
    coords_wrapped = mh.wrap_points(coords, box_dimensions)
    kd = KDTree(coords_wrapped, boxsize=box_dimensions)

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

    # Calculate and store distances between neighbor candidate pairs using centroids
    neighbor_distances = {}

    for i, j in neighbor_candidates:
        c_i = centroids[i]
        c_j = centroids[j]
        dist = mh.mic_distance(c_i, c_j, box_dimensions)
        neighbor_distances[(i, j)] = dist

    # Filter neighbor_candidates where distance < 10 Å
    neighbor_candidates = [pair for pair, dist in neighbor_distances.items() if dist < threshold_distance]
        
    return neighbor_candidates


# ------------------------------
# 4. BLOCKING ALGORITHM
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
        Dictionary where keys are molecule IDs (int) and values are
        the corresponding electron clump centroids (numpy arrays of shape (3,)).
    radii : dict[int, float]
        Dictionary where keys are molecule IDs (int) and values are
        their maximum radii measured from centroid to furthest atom.
    mol_meshes : dict[int, trimesh.Trimesh]
        Dictionary where keys are molecule IDs (int) and values are
        the corresponding molecular meshes (trimesh.Trimesh objects).

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
        Dictionary where keys are molecule IDs (int) and values are
        the corresponding electron clump centroids (numpy arrays of shape (3,)).
    centroids : dict[int, np.ndarray]
        Dictionary where keys are molecule IDs (int) and values are
        the corresponding centroids (numpy arrays of shape (3,)).
    radii : dict[int, float]
        Dictionary where keys are molecule IDs (int) and values are
        their maximum radii measured from centroid to furthest atom
    mol_meshes : dict[int, trimesh.Trimesh]
        Dictionary where keys are molecule IDs (int) and values are
        the corresponding molecular meshes (trimesh.Trimesh objects).
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

def find_neighbor_pairs(mol_meshes, df_gro, box_dimensions, path, k=10, export_csv=False):
    """
    Determine all unblocked neighbor pairs.

    Parameters
    ----------
    mol_meshes : dict[int, trimesh.Trimesh]
        Dictionary where keys are molecule IDs (int) and values are
        the corresponding molecular meshes (trimesh.Trimesh objects).
    df_gro : pd.DataFrame
        DataFrame containing molecular data, multi-indexed by (res_id, atom_name).
    box_dimensions : np.ndarray
        Simulation box_dimensions (Å), shape (3,) for orthorhombic or (3,3) for triclinic
    k : int, optional
        Number of nearest neighbors to consider per molecule.
    export_csv : bool, optional
        Whether to export intermediate data (e_centroids, centroids, neighbor_candidates or neighbor_pairs) as CSV files.

    Returns
    -------
    neighbor_pairs : list[tuple[int, int]]
        List of unblocked neighbor pairs (i, j).
    e_centroids : dict[int, np.ndarray]
        Dictionary where keys are molecule IDs (int) and values are
        the corresponding electron clump centroids (numpy arrays of shape (3,)).
    centroids : dict[int, np.ndarray]
        Dictionary where keys are molecule IDs (int) and values are
        the corresponding centroids (numpy arrays of shape (3,)).
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
    neighbor_pairs = true_neighbors(e_centroids, centroids, radii, mol_meshes, neighbor_candidates)
    
    
    # Export as csv file
    if export_csv is True:
        print('-' * 40)
        name = os.path.basename(path).rsplit('.', 1)[0] # e.g., 'npt-HK4'
        
        df_e_centroids = pd.DataFrame(e_centroids, index=['x', 'y', 'z']).T
        df_e_centroids.to_csv(f'{name}_e_centroids.csv', index_label='res_id', mode='w')
        print(f"Successfully exported {name}_e_centroids.csv")

        df_centroids = pd.DataFrame(centroids, index=['x', 'y', 'z']).T
        df_centroids.to_csv(f'{name}_centroids.csv', index_label='res_id', mode='w')
        print(f"Successfully exported {name}_centroids.csv")

        df_neighbors_candidates = pd.DataFrame(neighbor_candidates, columns=['mol_id_1', 'mol_id_2'])
        df_neighbors_candidates.to_csv(f'{name}_neighbor_candidates.csv', index=False, mode='w')
        print(f"Successfully exported {name}_neighbor_candidates.csv")

        df_neighbor_pairs = pd.DataFrame(neighbor_pairs, columns=['mol_id_1', 'mol_id_2'])
        df_neighbor_pairs.to_csv(f'{name}_neighbor_pairs.csv', index=False)
        print(f"Exported {name}_neighbor_pairs.csv")

    return neighbor_pairs, e_centroids, centroids, neighbor_candidates


# ------------------------------
# 5. IMPORT NEIGHBOR PAIRS
# ------------------------------

def import_csv(path):
    """
    Function to import CSV file and return data in appropriate format.
    
    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    dict[int, arr] or list[tuple]
        The imported data in appropriate format.
    """
    try:
        # Predefined formats
        centroids_format = ['res_id', 'x', 'y', 'z']
        neighbors_format = ['mol_id_1', 'mol_id_2']
        
        # Process CSV
        header = list(pd.read_csv(path, nrows=0)) # Checks the header of the CSV file
        if header == centroids_format: 
            df_centroids = pd.read_csv(path)
            data = {mol_id: np.array([x, y, z]) for _, mol_id, x, y, z in df_centroids.itertuples()}
            print(f"Successfully imported centroids CSV file: '{path}'.")
            return data
        elif header == neighbors_format:
            df_neighbors = pd.read_csv(path)
            data = [(mol_id_1, mol_id_2) for _, mol_id_1, mol_id_2 in df_neighbors.itertuples()]
            print(f"Successfully imported neighbors CSV file: '{path}'.")
            return data
        else:
            raise ValueError(f"Unrecognized CSV format. Found header: {header}")
    
    # Error checks
    except ValueError as e:
        print(f"[Error] {e}")
        return None
        
    except FileNotFoundError as e:
        print(e)
        return None
    
    except Exception as e:
        print(f"Error: An unexpected error occurred while importing '{path}'. Details: {e}")
        return None
   
   
# ------------------------------
# 6. VERIFY BLOCKING MOLECULES
# ------------------------------ 

# --------- HELPER FUNCTIONS ---------
    
def map_symmetric_neighbors(neighbor_candidates, centroids, mol_with_no_neighbors=True):
    """
    Function to create a symmetric mapping of neighbor candidates.
    Returns a dictionary mapping each molecule ID to a list of its neighbors,
    e.g. {1:[2,4], 2:[1,4], 3:[], 4:[1,2],...}.
    """
    # Create symmetric pairing
    neighbor_map = defaultdict(list)
    for mol_id_1, mol_id_2 in neighbor_candidates:
        neighbor_map[mol_id_1].append(mol_id_2) # Pass 1
        neighbor_map[mol_id_2].append(mol_id_1) # Pass 2
    
    # Molecules with no neighbors are added to dict
    if mol_with_no_neighbors is True:
        superset = set(centroids.keys()) # centroids imported just for its keys (mol ID)
        subset = set(neighbor_map.keys())
        no_neighbor_mols = superset - subset
        for mol_id in no_neighbor_mols:
            neighbor_map[mol_id] = []
        
    neighbor_candidates_symmetric = dict(neighbor_map)
    return neighbor_candidates_symmetric

def get_distinct_colors(n, alpha=200):
    """
    Return a list of n RGBA colors in 0-255 int format.
    method: 'hsv' (even hue spacing) or 'matplotlib_tab' (uses tab20).
    alpha: 0-255
    """
    # Invalid number or 0
    if n <= 0: return []
    
    # Evenly spaced HSV hues (good for 1 to 11 colors)
    hues = np.linspace(0, 1, n, endpoint=False)
    sats = 0.65   # saturation
    vals = 0.90   # value (brightness)
    colors = []
    for h in hues:
        rgb = mcolors.hsv_to_rgb((h, sats, vals))  # returns 3 floats 0..1
        colors.append((int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255), int(alpha)))
    return colors

def get_user_input_unpaired(exclusive_ids):
    """
    Handles the user prompt for selecting unpaired neighbors to view.
    """
    print("Select the non-paired neighbors to view. Use the following format:")
    print("  - For specific molecules: '[101, 104, 403]'")
    print("  - For all neighbors: 'all'")
    print(f"  - For no neighbors: ''\n{"-" * 40}")
    time.sleep(1)
    view_ids = input("View: ")
    
    # Process user input
    try: 
        view_input = view_ids.strip().lower()
        if view_input == 'all':
            user_select_ids = exclusive_ids
            return user_select_ids
        elif view_input == '': 
            user_select_ids = []
            return user_select_ids
        else: 
            user_select_ids = ast.literal_eval(view_ids) # Safely evaluate list input
            # Check if user input is a list of integers
            if not isinstance(user_select_ids, list) or not all(isinstance(int(x), int) for x in user_select_ids):
                 raise ValueError("Input is not a valid list of integers.")
            # Final validation check
            if not set(user_select_ids).issubset(set(exclusive_ids)):
                print(f"Selected molecule ID not in neighbor list. Please try again.")
                return None
            return user_select_ids
    
    # Error check
    except Exception as e: # Invalid input format
        print(f"Invalid input format. Please use the specified format or check IDs.")
        return None
    
# --------- MAIN FUNCTION ---------

def view_molecule(mol_id, e_centroids, mol_meshes, 
                  neighbor_candidates, neighbor_pairs,
                  view_unpaired_mols=None,
                  distinct_color=False,
                  alpha=255):
    """
    Function to visualize a molecule with its mesh, e-centroid, centroid, and neighbors.
    
    Parameters
    ----------
    mol_id: int
        The ID of the molecule to visualize.
    e_centroids : dict[int, np.ndarray]
        Dictionary where keys are molecule IDs (int) and values are
        the corresponding electron clump centroids (numpy arrays of shape (3,)).
    mol_meshes : dict[int, trimesh.Trimesh]
        Dictionary where keys are molecule IDs (int) and values are
        the corresponding molecular meshes (trimesh.Trimesh objects).
    neighbor_candidates : list[tuple[int, int]]
        List of candidate neighbor pairs (i, j) to test for blocking.
    neighbor_pairs : list[tuple[int, int]]
        List of unblocked neighbor pairs (i, j).
    view_unpaired_mols: boolean, int, list[int], None, optional
        List of molecule IDs to specifically view. 
            - True, view all unpaired neighbors.
            - False, view no unpaired neighbors.
            - int, view specified number of unpaired neighbors.
            - list, view specific unpaired neighbors.
            - None, prompt user for input.
    distinct_color: boolean, optional
        If True, use a distinct color scheme for the meshes for higher contrast.
    alpha: int, optional
        Alpha transparency value for the meshes (0-255). Default is 255 (opaque).
    Returns
    -------
    IPython.core.display.HTML
        A Trimesh scene object for visualization.
    """
    # Parameters
    snc = map_symmetric_neighbors(neighbor_candidates, e_centroids, mol_with_no_neighbors=True)
    snp = map_symmetric_neighbors(neighbor_pairs, e_centroids, mol_with_no_neighbors=True)
    snc_ids, snp_ids = snc[mol_id], snp[mol_id]
    exclusive_ids = sorted(list(set(snc_ids).difference(set(snp_ids))))
    
    # Show to user paired neighbors
    print(f'Neighbor pairs for molecule {mol_id}: {snp_ids}')
    print(f'Non-paired neighbors are: {exclusive_ids}\n{"-"*40}')
    time.sleep(1)
    
    # -----------------------------
    # User WANT TO SEE ALL (True), show all unpaired neighbors 
    if view_unpaired_mols is True: user_select_ids = exclusive_ids

    # User DOESN'T WANT TO SEE ANY (False), show no unpaired neighbors
    elif view_unpaired_mols is False: user_select_ids = []
    
    # User WANT TO A FEW, show the number of unpaired neighbors specified by user
    elif isinstance(view_unpaired_mols, int): 
        if view_unpaired_mols < 0:
            print("Please enter a non-negative integer for the number of unpaired neighbors to view."); return None
        elif view_unpaired_mols > len(exclusive_ids):
            print(f"Only {len(exclusive_ids)} unpaired neighbors available. Showing all of them.")
            user_select_ids = exclusive_ids
        else:
            user_select_ids = exclusive_ids[:view_unpaired_mols]
    
    # User DOES NOT SPECIFY (None), prompt user which unpaired neighbors to view
    elif view_unpaired_mols is None: user_select_ids = get_user_input_unpaired(exclusive_ids)
    
    # User DOES SPECIFY (list), use their selection of unpaired neighbors
    else:
        user_select_ids = view_unpaired_mols
        # Check if user input is valid
        if not set(user_select_ids).issubset(set(exclusive_ids)): 
            print(f"Selected molecule ID not in neighbor list. Please try again."); return None
    # -----------------------------

    # Verified successful input
    print(f"You have selected: {user_select_ids}"); time.sleep(1) 
    
    # Overall molecule meshes to show
    meshes_to_show_ids = sorted([mol_id] + snp_ids + user_select_ids)
    meshes_to_show = [mol_meshes[ids] for ids in meshes_to_show_ids]
    print(f'{'-'*40}\nYou are now viewing molecules {meshes_to_show_ids}')

    # Line mesh for unblocked neighbors
    meshes_lines = []
    target_coord = e_centroids[mol_id]
    for ids in snp_ids:
        neighbor_coord = e_centroids[ids]
        segment = [target_coord, neighbor_coord]
        cyl = trimesh.creation.cylinder(radius=0.05, segment=segment, sections=24)
        cyl.visual.face_colors = [0, 0, 0, 255] # Black color
        meshes_lines.append(cyl)
        
    # Palette color scheme for higher contrast
    if distinct_color is True:
        meshes_to_show = [mesh.copy() for mesh in meshes_to_show] # make copy of meshes to color
        palette = get_distinct_colors(len(meshes_to_show), alpha=alpha)
        for mesh, color in zip(meshes_to_show, palette):
            color = np.array(color)
            mesh.visual.face_colors = np.tile(color, (len(mesh.faces), 1))
    else: # Default color scheme
        for mesh in meshes_to_show:
            mesh.visual.face_colors[:, 3] = alpha  # Modify only alpha channel

    # Note: meshes_lines MUST come before meshes_to_show due opacity (z-value)
    meshes = trimesh.util.concatenate(meshes_lines + meshes_to_show)
    scene = trimesh.Scene(meshes)
    
    return scene.show(viewer='gl')

# -----------------------------

# Export wildcard
__all__ = ['find_neighbor_pairs', 'import_csv', 'view_molecule']

# Check from CLI
if __name__ == "__main__":
    print("Running blocking_algo.py as a script")