"""
This module provides functionality to convert molecular simulation data (coordinates
and atom types) into 3D geometric meshes, primarily using the ball-and-stick
representation suitable for visualization. It accounts for Periodic Boundary
Conditions (PBC) when constructing bonds across box boundaries.

The meshing process is optimized for performance using `trimesh` for geometry
operations and `multiprocessing` for parallel processing of individual molecules.

Main Functions:
- molecules_to_meshes(): Main entry point to convert a dictionary of molecules
  into a dictionary of trimesh objects in parallel.
  
Helper Functions:
- infer_element(): Utility to determine the element type from an atom name.
- build_molecule_ballstick(): Core function to generate the ball-and-stick mesh
  for a single molecule.
- process_single_molecule(): Wrapper to process a single molecule for parallelism.

Dependencies:
- numpy
- trimesh
- scipy (for KDTree)
- tqdm (for progress bar)
- multiprocessing
- mic_helper (local module for MIC calculations)
"""
import itertools
import multiprocessing as mp
import os
import sys
import threading
import time
from functools import partial

import numpy as np
import trimesh
from scipy.spatial import KDTree
from tqdm import tqdm

from . import mic_helper as mh

# --------- GLOBAL SCOPE VARIABLES ---------

# CONFIG               
sphere_radius_scale = 2.0                # balls (atoms): 2.0 × van der Waals radius   # idk why 1.5 :(
bond_radius = 0.1                        # sticks (bonds): cylinder radius in Å
sphere_subdiv = 2                        # atom sphere detail (2 is moderate)

# Radii (Å)
vdw = {"H":1.20,"C":1.70,"N":1.55,"O":1.52,"F":1.47,"P":1.80,"S":1.80,"Cl":1.75,"Na":2.27,"K":2.75,"Ca":2.31}
cov = {"H":0.31,"C":0.76,"N":0.71,"O":0.66,"F":0.57,"P":1.07,"S":1.05,"Cl":1.02,"Na":1.66,"K":2.03,"Ca":1.74}

# Element inference 
hash_atom = {"OW": "O", "HW": "H", "HW1": "H", "HW2": "H"} # Atom name aliases
color_map = {'O': np.array([220, 20, 60, 255], dtype=np.uint8),     # Crimson Red
             'N': np.array([65, 105, 225, 255], dtype=np.uint8),    # Royal Blue
             'C': np.array([128, 128, 128, 255], dtype=np.uint8),   # Gray
             'H': np.array([230, 230, 230, 255], dtype=np.uint8),   # Light Gray
            }


# ------------------------------
# CREATING MOLECULE MESHES
# ------------------------------

# --------- HELPER FUNCTIONS ---------

# Infer element from atom name
def infer_element(atomname):
    # common water aliases
    if atomname in hash_atom: 
        return hash_atom[atomname]
    
    # simple: first letter, capitalize second if lowercase
    a = ''.join([c for c in atomname if c.isalpha()])   
    
    # join() joins items in an iterable into one string, '' is specified as the separator.
    # isalpha() method returns True if all the characters are alphabet letters (a-z).

    if a == '': 
        return "C"

    if len(a) >= 2 and a[1].islower(): 
        return (a[0]+a[1]).capitalize()
    
    return a[0].upper()


# Helper function to create a dictionary of molecules from DataFrame
def create_mol_dict(df_gro):
    molecules = {}
    for res_id in df_gro.index.get_level_values('res_id').unique():
        residue_data = df_gro.xs(res_id, level='res_id')
        # itertuples() - much faster for large DataFrames
        atom_list = [(row.Index, np.array([row.x, row.y, row.z])) for row in residue_data.itertuples()]
        molecules[res_id] = atom_list
    return molecules


# Algorithm to create ball-and-stick model of molecule under PBC
def build_molecule_ballstick(coords, elements, 
                             vdw, cov, box_length, 
                             sphere_radius_scale=0.3, 
                             sphere_subdiv=2, 
                             bond_radius=0.1):
    """
    Build trimesh ball-and-stick model under periodic boundary conditions.
    """
    # radii arrays
    vdw_r = np.array([vdw.get(e, 1.70) for e in elements])
    cov_r = np.array([cov.get(e, 0.76) for e in elements])
    
    # number of atoms
    n = len(coords)
    
    # --- Process spheres using chunking--- 
    # Base icosphere (unit radius)
    base_sphere = trimesh.creation.icosphere(subdivisions=sphere_subdiv, radius=1.0)
    # Precompute position, radii and number of vertices for each sphere
    coords_wrapped = np.mod(coords, box_length)
    scaled_radii = vdw_r * sphere_radius_scale  
    num_sphere_vertices = len(base_sphere.vertices)
    # Chunk (pre-allocate) array to hold icosphere object
    meshes_sphere = np.empty(n, dtype=object)
    
    for idx, (pos, r) in enumerate(zip(coords_wrapped, scaled_radii)):
        sphere = base_sphere.copy()
        sphere.apply_scale(r)
        sphere.apply_translation(pos)
        sphere_color = color_map.get(elements[idx], color_map.get('C')) # element not found default to grey (C)
        sphere.visual.vertex_colors = np.tile(sphere_color, (num_sphere_vertices, 1))
        meshes_sphere[idx] = sphere 
        
    # --- Process cylinder with numba --- 
    # For small molecules roughly < 1500 atoms, brute force is most efficient
    # Base cylinder (for number of faces and color assignment)
    base_cyl = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=24)
    num_bond_faces = len(base_cyl.faces)
    bond_color = color_map.get('H') # light grey
    
    meshes_cylinder = []
    if n < 1500: # brute force
        for i in range(n):
            for j in range(i+1, n):
                d = mh.mic_distance(coords[i], coords[j], box_length)
                thr = 1.2 * (cov_r[i] + cov_r[j])
                if d < thr:
                    # Unwrap j relative to i
                    disp = mh.mic_vector(coords[j] - coords[i], box_length)
                    pos_i = np.mod(coords[i], box_length)
                    pos_j = pos_i + disp  # may fall outside box but correct bond vector
                    seg = np.vstack((pos_i, pos_j))
                    cyl = trimesh.creation.cylinder(radius=bond_radius,
                                                    segment=seg, sections=24)
                    cyl.visual.face_colors = np.tile(bond_color, (num_bond_faces, 1))
                    meshes_cylinder.append(cyl)
    else: # k-d tree 
        tree = KDTree(coords, leafsize=10)
        for i in range(n):
            _nearest_coords, nearest_index = tree.query(coords[i], k=9) # 8 neighbors
            for j in nearest_index[1:]: # Skip itself
                d = mh.mic_distance(coords[i], coords[j], box_length)
                thr = 1.2 * (cov_r[i] + cov_r[j])
                if d < thr:
                    # Unwrap j relative to i
                    disp = mh.mic_vector(coords[j] - coords[i], box_length)
                    pos_i = np.mod(coords[i], box_length)
                    pos_j = pos_i + disp # may fall outside box but correct bond vector
                    seg = np.vstack((pos_i, pos_j))
                    cyl = trimesh.creation.cylinder(radius=bond_radius,
                                                    segment=seg, sections=24)
                    cyl.visual.face_colors = np.tile(bond_color, (num_bond_faces, 1))
                    meshes_cylinder.append(cyl)

    # --- Merge all into one mesh ---
    molecule = trimesh.util.concatenate(meshes_sphere.tolist() + meshes_cylinder)
    return molecule


# Convert single molecule to mesh
def process_single_molecule(mol_items,  
                            vdw, cov, box_length,
                            sphere_radius_scale=0.3, 
                            sphere_subdiv=2, 
                            bond_radius=0.1):
    """Process a single molecule and return (mol_id, mesh)"""
    mol_id, atoms = mol_items
    elements = [infer_element(name) for name, _ in atoms]
    coords = np.vstack([pos for _, pos in atoms])
    
    mesh = build_molecule_ballstick(
        coords, elements, vdw, cov, box_length,
        sphere_radius_scale=sphere_radius_scale,
        sphere_subdiv=sphere_subdiv,
        bond_radius=bond_radius
    )
    
    return mol_id, mesh


# --------- MAIN FUNCTION ---------

# Function to convert all molecules to meshes
def molecules_to_meshes(df_gro, box_dimensions,
                        sphere_radius_scale=0.3,
                        sphere_subdiv=2,
                        bond_radius=0.1,
                        num_processes=None, context="spawn"):
    """
    Convert parsed molecules into trimesh meshes.

    Parameters
    ----------
    df_gro : pd.DataFrame
        DataFrame containing molecular data, multi-indexed by (res_id, atom_name).
    molecules : dict[int, list[tuple]]
        From parse_gro(): molecules[i] = [(atomname, coords), ...]
        coords must be in Å
    box_dimensions : np.ndarray
        Simulation box_dimensions (Å), shape (3,) for orthorhombic or (3,3) for triclinic
    sphere_radius_scale : float
        Scaling factor for atom radii
    sphere_subdiv : int
        Subdivisions for icosphere (mesh resolution)
    bond_radius : float
        Cylinder radius for bonds
    num_processes: int/None
        Number of CPU cores to use (all if not specified)
    context: str
        Either "spawn" or "fork". Windows users are only limited to using "spawn", while Mac and Linux users can use the faster "fork"

    Returns
    -------
    mol_meshes : dict[int, trimesh.Trimesh]
        Dictionary of molecule meshes keyed by mol_id
    """
    # create molecules dictionary if input is DataFrame
    molecules = create_mol_dict(df_gro)
    mol_items = list(molecules.items())
    num_mol = len(mol_items)
    
    # assume orthorhombic box for now
    if box_dimensions.shape == (3,):
        box_length = box_dimensions
    else:
        raise NotImplementedError("Triclinic box handling not yet implemented")
    
    # --- Multiprocessing ---
    # number of processes (use all CPUs if not specified)
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # partial function with fixed parameters
    process_func = partial(
        process_single_molecule,
        vdw=vdw,
        cov=cov,
        box_length=box_length,
        sphere_radius_scale=sphere_radius_scale,
        sphere_subdiv=sphere_subdiv,
        bond_radius=bond_radius
    )
    # By default set mp.get_context to spawn for compatibility with Linux, Mac and Windows users
    ctx = mp.get_context(context)
    with ctx.Pool(processes=num_processes) as pool:
        tqdm_iterator = tqdm(
            pool.imap(process_func, mol_items),
            total=num_mol,
            desc=f'Processing {num_mol} molecules with {num_processes} logical cores',
            colour='#7BC8F6'
        )
        mol_meshes = dict(tqdm_iterator) # initialise progress bar
        
    return mol_meshes


# ------------------------------
# EXPORTING MOLECULE MESHES
# ------------------------------

# --------- HELPER FUNCTIONS ---------

# Loading bar when tqdm does not work (e.g. single process)
class Spinner:
    def __init__(self, message="Working..."):
        self._message = message
        self._done = False
        self._thread = threading.Thread(target=self._animate)

    def _animate(self):
        spinner = itertools.cycle(['.', '..', '...'])
        while not self._done:
            # Use '\r' to return to the start of the line, overwriting previous text
            sys.stdout.write(f'\r{self._message}{next(spinner)}   ') 
            sys.stdout.flush() 
            time.sleep(0.5)

        # Final message to show completion and move to a new line
        sys.stdout.write(f'\rDone: {self._message} complete!   \n')
        sys.stdout.flush()

    def start(self):
        """Starts the spinner animation in a separate thread."""
        self._thread.start()

    def stop(self):
        """Stops the spinner and waits for the thread to finish."""
        self._done = True
        self._thread.join()
        

# Functions for NPZ
def npz_extract_mesh_data(mol_items):
    """
    Extract and return singular mesh data: mol_id, vertices, faces, and colors.
    """
    mol_id, mesh = mol_items

    # Extract individual mesh data
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    colors = mesh.visual.face_colors.copy()
    
    # Store the arrays with unique keys
    mesh_key_prefix = f'mesh_{mol_id:04d}'
    data = {
        f'{mesh_key_prefix}_vertices': vertices,
        f'{mesh_key_prefix}_faces': faces,
        f'{mesh_key_prefix}_colors': colors
    }
    return data

def npz_export_meshes(mol_meshes, path, num_processes=None, context='spawn'):
    """
    Export all molecular meshes to a single npz file.
    """
    # Generate output file name
    name = os.path.basename(path).rsplit('.', 1)[0] # e.g., 'npt-HK4'
    output_file = f'{name}_meshes.npz'
    
    # Converting for multiprocessing
    mol_items = list(mol_meshes.items())
    num_mol = len(mol_meshes)
    
    # --- Multiprocessing ---
    # number of processes (use all CPUs if not specified)
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    all_mesh_data = {}
    ctx = mp.get_context(context)
    with ctx.Pool(processes=num_processes) as pool:
        # Progress bar
        tqdm_iterator = tqdm(
            pool.imap(npz_extract_mesh_data, mol_items),
            total=num_mol,
            desc=f'Extracting mesh data with {num_processes} logical cores',
            colour='#7BC8F6'
        )
        # Iterate over and add results to dictionary
        for data_dict in tqdm_iterator:
            all_mesh_data.update(data_dict)
    print(f'Converted {num_mol} molecular meshes into {len(all_mesh_data)} numpy arrays.')
    
    # Save to .npz file
    loading = Spinner(f"Saving all meshes to {output_file}")
    loading.start()
    np.savez_compressed(output_file, **all_mesh_data)
    loading.stop()
    
    print(f"Successfully saved all meshes to a single file: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    return None


# Functions for PLY
def ply_export_single_mesh(mol_items, name):
    """Export a single mesh to a single PLY file."""
    mol_id, mesh = mol_items
    ply_file = os.path.join(f'{name}_meshes', f'mesh_{mol_id:04d}.ply')
    mesh.export(ply_file)
    return os.path.getsize(ply_file)

def ply_export_meshes(mol_meshes, path, num_processes=None, context='spawn'):
    """Create a directory, then store all meshes as PLY files."""
    # Generate output directory name
    name = os.path.basename(path).rsplit('.', 1)[0] # e.g., 'npt-HK4'
    dir = f'{name}_meshes'
    if not os.path.exists(dir): 
        os.makedirs(dir)
        
    # --- Multiprocessing ---
    # number of processes (use all CPUs if not specified)
    if num_processes is None: 
        num_processes = mp.cpu_count()
    
    # partial function with fixed parameters
    process_func = partial(ply_export_single_mesh, name=name)
    
    # parallel execution
    total = len(mol_meshes)
    mol_items = mol_meshes.items()
    total_size = 0
    ctx = mp.get_context(context)
    with ctx.Pool(processes=num_processes) as pool:
        tqdm_iterator = tqdm(
            # Use imap_unordered for a lazy, memory-efficient map
            pool.imap_unordered(process_func, mol_items),
            total=total,
            desc=f'Exporting {total} meshes with {num_processes} logical cores',
            colour='#7BC8F6'
        )
        for bytes in tqdm_iterator: # Initialise progress bar
            total_size += bytes
            
    print(f"Successfully saved all meshes into directory: {dir}")
    print(f"File size: {total_size / (1024*1024):.2f} MB")
    return None

# --------- MAIN FUNCTION ---------

def export_meshes(mol_meshes, path, export_format='npz', num_processes=None, context='spawn'):
    """
    Export molecular meshes to a specified file format ('npz' or 'ply').
    
    Creating a PLY file is faster but results in larger file sizes.
    Meanwhile, a NPZ file is more compact but takes longer to create.

    Parameters
    ----------
    mol_meshes : dict
        A dictionary where keys are molecule IDs (int) and values are
        the corresponding molecular meshes (trimesh.Trimesh objects).
        the corresponding molecular meshes (trimesh.Trimesh objects).
    export_format : str
        The file format used for exporting the meshes (e.g., 'npz', 'ply).
    num_processes : int, optional
        The number of processes to use for parallel execution. If None,
        the function will use all available CPU cores. Default is None.
    context : str, optional
        The multiprocessing context to use ('fork', 'spawn', or 'forkserver').
        Default is 'spawn'.

    Returns
    -------
    None
        The function performs a disk write operation and does not return a value.
    """
    if export_format == 'npz':
        npz_export_meshes(mol_meshes, path=path, num_processes=num_processes, context=context)
    elif export_format == 'ply':
        ply_export_meshes(mol_meshes, path=path, num_processes=num_processes, context=context)
    else:
        raise ValueError(f"Unsupported export format: {export_format}. Supported formats are 'ply' and 'npz'.")
    
    return None