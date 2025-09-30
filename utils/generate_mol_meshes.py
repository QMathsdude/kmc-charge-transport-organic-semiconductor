"""
hello
"""

import multiprocessing as mp
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
def molecules_to_meshes(molecules, box_dimensions,
                        sphere_radius_scale=0.3,
                        sphere_subdiv=2,
                        bond_radius=0.1,
                        num_processes=None):
    """
    Convert parsed molecules into trimesh meshes.

    Parameters
    ----------
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

    Returns
    -------
    mol_meshes : dict[int, trimesh.Trimesh]
        Dictionary of molecule meshes keyed by mol_id
    """
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

    # parallel execution
    mol_items = list(molecules.items())
    num_mol = len(mol_items)
    
    with mp.Pool(processes=num_processes) as pool:
        tqdm_iterator = tqdm(
            pool.imap(process_func, mol_items),
            total=num_mol,
            desc=f'Processing {num_mol} molecules with {num_processes} logical cores',
            colour='#7BC8F6'
        )
        
        mol_meshes = list(tqdm_iterator) # initialise progress bar
        
    return dict(mol_meshes) # return dictionary