"""
This module provides geometric analysis tools for molecular simulations,
including computation of 3D alpha shapes (concave hulls) and handling of
periodic boundary conditions (PBC).

Main Functions:
- alpha_shape_3D(points, alpha): Compute the alpha shape (concave hull) of a set of 3D points.
- unwrap_points(points, box_length): Unwrap molecule coordinates using the Minimum Image Convention.
- alpha_shape_3D_pbc(points, alpha, box_length): Compute the alpha shape of a molecule under PBC.

Dependencies:
- numpy
- scipy.spatial.Delaunay
- collections.Counter

Typical use cases include identifying the surface of molecular clusters and
analyzing molecular geometry in periodic simulation boxes.
"""


import numpy as np
from scipy.spatial import Delaunay
from collections import Counter


def alpha_shape_3D(points, alpha):
    """
    Compute the 3D alpha shape (concave hull) of a set of points.

    Parameters
    ----------
    points : (n, 3) array-like
        Array of 3D coordinates representing the points.
    alpha : float
        Alpha parameter controlling the concavity. Smaller values capture more detail.

    Returns
    -------
    hull_faces : list of tuples
        List of faces (as tuples of point indices) forming the surface of the alpha shape.

    Raises
    ------
    ValueError
        If fewer than 4 points are provided.
    """
    if len(points) < 4:
        raise ValueError("Need at least 4 points for a 3D hull")
    
    tetra = Delaunay(points)
    faces = []

    for simplex in tetra.simplices:
        # Vertices of the tetrahedron
        pts = points[simplex]
        
        # Compute circumsphere radius
        A = np.hstack((2*(pts - pts[0]), np.ones((4,1))))
        b = np.sum(pts**2 - pts[0]**2, axis=1)
        try:
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            center = x[:-1] + pts[0]
            r2 = np.sum((pts[0] - center)**2)
        except np.linalg.LinAlgError:
            continue
        
        if r2 < (1.0/alpha)**2:
            # Add boundary faces (each tetra has 4 faces)
            for i in range(4):
                face = tuple(sorted(simplex[np.arange(4) != i]))
                faces.append(face)
    
    # Remove internal faces (those appearing twice)
    face_count = Counter(faces)
    hull_faces = [f for f, c in face_count.items() if c == 1]
    
    return hull_faces


def unwrap_points(points, box_length):
    """
    Unwrap molecule coordinates using the Minimum Image Convention (MIC).

    Parameters
    ----------
    points : (n, 3) array-like
        Array of 3D coordinates to unwrap.
    box_length : float or array-like
        Length of the simulation box (single float for cubic, or array for each dimension).

    Returns
    -------
    unwrapped : (n, 3) ndarray
        Coordinates unwrapped relative to the first atom, accounting for periodic boundaries.
    """
    unwrapped = np.zeros_like(points)
    unwrapped[0] = points[0]  # reference atom
    
    for i in range(1, len(points)):
        delta = points[i] - points[i-1]
        # apply MIC shift
        delta -= box_length * np.round(delta / box_length)
        unwrapped[i] = unwrapped[i-1] + delta
    
    return unwrapped


def alpha_shape_3D_pbc(points, alpha, box_length):
    """
    Compute the 3D alpha shape of a molecule under periodic boundary conditions (PBC).

    Parameters
    ----------
    points : (n, 3) array-like
        Array of 3D coordinates representing the points.
    alpha : float
        Alpha parameter controlling the concavity.
    box_length : float or array-like
        Length of the simulation box.

    Returns
    -------
    faces : list of tuples
        List of faces (as tuples of point indices) forming the surface of the alpha shape.
    unwrapped : (n, 3) ndarray
        Unwrapped coordinates used for alpha shape calculation.
    """
    # unwrap molecule coordinates
    unwrapped = unwrap_points(points, box_length)
    
    # compute alpha shape on unwrapped coordinates
    faces = alpha_shape_3D(unwrapped, alpha)
    
    return faces, unwrapped
