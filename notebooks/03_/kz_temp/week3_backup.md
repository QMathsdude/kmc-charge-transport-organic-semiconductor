# Report 3
## :speech_balloon: Introduction
Hello
<div style="text-align: center;">
  <img src="../reports/images/02_Angles_Initial_Idea.jpg" style="max-width: 100%; height: auto;">
</div>

## :dart: Goal

The goal this week was ...
<b>

- hello

</b>

## :white_check_mark: Task(s) Accomplished 
<b>

1. Testing

</b>

## :balance_scale: Comparison(s) between Methods

| Criteria | :package: **Sub-box** | :japanese_ogre: **Brute-Force** |
|:---|:---|:---|
| **Primary Use** | General-purpose programming, Machine Learning | Statistical analysis, Data visualization |
| **Learning Curve** | Often considered easier for beginners | Can be steeper due to specialized syntax |
| **Ecosystem** | Vast and versatile (TensorFlow, PyTorch, etc.) | Strong focus on statistics (ggplot2, tidyr, etc.) |
| **Community** | Large, general-purpose programming community | Large, statistics-focused community |

## :package: Method 1: Sub-box

### Minimum Image Convection

In molecular simulations, we often simulate a finite number of particles inside a cubic periodic box to mimic bulk matter. Periodic boundary conditions (PBC) replicate the box infinitely in all directions.

<img src="images/02_pbc.png" alt="isolated" width="300"/>

When computing distances between particles, we must consider the periodic identical particle on other unit cells, but is the shortest path.

Here, we use the Minimum Image Convention (MIC), which only consider the nearest periodic image of each particle when computing pairwise distances.

MIC redefines the displacement to be the shortest path on the periodic domain, ensuring $Δx∈[−L/2,+L/2).$

If the distance between two points within the same unit cell is $\Delta x=x_2 - x_1$, then the minimum image displacement is $\Delta x_{mic} = \Delta x - L \: \text{round}\cdot(\frac{\Delta x}{L})$

For example, suppose we want to compute the displacement vector between two points $x_1 = 0.2$ and $x_2 = 0.9$. Naively: $\Delta x = 0.7$. But this is not the shortest path, it's only 0.3 if we go backwards.

```python
def minimum_image(dx, box_length):
    """Returns the minimum image of a coordinate difference."""
    reciprocal_half_box = 1.0 / (0.5 * box_length) 
    k = int(dx * reciprocal_half_box)  
    return dx - k * box_length  
```

With this, we can determine the midpoint of two points under PBC.
```python
def midpoint_pbc(p1, p2, box_length):
    """Returns the midpoint between two positions under periodic boundary conditions."""
    dx = p2 - p1
    dx_mic = minimum_image(dx, box_length)
    midpoint = (p1 + 0.5 * dx_mic) % box_length
    return midpoint
```

Without MIC, molecules may appear artificially far apart or “split” across boundaries, leading to wrong result.


### Spherical molecule model

To model the molecules in question, we first approximate them as spheres with smallest volume that contains all of their atoms.

Given a set of points, the **Minimum Enclosing Ball** (MEB) is the sphere that encloses all points such that the radius is minimum.

The `miniball` algorithm computes this MEB efficiently. 

Python has a prebuilt `miniball` library and we modified it for PBC.

```python
def mec_pbc_miniball(positions, box_length, references=None):
    """
    Finds approximate minimum enclosing sphere under PBC using the miniball library.
    """
    n = len(positions)
    if references is None:
        references = range(n)
    
    best_center = None
    best_radius = np.inf
    
    for i in references:
        ref = positions[i]
        
        # Unwrap positions relative to reference
        diffs = positions - ref
        diffs = mic_vector(diffs, box_length)
        unwrapped = ref + diffs
        
        # Compute Euclidean MEB via miniball
        center_unwrapped, radius_unwrapped_sq = miniball.get_bounding_ball(unwrapped)
        radius_unwrapped = np.sqrt(radius_unwrapped_sq)
        
        # Map center into primary box
        center_primary = np.mod(center_unwrapped, box_length)
        
        # Check radius under true PBC
        dists = np.array([mic_distance(center_primary, p, box_length) for p in positions])
        radius_check = np.max(dists)
        
        if radius_check < best_radius:
            best_radius = radius_check
            best_center = center_primary
    
    return best_center, best_radius
```
Here is how the output appear:

<img src="images/02_sphere_model.png" alt="isolated" width="400"/>



### Blocking algorithm




## :japanese_ogre: Method 2: Brute-Force
Hello
### :space_invader: Code
```python
print("Hello Dunia")
```