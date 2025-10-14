"""
This module provides algorithms for making networkx graph with nodes as molecules and edges as pairs that have e_coupling and distance data as attributes
 
Sections:
----------
1. Make NetworkX Graph
    - Main Function:
      - Returns networkx_graph object, with option to show pairs without e_coupling data and option to plot in 3D.

2. Find Average Pairs per Molecule
    - Main Function:
      - Prints average pairs per molecule and draws histogram.
"""

# Imports
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
from . import mic_helper as mh

# ------------------------------
# MAKE NETWORKX GRAPH
# ------------------------------

def networkx_graph_making(centroids_path, neighbor_pairs_path, e_coupling_data_path, pairs_without_e_coupling_to_txt = True, plot_3D_to_gif = False):
    """
    Returns networkx_graph object, with option to show pairs without e_coupling data and option to plot in 3D.

    Parameters
    ----------
    centroids_path : str
        path to 'molecule_centroids.csv'.
    neighbor_pairs_path : str
        path to 'molecule_neighbor_pairs.csv'.
    e_coupling_data_path : str
        path to 'e-coupling_with_nn_distance.txt'. See under data folder for more information of txt formatting.
    pairs_without_e_coupling_to_txt : bool, optional
        Whether to export pairs without e-coupling data to txt (True by default).
    plot_3D_to_gif : bool, optional
        Whether to create 3D_model of graph in .gif file (False by default).

    Returns
    -------
    G : networkx.Graph object
        Networkx graph with nodes as molecules and edges as pairs that have e_coupling and distance data as attributes
    """
    # Get centroids from csv
    centroids_arr = np.loadtxt(centroids_path,delimiter=',',skiprows=1,usecols=range(1,4))

    # Get unblocked pairs
    pairs = np.loadtxt(neighbor_pairs_path,skiprows=1,delimiter=',',dtype=int)
    
    # Get e-coupling data from file
    data = np.loadtxt(
    e_coupling_data_path,
    delimiter="\t",
    skiprows=1,
    )
    mol_i, mol_j_nearest, extracted_value, _, _ = zip(*data)

    mol_i = list(map(int,mol_i))
    mol_j_nearest = list(map(int,mol_j_nearest))
    e_coupled_pairs = list(zip(mol_i, mol_j_nearest))

    # Fix order of ID, lower number on the left 
    e_coupled_pairs_arr = [(None,)] * len(e_coupled_pairs) #empty list to append into
    num = 0
    for (p,q) in e_coupled_pairs:
        if p > q:
            e_coupled_pairs_arr[num] = (q,p)
        else:
            e_coupled_pairs_arr[num] = (p,q)
        num +=1

    # Set edge attributes
    # e_coupling data
    e_coupled_pairs_dict = {}
    for item in e_coupled_pairs:
        p, q = item
        index = e_coupled_pairs.index(item)
        
        e_coupling = extracted_value[index]
        
        if p < q:
            pair_order = (p,q)
        else:
            pair_order = (q,p)

        if item in e_coupled_pairs_dict: # if duplicate
            if current_value := e_coupled_pairs_dict[item]["e_coupling"] < e_coupling:
                e_coupled_pairs_dict[pair_order] = {"e_coupling": e_coupling}
        else:
            e_coupled_pairs_dict[pair_order] = {"e_coupling": e_coupling}

    # distance data
    pairs_distance = {}
    for (p,q) in pairs:
        distance = mh.mic_distance(a= centroids_arr[p-1], b= centroids_arr[q-1], box_dimensions=112.4798)
        pairs_distance[(p,q)] = {"distance": distance}

    # Setting edge attributes
    G = nx.Graph()
    G.add_nodes_from(list(range(1,1502)))
    G.add_edges_from(pairs)

    nx.set_edge_attributes(G,e_coupled_pairs_dict)

    # Show edges without e-coupling
    if pairs_without_e_coupling_to_txt == True:
        edges_without_attributes=[]
        for u,v, data in G.edges(data=True):
            if not data:
                edges_without_attributes.append((u,v))

        molecule_name = centroids_path.rsplit('_', 1)[0]
        print(f"The pairs without e_coupling has been exported to: {molecule_name}_pairs_without_e_coupling.txt")
        np.savetxt(fname=f"{molecule_name}_pairs_without_e_coupling.txt", X= edges_without_attributes,fmt='%i')  

    # add Distance attribute of other edges
    nx.set_edge_attributes(G,pairs_distance)

    # 3D plotting
    if plot_3D_to_gif is True:
        edges_arr = np.empty(shape=(pairs.shape[0],2,3))
        numerator = 0
        for (p,q) in pairs:
            edges_arr[numerator] = [centroids_arr[p-1],centroids_arr[q-1]]
            numerator += 1
        nodes = centroids_arr
        edges = edges_arr
        
        def init():
            ax.clear()
            ax.scatter(*nodes.T, alpha=0.2, s=100, color="blue")
            for vizedge in edges:
                ax.plot(*vizedge.T, color="gray")
            ax.grid(False)
            ax.set_axis_off()


        def _frame_update(index):
            ax.view_init(index * 0.2, index * 0.5)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.tight_layout()

        ani = animation.FuncAnimation(
            fig,
            _frame_update,
            init_func=init,
            interval=50,
            cache_frame_data=False,
            frames=100,
        )

        ani.save(filename="temp.gif")
        plt.show()

    return G

# ------------------------------
# FIND AVERAGE PAIRS PER MOLECULE
# ------------------------------

def average_pairs_per_molecule(G):
    """
    Prints average pairs per molecule and draws histogram.

    Parameters
    ----------
    G : networkx.Graph object
    """
    # Check degree and average degree
    degree_hist = nx.degree_histogram(G)
    x_labels = list(range(len(degree_hist)))

    total_nodes = sum(degree_hist)
    weighted_total = 0
    for item in degree_hist:
        weighted_total += item * degree_hist.index(item)

    average = weighted_total/total_nodes

    print(f"Average pairs per molecule: {average}")

    plt.bar(x_labels,degree_hist)