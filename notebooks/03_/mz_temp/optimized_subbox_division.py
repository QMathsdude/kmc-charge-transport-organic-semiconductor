#imports
import numpy as np
import pandas as pd

def subbox_division(number_of_subboxes, box_dimensions, midpoints_df):
    subboxes_dimensions = [x / number_of_subboxes ** (1 / 3) for x in box_dimensions] #get subbox dimensions and put into a list


    # Make a list of all possible indices of subboxes, additionally with an outer layer
    matrix = [] #Empty list to append into later
    n = round(number_of_subboxes ** (1 / 3)) # Maximum indices for the box, excluding outer layer
    # Nested for loop to create combinations of 3 from a set of numbers
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            for k in range(1, n + 1):
                matrix.append([i, j, k])

    # Find molecules within a subbox and insert it as a new row into a dataframe
    subbox = pd.DataFrame(columns=["index", "res_id", "x", "y", "z"])
    iterate = 1
    for p, q, r in matrix:
        molecule_in_subbox = midpoints_df.loc[
            (midpoints_df["mid_x"] < subboxes_dimensions[0] * p)
            & (midpoints_df["mid_y"] < subboxes_dimensions[1] * q)
            & (midpoints_df["mid_z"] < subboxes_dimensions[2] * r)
            & (midpoints_df["mid_x"] >= subboxes_dimensions[0] * (p - 1))
            & (midpoints_df["mid_y"] >= subboxes_dimensions[1] * (q - 1))
            & (midpoints_df["mid_z"] >= subboxes_dimensions[2] * (r - 1))
        ]

        subbox.loc[iterate] = {
            "index": (p, q, r),
            "res_id": molecule_in_subbox["res_id"].values,
            "x": molecule_in_subbox["mid_x"].values,
            "y": molecule_in_subbox["mid_y"].values,
            "z": molecule_in_subbox["mid_z"].values,
        }
        iterate += 1
    
    # Create empty lists to append to
    edges_res_id = []
    edges_index = []

    # Select a box
    for n in subbox.index:
        selected_box = subbox.at[n,"index"] 
        selected_box_str = ",".join(map(str,selected_box))
    # Get indices of adjacent boxes to the selected box
        adjacent_boxes = []
        for (p,q,r) in [list(selected_box)]:
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    for k in [-1,0,1]:
                        x = p + i
                        y = q + j
                        z = r + k
                        match x:
                            case 0:
                                x = 11
                            case 12:
                                x = 1
                            case _:
                                x = x
                        match y:
                            case 0:
                                y = 11
                            case 12:
                                y = 1
                            case _:
                                y = y
                        match z:
                            case 0:
                                z = 11
                            case 12:
                                z = 1
                            case _:
                                z = z
                        adjacent_boxes.append((x,y,z))
            # adjacent_boxes.remove((p,q,r)) # Uncomment this line to exclude itself from adjacent_boxes
    # For loop iterating over each adjacent box to find the nodes inside them, then append into list
        for (p,q,r) in adjacent_boxes:
            adjacent_box_str = ",".join(map(str,(p,q,r)))
            adjacent_nodes = subbox.loc[subbox["index"] == (p, q, r), "res_id"]
            if adjacent_nodes.empty:
                        continue
            edges_res_id.append(adjacent_nodes.values[0]) 
            edges_index.append((selected_box_str,adjacent_box_str))

    subboxes = pd.DataFrame(data = {'res_id': edges_res_id}, index = pd.MultiIndex.from_tuples(edges_index, names=['select_index', 'adjacent_index']))
    # subboxes.to_csv(path_or_buf="put path here")
    return subboxes

def flatten_subboxes(subboxes):
    # Group edges_df by 'select_index' and aggregate all res_id lists from nearby indices
    def flatten(lists):
        return [item for sublist in lists for item in sublist]

    nearby_resid_df = (
        subboxes.groupby('select_index')['res_id']
        .apply(lambda x: flatten([v if isinstance(v, (list, np.ndarray)) else [] for v in x]))
        .reset_index()
        .rename(columns={'res_id': 'nearby_res_id'})
    )

    return nearby_resid_df