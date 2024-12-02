import numpy as np
import tifffile as tiff
from skimage.morphology import skeletonize  # Import for skeletonization
from skan import Skeleton, summarize
from joblib import Parallel, delayed
import sys

def neighbors(coord):
    x, y, z = coord
    return [
        # 6 primary (axis-aligned) neighbors
        (x.item() + 1, y.item(), z.item()), (x.item() - 1, y.item(), z.item()), 
        (x.item(), y.item() + 1, z.item()), (x.item(), y.item() - 1, z.item()),
        (x.item(), y.item(), z.item() + 1), (x.item(), y.item(), z.item() - 1),
        
        # 12 diagonal neighbors (changing two axes)
        (x.item() + 1, y.item() + 1, z.item()), (x.item() - 1, y.item() - 1, z.item()),
        (x.item() + 1, y.item(), z.item() + 1), (x.item() - 1, y.item(), z.item() - 1),
        (x.item(), y.item() + 1, z.item() + 1), (x.item(), y.item() - 1, z.item() - 1),
        (x.item() + 1, y.item() - 1, z.item()), (x.item() - 1, y.item() + 1, z.item()),
        (x.item() + 1, y.item(), z.item() - 1), (x.item() - 1, y.item(), z.item() + 1),
        (x.item(), y.item() + 1, z.item() - 1), (x.item(), y.item() - 1, z.item() + 1)
    ]

# Define the is_junction_point function to identify junction points
def is_junction_point(coord, skeleton):
    neighboring_coords = neighbors(coord)
    counter = 0
    for neigh in neighboring_coords:
        if skeleton[neigh[0], neigh[1], neigh[2]].item() > 0:
            counter = counter + 1
    
    if counter > 2:
        return True
    else:
        return False

    # skeleton_neighbors = sum([skeleton[tuple(neigh)] > 0 for neigh in neighboring_coords])
    # return skeleton_neighbors.item() > 2  # A junction point has more than 2 neighbors

# Define the main function to remove branches of type 1 based on the criteria
def remove_branches(skeleton, branch_data, skel, thickness_map, length_threshold, thickness_threshold):
    """
    Removes branches of type 1 from the skeleton based on length and thickness criteria,
    while preserving junction points.
    
    Parameters:
    skeleton (np.array): The 3D skeleton array to clean.
    branch_data (pd.DataFrame): DataFrame with branch information including branch type and distance.
    skel (Skeleton): Skeleton object to access branch coordinates.
    thickness_map (np.array): Array containing thickness values for each voxel.
    length_threshold (float): Maximum length for branches to consider for deletion.
    thickness_threshold (float): Minimum thickness for branches to consider for deletion.
    
    Returns:
    np.array: A cleaned version of the skeleton with specified branches removed.
    """
    # Copy the skeleton to avoid modifying the original
    skeleton_cleaned = np.copy(skeleton)
    
    # Iterate over each branch
    for branch_id in branch_data.index:
        # Only process branches of type 1
        if branch_data.loc[branch_id, 'branch-type'] == 1:
            branch_length = branch_data.loc[branch_id, 'branch-distance']
            
            # Check length condition
            if branch_length <= length_threshold:
                coordinates = skel.path_coordinates(branch_id)
                
                thickness_values = []
                for coord in coordinates:
                    thick_val = thickness_map[coord[0], coord[1], coord[2]]
                    thickness_values.append(thick_val.item())

                # Check thickness condition
                if min(thickness_values) <= thickness_threshold:
                    # Remove the branch, skipping junction points
                    for coord in coordinates:
                        # rounded_coord = tuple(np.round(coord).astype(int))
                        if not is_junction_point(coord, skeleton):
                            skeleton_cleaned[coord[0], coord[1], coord[2]] = 0
    
    return skeleton_cleaned


def print_branches(skeleton_cleaned):
    skel = Skeleton(skeleton_cleaned)
    branch_data = summarize(skel)

    branch_type_0 = branch_data[branch_data['branch-type'] == 0]
    branch_type_1 = branch_data[branch_data['branch-type'] == 1]
    branch_type_2 = branch_data[branch_data['branch-type'] == 2]
    branch_type_3 = branch_data[branch_data['branch-type'] == 3]

    print(f"Number of branch type 0: {len(branch_type_0)}")
    print(f"Number of branch type 1: {len(branch_type_1)}")
    print(f"Number of branch type 2: {len(branch_type_2)}")
    print(f"Number of branch type 3: {len(branch_type_3)}")
