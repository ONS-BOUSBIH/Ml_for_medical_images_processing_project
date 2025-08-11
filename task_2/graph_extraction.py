import os
import re
import argparse
import networkx as nx
from skimage.morphology import skeletonize
from utils_graph import (
    load_data,
    keep_two_components,
    skeleton_to_graph,
    separate_graph_into_components,
    simplify_graph,
    merge_close_nodes,
    set_root_node_highest_leaf,
    set_root_node_closest_leaf_to_center_node,
    set_root_node_clostest_leaf_to_center_high_point,
    direct_edges_from_root,
    merge_components,
    save_graph_as_json
)
# =====================================
#  File for graph extraiction
# =====================================

# =====================================
#  Helper Functions
# =====================================

def mask_to_graph(ct_scan, data_mask, image, merge_param, root_method):
    """
    Converts a segmentation mask into a graph representation.

    Args:
        ct_scan (numpy.ndarray): CT scan data.
        data_mask (numpy.ndarray): Segmentation mask.
        image (nibabel.Nifti1Image): Original image with affine transformation.
        merge_param (int): Parameter controlling node merging distance.
        root_method (str): Method to set the root node.

    Returns:
        networkx.Graph: The generated graph.
    """
    skeleton = skeletonize(data_mask, method='lee')
    skeleton = keep_two_components(skeleton)
    G = skeleton_to_graph(skeleton, image.affine)
    G1, G2 = separate_graph_into_components(G)

    # Simplify graphs and merge close nodes
    G1, G2 = simplify_graph(G1), simplify_graph(G2)
    G1, G2 = merge_close_nodes(G1, merge_param), merge_close_nodes(G2, merge_param)
    G1, G2 = simplify_graph(G1), simplify_graph(G2)

    # Set the root node based on user selection
    if root_method == "highest_leaf":
        G1, G2 = set_root_node_highest_leaf(G1), set_root_node_highest_leaf(G2)
    elif root_method == "closest_leaf_to_center":
        G1, G2 = set_root_node_closest_leaf_to_center_node(G1), set_root_node_closest_leaf_to_center_node(G2)
    elif root_method == "closest_leaf_to_center_high_point":
        G1, G2 = set_root_node_clostest_leaf_to_center_high_point(G1, ct_scan, image.affine), \
                 set_root_node_clostest_leaf_to_center_high_point(G2, ct_scan, image.affine)
    else:
        raise ValueError(f"Invalid root method: {root_method}")

    # Direct edges from root and merge components
    G1, G2 = direct_edges_from_root(G1), direct_edges_from_root(G2)
    G = merge_components(G1, G2)

    return G


def process_all_labels(image_folder, mask_folder, output_graph_folder, merge_param, root_method):
    """
    Process all segmentation masks in the given folder and generate graphs.

    Args:
        image_folder (str): Path to folder containing CT scans.
        mask_folder (str): Path to folder containing segmentation masks.
        output_graph_folder (str): Path where the graphs will be saved.
        merge_param (int): Parameter controlling node merging distance.
        root_method (str): Method to set the root node.
    """
    # Ensure the output folder exists
    os.makedirs(output_graph_folder, exist_ok=True)

    # Regular expression to match .label.nii.gz files and extract file_id
    label_pattern = re.compile(r"(.+)\.label\.nii\.gz$", re.IGNORECASE)

    # Process all label files in the folder
    for file in os.listdir(mask_folder):
        match = label_pattern.match(file.strip())
        if match:
            file_id = match.group(1)
            print(f"Processing {file_id}...")

            try:
                # Load CT scan and mask
                ct_scan, data_mask, image = load_data(file_id, image_folder, mask_folder)

                # Convert mask to graph
                G = mask_to_graph(ct_scan, data_mask, image, merge_param, root_method)

                # Save the graph as JSON
                save_graph_as_json(G, file_id, output_graph_folder)

            except Exception as e:
                print(f"Error processing {file_id}: {e}")

    print("Processing completed.")


# =====================================
#  Main Execution
# =====================================

def main():
    """
    Main function to process medical image segmentations and generate graphs.
    Uses argument parsing for flexibility.
    """
    parser = argparse.ArgumentParser(description="Process NIfTI segmentations and generate graphs.")

    # Folder path arguments with default values from the original version
    parser.add_argument(
        "--image_folder", 
        type=str, 
        default="/data/training_data", 
        help="Path to folder containing CT scans. (Default: /data/training_data)"
    )
    parser.add_argument(
        "--mask_folder", 
        type=str, 
        default="./../../research-contributions/UNETR/BTCV/training_from_scratch/output_segmentations_for_graph_w_sol_no_pp", 
        help="Path to folder containing segmentation masks. (Default: original path)"
    )
    parser.add_argument(
        "--output_graph_folder", 
        type=str, 
        default="./graph_test", 
        help="Folder to save the output graph JSONs. (Default: ./graph_test)"
    )

    # Processing parameters
    parser.add_argument(
        "--merge_param", 
        type=int, 
        default=5, 
        help="Parameter controlling node merging distance. (Default: 5)"
    )
    parser.add_argument(
        "--root_method", 
        type=str, 
        choices=["highest_leaf", "closest_leaf_to_center", "closest_leaf_to_center_high_point"],
        default="closest_leaf_to_center_high_point", 
        help="Method for setting the root node. (Default: closest_leaf_to_center_high_point)"
    )

    args = parser.parse_args()

    process_all_labels(
        args.image_folder, 
        args.mask_folder, 
        args.output_graph_folder, 
        args.merge_param, 
        args.root_method
    )


if __name__ == "__main__":
    main()