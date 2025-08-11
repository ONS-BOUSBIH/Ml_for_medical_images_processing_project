import os
import json
import numpy as np
import networkx as nx
import torch
import nibabel as nib
from monai.transforms import KeepLargestConnectedComponent

# =====================================
#  untils file for graph extraiction
# =====================================


def load_data(file_id, image_folder, mask_folder):
    """
    Loads the CT scan and corresponding mask using a given file ID.

    Parameters:
        file_id (str): The identifier for the image and mask files (e.g., '093434').
        image_folder (str): Path to the folder containing CT scan images.
        mask_folder (str): Path to the folder containing segmentation masks.

    Returns:
        tuple: (CT scan data as a NumPy array, mask data as a NumPy array, nibabel image object)
    """
    # Construct absolute file paths
    image_path = os.path.join(image_folder, f"{file_id}.img.nii.gz")
    mask_path = os.path.join(mask_folder, f"{file_id}.label.nii.gz")

    # Check if files exist before loading
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"CT scan file not found: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    # Load CT scan
    image = nib.load(image_path)
    ct_scan = image.get_fdata()

    # Load mask
    mask = nib.load(mask_path)
    data_mask = mask.get_fdata()

    return ct_scan, data_mask, image


def keep_two_components(skeleton):
    """
    Keeps the two largest connected components in a 3D skeletonized volume.

    Parameters:
        skeleton (numpy.ndarray): The input 3D volume as a NumPy array.

    Returns:
        numpy.ndarray: The processed volume with only the two largest components.
    """
    # Convert numpy array to PyTorch tensor
    skeleton = torch.from_numpy(skeleton)

    # Add channel dimension if needed (shape becomes (1, depth, height, width))
    skeleton = skeleton.unsqueeze(0)

    # Apply the KeepLargestConnectedComponent transform to retain 2 components
    keep_largest = KeepLargestConnectedComponent(is_onehot=False, connectivity=3, num_components=2)
    skeleton = keep_largest(skeleton)

    # Remove channel dimension to return to original shape
    skeleton = skeleton.squeeze(0)

    # Convert back to NumPy array
    skeleton_np = skeleton.cpu().numpy()
    
    return skeleton_np  # Ensure returning NumPy array



def skeleton_to_graph(skeleton, image_affine):
    """
    Convert a 3D skeleton to a directed JSON-formatted graph with world coordinates.

    Args:
        skeleton (np.ndarray): Binary 3D skeleton array.
        image_affine (np.ndarray): Affine transformation matrix of the image (4x4).
    """
 
    G = nx.Graph()

    # Step 1: Extract nodes from skeleton points
    node_id = 0
    node_map = {}  # Maps (z, y, x) -> node_id
    
    for x, y, z in zip(*np.nonzero(skeleton)):
        pos_voxel = [float(x), float(y), float(z)]
        pos_world = voxel_to_world(pos_voxel, image_affine)  # Convert to world coordinates
        G.add_node(node_id, pos=pos_world, is_root=False)  # Default: not root
        node_map[(x, y, z)] = node_id
        node_id += 1

    # Step 2 Add egdes
    offsets = [
        (x, y, z) 
    for x in [-1, 0, 1] 
    for y in [-1, 0, 1] 
    for z in [-1, 0, 1] 
    if not (x == 0 and y == 0 and z == 0)
    ]
    for x, y, z in node_map.keys():
        source_id = node_map[(x, y, z)]
        for dx, dy, dz in offsets:
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor in node_map:
                target_id = node_map[neighbor]
                # Calculate Euclidean distance in world space
                pos1_world = voxel_to_world([x, y, z], image_affine)
                pos2_world = voxel_to_world([x + dx, y + dy, z + dz], image_affine)
                length = np.linalg.norm(np.array(pos2_world) - np.array(pos1_world))

                G.add_edge(
                    source_id,
                    target_id,
                    length=length,
                    skeletons=[pos1_world, pos2_world],  # Store world coordinates
                )
            
    return G


def voxel_to_world(voxel_pos, affine_matrix):
    """
    A function that transform voxel coordinates to world coordinates

    """
   
    # Convert voxel coordinates to homogeneous coordinates (add a 1 for the homogeneous coordinate)
    voxel_h = np.array([voxel_pos[0], voxel_pos[1], voxel_pos[2], 1])
    
    # Apply the affine transformation
    world_h = affine_matrix @ voxel_h
   
    
    # Return world coordinates, excluding the homogeneous coordinate (the last element)
    return [ world_h[0], world_h[1], world_h[2]]


def separate_graph_into_components(G):
    """
    Separates a graph into the two largest connected components.

    Parameters:
        G (networkx.Graph): The input graph.

    Returns:
        tuple: Two subgraphs representing the largest connected components.
    """
    # Find all connected components of the graph and sort by size (descending)
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    
    # Ensure there are at least two components
    if len(components) < 2:
        raise ValueError(f"The graph does not have enough connected components. It has {len(components)}")

    # Get the two largest components
    G1 = G.subgraph(components[0]).copy()
    G2 = G.subgraph(components[1]).copy()
    
    if len(components) > 2:
        G3 = G.subgraph(components[2]).copy()
        #print_graph_connections(G3)

    return G1, G2


def simplify_graph(G):
    """
    Simplify the graph by removing all nodes of degree 2 and connecting their neighbors directly.
    The new edge length is the sum of the removed edges' lengths.

    Parameters:
        G (networkx.Graph): The input graph.
        
    Returns:
        networkx.Graph: The simplified graph.
    """
    G = G.copy()  # Work on a copy to avoid modifying the original graph

    # Continue processing until no nodes with degree 2 remain
    while True:
        degree_2_nodes = [node for node in G.nodes() if G.degree(node) == 2]
        
        if not degree_2_nodes:
            break  # Exit the loop if no degree-2 nodes are left

        for node in degree_2_nodes:
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 2:
                neighbor1, neighbor2 = neighbors

                # Calculate the sum of the edge lengths of the removed node
                length1 = G.edges[node, neighbor1].get('length', 0)
                length2 = G.edges[node, neighbor2].get('length', 0)
                new_length = length1 + length2

                # Connect the two neighbors with the accumulated length if not already connected
                if not G.has_edge(neighbor1, neighbor2):
                    G.add_edge(neighbor1, neighbor2, length=new_length)

                # Remove the current degree-2 node
                G.remove_node(node)
    return G

                
def merge_close_nodes(G, threshold):
    """
    Merges nodes in a graph if the edge length is below the given threshold.
    The new node position is the mean of both nodes and stored as a Python list.
    The length of the deleted edge is equally distributed among the new edges.

    Parameters:
        G (networkx.Graph): The input graph with node positions stored in 'pos' attribute.
        threshold (float): Distance threshold below which nodes will be merged.

    Returns:
        networkx.Graph: The simplified graph with merged nodes.
    """
    G = G.copy()  # Work on a copy to preserve the original graph
    merged = True  # Flag to keep track of merging operations

    while merged:
        merged = False  # Assume no merge initially

        # Iterate through edges and check their lengths from the 'length' attribute
        for u, v, data in list(G.edges(data=True)):
            length = data.get('length', 0)  # Get the edge length from attributes, default to 0

            if length < threshold:
                # Compute new position as mean of both node positions and convert to Python list
                pos_u = G.nodes[u]['pos']
                pos_v = G.nodes[v]['pos']
                new_pos = [(pos_u[i] + pos_v[i]) / 2 for i in range(len(pos_u))]  # Calculate mean position

                # Create new merged node with position stored as a list
                new_node = f"{u}_{v}"  # Unique identifier for merged nodes
                G.add_node(new_node, pos=new_pos)

                # Get all neighbors excluding the nodes being merged
                neighbors = set(G.neighbors(u)).union(set(G.neighbors(v)))
                neighbors.discard(u)
                neighbors.discard(v)

                # Distribute the removed edge length equally to the new edges
                num_neighbors = len(neighbors)
                if num_neighbors > 0:
                    additional_length = length / num_neighbors
                else:
                    additional_length = 0

                for neighbor in neighbors:
                    # Get the original edge lengths from u and v to their respective neighbors
                    length_u_neighbor = G[u][neighbor].get('length', 0) if G.has_edge(u, neighbor) else 0
                    length_v_neighbor = G[v][neighbor].get('length', 0) if G.has_edge(v, neighbor) else 0

                    # Add the distributed length to the existing edge length
                    new_edge_length = length_u_neighbor + length_v_neighbor + additional_length

                    G.add_edge(new_node, neighbor, length=new_edge_length)

                # Remove old nodes and edges
                G.remove_node(u)
                G.remove_node(v)

                merged = True  # Mark that a merge occurred
                break  # Restart iteration after merging

    return G


def set_root_node_clostest_leaf_to_center_high_point(G, ct_scan, image_affine):
    """
    Finds the degree-1 node closest to the center of the highest slice in the CT scan
    and marks it as the root.

    Parameters:
        G (networkx.Graph): The input graph with node positions stored in 'pos_world'.
        ct_scan (numpy.ndarray): 3D CT scan volume.
        voxel_to_world (function): Function to convert voxel coordinates to world coordinates.
        image_affine (numpy.ndarray): Affine transformation matrix for voxel-to-world conversion.

    Returns:
        int: The root node ID that was marked as root.
    """
    # Step 1: Find the highest slice in the CT scan (non-zero slice)
    non_empty_slices = np.where(ct_scan.any(axis=(0, 1)))[0]
    highest_slice_index = np.max(non_empty_slices)  # Get the highest non-empty slice

    # Step 2: Find the center of the highest slice (in voxel coordinates)
    highest_slice_voxels = np.argwhere(ct_scan[:, :, highest_slice_index] > 0)
    center_voxel = np.mean(highest_slice_voxels, axis=0)
    center_voxel_3d = np.array([center_voxel[0], center_voxel[1], highest_slice_index])

    # Convert center voxel to world coordinates
    center_world = voxel_to_world(center_voxel_3d, image_affine)

    # Step 3: Find all leaf (degree-1) nodes
    leaf_nodes = [node for node in G.nodes if G.degree(node) == 1]

    if not leaf_nodes:
        raise ValueError("No leaf nodes (degree 1) found in the graph.")

    # Step 4: Find the closest leaf node to the center of the highest slice
    min_distance = float('inf')
    root_node = None

    for node in leaf_nodes:
        node_pos = np.array(G.nodes[node]['pos'])
        distance = np.linalg.norm(node_pos - center_world)
        if distance < min_distance:
            min_distance = distance
            root_node = node

    # Step 5: Mark the closest leaf node as root
    nx.set_node_attributes(G, False, 'is_root')  # Reset all root values to False
    G.nodes[root_node]['is_root'] = True

    return G


def set_root_node_highest_leaf(G):
    """
    Finds the degree-1 node (leaf node) with the highest z-coordinate in the graph
    and marks it as the root.

    Parameters:
        G (networkx.Graph): The input graph with node positions stored in 'pos'.

    Returns:
        networkx.Graph: The graph with the highest leaf node marked as the root.
    """
    # Step 1: Find all leaf (degree-1) nodes
    leaf_nodes = [node for node in G.nodes if G.degree(node) == 1]

    if not leaf_nodes:
        raise ValueError("No leaf nodes (degree 1) found in the graph.")

    # Step 2: Identify the leaf node with the highest z-coordinate
    root_node = max(leaf_nodes, key=lambda node: G.nodes[node]['pos'][2])

    # Step 3: Mark the highest leaf node as root
    nx.set_node_attributes(G, False, 'is_root')  # Reset all root values to False
    G.nodes[root_node]['is_root'] = True

    return G


def set_root_node_closest_leaf_to_center_node(G):
    """
    1) Finds the tree center(s) of a weighted tree (edge attribute 'length').
    2) From the center(s), finds the closest leaf (degree=1) node in terms of
       path distance (summing 'length' along edges).
    3) Marks that leaf node as 'is_root' = True in G and returns it.

    Parameters
    ----------
    G : networkx.Graph
        A tree (or tree-like) graph where each edge has a 'length' attribute.

    Returns
    -------
    root_node : Node
        The closest leaf node to the center(s), marked as 'is_root' = True.
    center_nodes : list
        The node(s) that form the tree center.
    """

    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty; cannot find a center or root.")

    # 1) Find tree center(s)
    # Step A) Find the diameter endpoints via two Dijkstra calls
    leaves = [n for n in G.nodes if G.degree(n) == 1]
    if not leaves:
        raise ValueError("No leaf (degree=1) nodes found. Ensure G is a tree.")

    start_node = leaves[0]
    dist_from_start = nx.single_source_dijkstra_path_length(G, start_node, weight='length')
    # farthest node from the start_node:
    end1, _ = max(dist_from_start.items(), key=lambda x: x[1])

    dist_from_end1, paths_from_end1 = nx.single_source_dijkstra(G, end1, weight='length')
    end2, diameter_length = max(dist_from_end1.items(), key=lambda x: x[1])
    diameter_path = paths_from_end1[end2]

    # Step B) Identify the center node(s) along the diameter path
    half_len = diameter_length / 2.0
    cumulative_dist = 0.0
    center_nodes = []

    for i in range(len(diameter_path) - 1):
        current_node = diameter_path[i]
        next_node = diameter_path[i + 1]
        edge_len = G[current_node][next_node]['length']

        if cumulative_dist <= half_len < (cumulative_dist + edge_len):
            # The halfway point is within or at the boundary of this edge
            within_edge_offset = half_len - cumulative_dist

            # If exactly on current_node:
            if abs(within_edge_offset) < 1e-9:
                center_nodes = [current_node]
            # If exactly on next_node:
            elif abs((cumulative_dist + edge_len) - half_len) < 1e-9:
                center_nodes = [next_node]
            else:
                # Halfway is strictly inside the edge => "two-center" case in continuous sense
                center_nodes = [current_node, next_node]
            break

        cumulative_dist += edge_len
    else:
        # If we never broke, it means half_len is at the very end => the center is end2
        center_nodes = [diameter_path[-1]]

    # 2) From the center(s), find the closest leaf
    # We have 1 or 2 center nodes (in most cases). We'll pick the leaf that is
    # minimal in distance from ANY of these centers.
    all_leaves = [n for n in G.nodes if G.degree(n) == 1]
    if not all_leaves:
        raise ValueError("No leaves in the graph, cannot pick a 'root' from leaves.")

    best_leaf = None
    best_dist = float('inf')

    for c in center_nodes:
        dist_dict = nx.single_source_dijkstra_path_length(G, c, weight='length')
        for leaf in all_leaves:
            d = dist_dict[leaf]
            if d < best_dist:
                best_dist = d
                best_leaf = leaf

    # 3) Mark the best_leaf as root
    nx.set_node_attributes(G, False, 'is_root')
    if best_leaf is not None:
        G.nodes[best_leaf]['is_root'] = True

    return G


def direct_edges_from_root(G):
    """
    Directs the graph starting from the root node
    
    Parameters:
        G (networkx.Graph): The undirected input graph.

    Returns:
        networkx.Graph: The graph directed sarting from the root node.
    """
    # Find root node
    root = None
    for node, data in G.nodes(data=True):
        if data.get('is_root', False):
            root = node
            break

    if root is None:
        raise ValueError("No root node found. Ensure that a node has the 'is_root' attribute set to True.")
    
    # Create new graph and set for visited nodes
    directed_G = nx.DiGraph()
    visited = set()

    def dfs(node):
        visited.add(node)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                directed_G.add_edge(node, neighbor)
                dfs(neighbor)
            elif not directed_G.has_edge(neighbor, node):  # Prevent duplicates
                directed_G.add_edge(neighbor, node)

    # DFS strating from the root node
    dfs(root)

    # Fill new graph
    for node, data in G.nodes(data=True):
        directed_G.add_node(node, **data)

    for u, v in directed_G.edges():
        if G.has_edge(u, v):
            directed_G.edges[u, v].update(G.edges[u, v])

    return directed_G


def merge_components(G1, G2):
    """
    Merges two separate connected components (graphs) into a single graph without adding an edge.

    Parameters:
        G1 (networkx.Graph): The first connected component.
        G2 (networkx.Graph): The second connected component.

    Returns:
        networkx.Graph: A single graph containing all nodes and edges from both components.
    """
    # Create a new graph that combines both components
    merged_graph = nx.Graph()

    # Add nodes and edges from both graphs into the new graph
    merged_graph.add_nodes_from(G1.nodes(data=True))
    merged_graph.add_edges_from(G1.edges(data=True))

    merged_graph.add_nodes_from(G2.nodes(data=True))
    merged_graph.add_edges_from(G2.edges(data=True))

    return merged_graph


def graph_to_json(G):
    """
    Converts a graph to JSON format with world coordinates
    
    Parameters:
        G (networkx.Graph): The input graph.

    Returns:
        networkx.Graph: The graph with the highest leaf node marked as the root.
    """
    
    json_data = {
        "directed": True,
        "multigraph": False,
        "graph": {"coordinateSystem": "RAS"},
       "nodes": [
        {
            "id": int(n),
            "pos": data["pos"],
            "is_root": data.get("is_root", False),  # Safely retrieve the is_root attribute
        }
        for n, data in G.nodes(data=True)
    ],
        "edges": [
            {
                "source": int(u),
                "target": int(v),
                "length": float(data["length"]),
                #"skeletons": data["skeletons"],
            }
            for u, v, data in G.edges(data=True)
        ],
    }

    return json_data


def save_graph_as_json(G, file_id, output_folder):
    """
    Converts a NetworkX graph to JSON and saves it to a specified folder.

    Parameters:
        G (networkx.Graph): The input graph.
        file_id (str): The identifier used for the filename (e.g., 'c6f3ac').
        output_folder (str): The folder where the JSON file should be saved.

    Returns:
        str: The path of the saved JSON file.
    """

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Generate the output filename
    filename = f"{file_id}.graph.json"
    output_path = os.path.join(output_folder, filename)

    # Convert the graph to JSON format
    graph_json = graph_to_json(G)  # Converts graph to JSON-compatible format

    # Save JSON to file
    with open(output_path, 'w') as json_file:
        json.dump(graph_json, json_file, indent=4)

    print(f"Graph saved to: {output_path}")
    return graph_json