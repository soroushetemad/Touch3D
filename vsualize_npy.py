import numpy as np
import open3d as o3d
import sys
import os

def visualize_point_cloud(npy_file_path):
    """
    Loads a point cloud from a .npy file and visualizes it using Open3D.

    Args:
        npy_file_path (str): The path to the .npy file containing point cloud data.
    """
    if not os.path.exists(npy_file_path):
        print(f"Error: File not found at {npy_file_path}")
        return

    try:
        # Load the data from the .npy file
        point_data = np.load(npy_file_path)

        # Check if the loaded data is suitable (e.g., N x 3 array for point cloud)
        # Files named with "_CV_" likely contain metrics (N x 2 array)
        is_point_cloud = isinstance(point_data, np.ndarray) and point_data.ndim == 2 and point_data.shape[1] == 3

        if not is_point_cloud:
            print(f"Info: The file {npy_file_path} does not appear to contain N x 3 point cloud data.")
            print(f"Data shape: {point_data.shape}, Data type: {type(point_data)}")
            # Check if it might be the metrics file (N x 2)
            if isinstance(point_data, np.ndarray) and point_data.ndim == 2 and point_data.shape[1] == 2:
                 print("\nThis might be a metrics history file (steps vs. coverage). Displaying array content:")
                 print(point_data)
                 # Optionally, you could add matplotlib plotting here for the metrics
            else:
                print("Unrecognized data format.")
            return

        print(f"Loaded {point_data.shape[0]} points from {npy_file_path}")

        # Create an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()

        # Assign the loaded points to the PointCloud object
        pcd.points = o3d.utility.Vector3dVector(point_data)

        # Optional: Add color to the points for better visibility
        pcd.paint_uniform_color([0.5, 0.7, 0.9]) # Light blue color

        # Visualize the point cloud
        print("Visualizing point cloud. Close the Open3D window to exit.")
        o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud: {os.path.basename(npy_file_path)}")

    except Exception as e:
        print(f"An error occurred while loading or visualizing {npy_file_path}: {e}")

if __name__ == "__main__":
    default_path = "outputs/DEMO_model_96.95.npy" # Default path to the .npy file

    # providing path as a command-line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_path
        print(f"No file path provided via command line. Using default: {default_path}")

    visualize_point_cloud(file_path)